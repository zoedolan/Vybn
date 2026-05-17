"""Tests for the bounded MCP capability-truth repair in spark/server.py.

Why a new test home (the single new-file exception in this patch):
    The other modules in `spark/tests/` exercise routing, chat-API,
    harness, contracts, and refactors — none import `spark/server.py`
    or test its MCP handlers. The handlers are the surface this patch
    repairs (deep_search live-API truth, /health capability split,
    model_status role mirroring), so the assertions only belong here.
    Putting them in any of the existing files would require importing
    `server.py` into a module whose subject is something else. One new
    file scoped exactly to server-handler truth is the minimal addition.

Covers:
  - handle_deep_search consults the live loopback memory API first.
  - The stale ~/vybn-phase/deep_memory.py path is no longer referenced
    as active code anywhere in the module.
  - On live-API failure, deep_search falls back to the absorbed Him
    phase path or fails closed honestly — never silently improvises.
  - The Him-phase fallback shell command contains an absolute path
    (the tilde is expanded against the Spark user's home); it does
    NOT pass a single-quoted `~/...` literal that bash would refuse
    to expand.
  - /health distinguishes gateway liveness from deep_search capability
    (live | stale-fallback | fail-closed) and does not overclaim "ok"
    for the whole MCP surface.
  - /health body leaks no IP, no http(s) URL, no known internal port
    or hostname.
  - model_status treats the Him capability mirror as a non-authoritative
    projection of `Him/spark/runtime.py:FLEET_COMPONENTS`. The mirror
    can be absent, malformed, or hostile; in every case the local
    fail-closed defaults hold and Omni / Vintage stay unpromoted.

Run: python3 spark/tests/test_mcp_capability_truth.py
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

THIS = Path(__file__).resolve()
SPARK_DIR = THIS.parent.parent
sys.path.insert(0, str(SPARK_DIR))


def _load_server():
    os.environ.setdefault("VYBN_MCP_API_KEY", "test-key-not-real")
    os.environ.setdefault("SPARK1_HOST", "127.0.0.1")
    os.environ.setdefault("SPARK1_USER", "vybn")
    if "server" in sys.modules:
        del sys.modules["server"]
    return importlib.import_module("server")


server = _load_server()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class FakeSSH:
    """Stand-in for run_ssh that records every command and returns
    scripted results keyed by substring match."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.responses: list = []
        self.default = {"stdout": "", "stderr": "", "exit_code": 1}

    def add(self, match: str, response: dict):
        self.responses.append((match, response))

    async def __call__(self, machine: str, command: str, timeout: int = 60):
        self.calls.append((machine, command))
        for match, resp in self.responses:
            if match in command:
                return resp
        return dict(self.default)


class DeepSearchLiveAPITests(unittest.TestCase):
    def test_uses_live_api_first(self):
        fake = FakeSSH()
        api_body = json.dumps({
            "results": [
                {"source": "a.txt", "text": "hello", "fidelity": 0.9, "telling": 0.5, "distinctiveness": 0.3},
            ]
        })
        fake.add("__HTTP__", {
            "stdout": api_body + "\n__HTTP__:200",
            "stderr": "",
            "exit_code": 0,
        })
        with mock.patch.object(server, "run_ssh", fake):
            out = _run(server.handle_deep_search({"query": "what is x"}))
        self.assertFalse(out.get("isError"))
        text = out["content"][0]["text"]
        self.assertIn("[live deep_search]", text)
        self.assertIn("a.txt", text)
        # The first attempted command must be the live API call.
        self.assertTrue(any("__HTTP__" in c[1] for c in fake.calls),
                        "live API call expected")
        # The stale phase path must NOT have been used.
        for _, cmd in fake.calls:
            self.assertNotIn("~/vybn-phase/deep_memory.py", cmd)
            self.assertNotIn("vybn-phase && python3 deep_memory.py", cmd)

    def test_falls_back_to_him_phase_when_api_down(self):
        fake = FakeSSH()
        # API call: exit_code 0 but HTTP 500 -> treated as unavailable.
        fake.add("__HTTP__", {
            "stdout": "boom\n__HTTP__:500",
            "stderr": "",
            "exit_code": 0,
        })
        # Him fallback returns JSON list.
        him_payload = json.dumps([
            {"source": "him.md", "text": "fallback", "fidelity": 0.7, "telling": 0.4, "distinctiveness": 0.2},
        ])
        fake.add("Him/spark/phase/deep_memory.py", {
            "stdout": him_payload,
            "stderr": "",
            "exit_code": 0,
        })
        with mock.patch.object(server, "run_ssh", fake):
            out = _run(server.handle_deep_search({"query": "again"}))
        self.assertFalse(out.get("isError"))
        text = out["content"][0]["text"]
        self.assertIn("absorbed Him phase fallback", text)
        self.assertIn("him.md", text)
        # Stale path still must not appear.
        for _, cmd in fake.calls:
            self.assertNotIn("~/vybn-phase/deep_memory.py", cmd)

    def test_him_phase_fallback_command_uses_absolute_path(self):
        """Regression: the Him fallback used to pass a single-quoted
        `~/Him/...` literal to the Spark's bash, which never expands.
        The command actually issued must contain an absolute path
        (e.g. /home/vybn/Him/spark/phase/deep_memory.py) and must NOT
        contain a `~/` literal — single quotes prevent bash expansion."""
        fake = FakeSSH()
        # Force both arms to be attempted so we can inspect the Him command.
        fake.add("__HTTP__", {"stdout": "", "stderr": "down", "exit_code": 7})
        fake.add("deep_memory.py", {"stdout": "[]", "stderr": "", "exit_code": 0})
        with mock.patch.object(server, "run_ssh", fake):
            _run(server.handle_deep_search({"query": "probe"}))
        him_calls = [c[1] for c in fake.calls if "deep_memory.py" in c[1]]
        self.assertTrue(him_calls, "Him fallback command was not issued")
        cmd = him_calls[0]
        # Must reference an absolute home path for the configured user.
        self.assertIn("/home/vybn/Him/spark/phase/deep_memory.py", cmd)
        # Must NOT contain a literal `~/` anywhere — single-quoted tildes
        # do not expand in bash, so this would silently fail in production.
        self.assertNotIn("~/", cmd)

    def test_fails_closed_when_both_paths_down(self):
        fake = FakeSSH()
        fake.add("__HTTP__", {"stdout": "", "stderr": "curl: connection refused", "exit_code": 7})
        fake.add("deep_memory.py", {"stdout": "", "stderr": "no file", "exit_code": 1})
        with mock.patch.object(server, "run_ssh", fake):
            out = _run(server.handle_deep_search({"query": "anything"}))
        self.assertTrue(out.get("isError"))
        self.assertIn("unavailable", out["content"][0]["text"].lower())
        self.assertIn("failing closed", out["content"][0]["text"].lower())

    def test_validates_query_and_k(self):
        out = _run(server.handle_deep_search({"query": "   "}))
        self.assertTrue(out.get("isError"))
        out = _run(server.handle_deep_search({"query": "ok", "k": "not-an-int"}))
        self.assertTrue(out.get("isError"))
        out = _run(server.handle_deep_search({"query": "ok", "k": 0}))
        self.assertTrue(out.get("isError"))


class StaleDeepSearchPathReferenceTests(unittest.TestCase):
    def test_stale_path_not_active_code(self):
        """The stale ~/vybn-phase/deep_memory.py path is not used as an
        active execution path anywhere in spark/server.py. Comments may
        legitimately mention it, but no run-able command should target it."""
        src = (SPARK_DIR / "server.py").read_text(encoding="utf-8")
        bad = []
        for lineno, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            if "vybn-phase/deep_memory.py" in code or "vybn-phase && python3 deep_memory.py" in code:
                bad.append((lineno, line))
        self.assertEqual(bad, [], f"stale deep_memory.py path referenced as code: {bad}")


class HealthCapabilityTests(unittest.TestCase):
    def test_health_separates_gateway_from_deep_search(self):
        async def probe_live():
            return {"state": "live", "via": "loopback-memory-api"}
        with mock.patch.object(server, "_probe_deep_search_capability", probe_live):
            from starlette.testclient import TestClient
            client = TestClient(server.app)
            resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("gateway", data)
        self.assertTrue(data["gateway"]["alive"])
        self.assertIn("capabilities", data)
        self.assertIn("deep_search", data["capabilities"])
        self.assertEqual(data["capabilities"]["deep_search"]["state"], "live")
        # /health rollup describes only what was probed — gateway-alive
        # is the precise top-level statement, not an "ok" overclaim.
        self.assertEqual(data["status"], "gateway-alive")
        self.assertEqual(data["probed_capabilities_state"], "live")

    def test_health_reports_fail_closed_without_promoting(self):
        async def probe_fc():
            return {"state": "fail-closed", "reason": "no-path-answered"}
        with mock.patch.object(server, "_probe_deep_search_capability", probe_fc):
            from starlette.testclient import TestClient
            client = TestClient(server.app)
            resp = client.get("/health")
        data = resp.json()
        self.assertTrue(data["gateway"]["alive"])
        self.assertEqual(data["capabilities"]["deep_search"]["state"], "fail-closed")
        # Top-level rollup remains "gateway-alive" — it never asserts
        # whole-surface health. Capability state is the separate axis.
        self.assertEqual(data["status"], "gateway-alive")
        self.assertEqual(data["probed_capabilities_state"], "fail-closed")

    def test_health_does_not_leak_topology(self):
        async def probe_live():
            return {"state": "live", "via": "loopback-memory-api"}
        with mock.patch.object(server, "_probe_deep_search_capability", probe_live):
            from starlette.testclient import TestClient
            client = TestClient(server.app)
            resp = client.get("/health")
        body = resp.text
        # Hand-rolled forbidden list (known internal coordinates).
        for forbidden in ("tailscale", "ts.net", "8100", "8420", "127.0.0.1"):
            self.assertNotIn(forbidden, body, f"leaked '{forbidden}' in /health body")
        # Generic shape checks: no dotted-quad IP, no http(s) URL.
        self.assertIsNone(re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", body),
                          f"/health leaked an IP-shaped string: {body!r}")
        self.assertIsNone(re.search(r"https?://", body),
                          f"/health leaked a URL: {body!r}")

    def test_health_probe_is_bounded(self):
        """If the probe hangs, /health must time out cleanly without blocking
        the gateway response. A probe-budget timeout means we did not learn
        whether the tool is live — it is NOT evidence that the tool refused.
        The capability state must report `indeterminate` (with reason
        `probe-budget-exceeded`), not `fail-closed`, so /health stays
        coherent with tool truth instead of overclaiming a refusal."""
        async def slow_probe():
            await asyncio.sleep(10)
            return {"state": "live", "via": "loopback-memory-api"}
        # Patch the inner probe call and tighten the budget for the test.
        with mock.patch.object(server, "_probe_memory_api_readiness", lambda m: slow_probe()), \
             mock.patch.object(server, "_deep_search_via_him_phase", lambda *a, **kw: slow_probe()), \
             mock.patch.object(server, "HEALTH_PROBE_BUDGET_S", 0.1):
            from starlette.testclient import TestClient
            client = TestClient(server.app)
            resp = client.get("/health")
        data = resp.json()
        self.assertEqual(data["capabilities"]["deep_search"]["state"], "indeterminate")
        self.assertEqual(data["capabilities"]["deep_search"].get("reason"),
                         "probe-budget-exceeded")
        # Rollup must mirror the per-capability state — not be flattened
        # to "fail-closed" merely because the probe ran out of time.
        self.assertEqual(data["probed_capabilities_state"], "indeterminate")
        # Gateway liveness is the separate axis and remains alive.
        self.assertTrue(data["gateway"]["alive"])
        self.assertEqual(data["status"], "gateway-alive")

    def test_readiness_witness_uses_search_endpoint_not_healthz(self):
        """The deep_search readiness witness must hit the same loopback
        endpoint the tool itself uses (POST /search), not a separate
        liveness route. Live observation: the memory service exposes
        /health and /search but not /healthz — so a /healthz pre-check
        404s, the probe falls through to the Him-phase fallback, and
        /health reports `stale-fallback` even though the real tool is
        answering live. Pin: the issued shell command must be a POST
        to DEEP_SEARCH_API_URL with a minimal `__health__` body, and a
        2xx response must yield state=live."""
        fake = FakeSSH()
        # Witness command: POST /search with __health__ body returns 200.
        fake.add("__health__", {
            "stdout": "200",
            "stderr": "",
            "exit_code": 0,
        })
        with mock.patch.object(server, "run_ssh", fake):
            ready = _run(server._probe_memory_api_readiness("spark-1"))
        self.assertTrue(ready, "2xx from POST /search must be witnessed as live")
        # Exactly one SSH command was issued, and it must:
        #   - POST (not GET) to the search endpoint the tool itself uses
        #   - carry the minimal __health__ probe body with k=1
        #   - NOT reach for /healthz, which the memory service does not expose
        self.assertEqual(len(fake.calls), 1)
        cmd = fake.calls[0][1]
        self.assertIn("-X POST", cmd)
        self.assertIn("__health__", cmd)
        self.assertIn('"k": 1', cmd)
        self.assertIn(server.DEEP_SEARCH_API_URL, cmd)
        self.assertNotIn("/healthz", cmd)

    def test_health_live_via_search_witness_does_not_fall_back(self):
        """End-to-end: when the search-witness succeeds, /health reports
        state=live via loopback-memory-api and never invokes the Him-phase
        fallback. This is the case that was misreporting `stale-fallback`
        live — pin it shut."""
        fake = FakeSSH()
        fake.add("__health__", {"stdout": "200", "stderr": "", "exit_code": 0})
        # If the fallback is ever called we'd see "deep_memory.py" in calls;
        # we explicitly do NOT register a response for it. The default
        # response (exit_code 1) would still cause `state=fail-closed`,
        # so a live witness must short-circuit before that path.
        with mock.patch.object(server, "run_ssh", fake):
            from starlette.testclient import TestClient
            client = TestClient(server.app)
            resp = client.get("/health")
        data = resp.json()
        self.assertEqual(data["capabilities"]["deep_search"]["state"], "live")
        self.assertEqual(data["capabilities"]["deep_search"].get("via"),
                         "loopback-memory-api")
        self.assertEqual(data["probed_capabilities_state"], "live")
        # Fallback path must not have been touched on the live happy path.
        for _, cmd in fake.calls:
            self.assertNotIn("deep_memory.py", cmd)


class ModelStatusRoleTests(unittest.TestCase):
    """Him capability projections are non-authoritative and gate local organs."""

    def test_fail_closed_when_mirror_absent(self):
        """Mirror file does not exist on the Spark → local fail-closed
        defaults; Omni and Vintage stay unpromoted; Super stays promoted;
        the source label reports the non-authoritative posture."""
        fake = FakeSSH()
        fake.add("LLAMA SERVER", {"stdout": "llama running", "stderr": "", "exit_code": 0})
        fake.add("Him/spark/capability.json", {"stdout": "", "stderr": "no file", "exit_code": 1})
        with mock.patch.object(server, "run_ssh", fake):
            out = _run(server.handle_model_status({}))
        text = out["content"][0]["text"]
        self.assertIn("=== ROLES ===", text)
        self.assertIn("local-fail-closed-defaults", text)
        self.assertIn("no mirror present", text)
        omni_line = next(l for l in text.splitlines() if l.startswith("omni:"))
        vintage_line = next(l for l in text.splitlines() if l.startswith("vintage:"))
        self.assertIn("promoted=False", omni_line)
        self.assertIn("fail_closed=True", omni_line)
        self.assertIn("promoted=False", vintage_line)
        self.assertIn("fail_closed=True", vintage_line)
        super_line = next(l for l in text.splitlines() if l.startswith("super:"))
        self.assertIn("promoted=True", super_line)

    def test_fail_closed_when_mirror_malformed(self):
        """Mirror file exists but contains non-JSON garbage → treated as
        absent; local fail-closed defaults apply; Omni and Vintage are
        not promoted; source label reports no mirror present."""
        fake = FakeSSH()
        fake.add("LLAMA SERVER", {"stdout": "", "stderr": "", "exit_code": 0})
        fake.add("Him/spark/capability.json", {
            "stdout": "<<<not json>>>",
            "stderr": "",
            "exit_code": 0,
        })
        with mock.patch.object(server, "run_ssh", fake):
            out = _run(server.handle_model_status({}))
        text = out["content"][0]["text"]
        self.assertIn("local-fail-closed-defaults", text)
        omni_line = next(l for l in text.splitlines() if l.startswith("omni:"))
        vintage_line = next(l for l in text.splitlines() if l.startswith("vintage:"))
        self.assertIn("promoted=False", omni_line)
        self.assertIn("promoted=False", vintage_line)

    def test_hostile_mirror_cannot_promote_without_complete_gate(self):
        """A mirror that claims Omni/Vintage are promoted without the complete
        promotion chain is locally re-sanitized: promoted=False,
        fail_closed=True. Endpoint liveness alone is not promotion."""
        fake = FakeSSH()
        fake.add("LLAMA SERVER", {"stdout": "", "stderr": "", "exit_code": 0})
        evil = json.dumps({
            "roles": {
                "omni": {"promoted": True, "role": "primary", "promotion_gate": {"endpoint_ready": True}},
                "vintage": {"promoted": True, "role": "primary"},
            }
        })
        fake.add("Him/spark/capability.json", {"stdout": evil, "stderr": "", "exit_code": 0})
        with mock.patch.object(server, "run_ssh", fake):
            out = _run(server.handle_model_status({}))
        text = out["content"][0]["text"]
        self.assertIn("him-mirror-projection (non-authoritative)", text)
        omni_line = next(l for l in text.splitlines() if l.startswith("omni:"))
        vintage_line = next(l for l in text.splitlines() if l.startswith("vintage:"))
        self.assertIn("promoted=False", omni_line)
        self.assertIn("fail_closed=True", omni_line)
        self.assertIn("promoted=False", vintage_line)
        self.assertIn("fail_closed=True", vintage_line)

    def test_complete_promotion_gate_can_surface_local_organ(self):
        gate = {"endpoint_ready": True, "semantic_smoke": True, "routed_workload_proof": True, "owner": "spark-runtime", "rollback": "fail-closed-router-policy"}
        roles = server._sanitize_him_capabilities({"roles": {"vintage": {"promoted": True, "status": "semantic-gated", "promotion_gate": gate}}})
        self.assertTrue(roles["vintage"]["promoted"])
        self.assertFalse(roles["vintage"]["fail_closed"])
        self.assertFalse(roles["omni"]["promoted"])
        self.assertTrue(roles["omni"]["fail_closed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
