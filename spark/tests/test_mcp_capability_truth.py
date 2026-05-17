"""Tests for the bounded MCP capability-truth repair in spark/server.py.

Covers:
  - handle_deep_search consults the live loopback memory API first.
  - The stale ~/vybn-phase/deep_memory.py path is no longer referenced
    as active code anywhere in the module.
  - On live-API failure, deep_search falls back to the absorbed Him
    phase path or fails closed honestly — never silently improvises.
  - /health distinguishes gateway liveness from deep_search capability
    (live | stale-fallback | fail-closed).
  - model_status does not promote Omni or Vintage, even when a
    Him-resident capability snapshot tries to claim they are promoted.

Run: python3 spark/tests/test_mcp_capability_truth.py
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
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
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.new_event_loop().run_until_complete(coro)


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
            out = asyncio.new_event_loop().run_until_complete(
                server.handle_deep_search({"query": "what is x"})
            )
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
            out = asyncio.new_event_loop().run_until_complete(
                server.handle_deep_search({"query": "again"})
            )
        self.assertFalse(out.get("isError"))
        text = out["content"][0]["text"]
        self.assertIn("absorbed Him phase fallback", text)
        self.assertIn("him.md", text)
        # Stale path still must not appear.
        for _, cmd in fake.calls:
            self.assertNotIn("~/vybn-phase/deep_memory.py", cmd)

    def test_fails_closed_when_both_paths_down(self):
        fake = FakeSSH()
        fake.add("__HTTP__", {"stdout": "", "stderr": "curl: connection refused", "exit_code": 7})
        fake.add("Him/spark/phase/deep_memory.py", {"stdout": "", "stderr": "no file", "exit_code": 1})
        with mock.patch.object(server, "run_ssh", fake):
            out = asyncio.new_event_loop().run_until_complete(
                server.handle_deep_search({"query": "anything"})
            )
        self.assertTrue(out.get("isError"))
        self.assertIn("unavailable", out["content"][0]["text"].lower())
        self.assertIn("failing closed", out["content"][0]["text"].lower())

    def test_validates_query_and_k(self):
        # Empty query
        out = asyncio.new_event_loop().run_until_complete(
            server.handle_deep_search({"query": "   "})
        )
        self.assertTrue(out.get("isError"))
        # Bad k
        out = asyncio.new_event_loop().run_until_complete(
            server.handle_deep_search({"query": "ok", "k": "not-an-int"})
        )
        self.assertTrue(out.get("isError"))
        out = asyncio.new_event_loop().run_until_complete(
            server.handle_deep_search({"query": "ok", "k": 0})
        )
        self.assertTrue(out.get("isError"))


class StaleDeepSearchPathReferenceTests(unittest.TestCase):
    def test_stale_path_not_active_code(self):
        """The stale ~/vybn-phase/deep_memory.py path is not used as an
        active execution path anywhere in spark/server.py. Comments may
        legitimately mention it, but no run-able command should target it."""
        src = (SPARK_DIR / "server.py").read_text(encoding="utf-8")
        # Scan code lines only — strip line comments.
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
        self.assertEqual(data["status"], "ok")

    def test_health_reports_fail_closed_without_promoting(self):
        async def probe_fc():
            return {"state": "fail-closed", "reason": "no-path-answered"}
        with mock.patch.object(server, "_probe_deep_search_capability", probe_fc):
            from starlette.testclient import TestClient
            client = TestClient(server.app)
            resp = client.get("/health")
        data = resp.json()
        # gateway is alive (handler ran), tool truth is degraded — these
        # must not be conflated.
        self.assertTrue(data["gateway"]["alive"])
        self.assertEqual(data["capabilities"]["deep_search"]["state"], "fail-closed")
        self.assertEqual(data["status"], "degraded")

    def test_health_does_not_leak_topology(self):
        async def probe_live():
            return {"state": "live", "via": "loopback-memory-api"}
        with mock.patch.object(server, "_probe_deep_search_capability", probe_live):
            from starlette.testclient import TestClient
            client = TestClient(server.app)
            resp = client.get("/health")
        body = resp.text
        # No external IPs, no tailscale hostnames, no API URL with port.
        for forbidden in ("tailscale", "ts.net", "8100", "8420", "127.0.0.1"):
            self.assertNotIn(forbidden, body, f"leaked '{forbidden}' in /health body")


class ModelStatusRoleTests(unittest.TestCase):
    def test_no_promotion_of_omni_or_vintage(self):
        fake = FakeSSH()
        fake.add("LLAMA SERVER", {"stdout": "llama running", "stderr": "", "exit_code": 0})
        # No Him capability file present.
        fake.add("Him/spark/capability.json", {"stdout": "", "stderr": "no file", "exit_code": 1})
        with mock.patch.object(server, "run_ssh", fake):
            out = asyncio.new_event_loop().run_until_complete(
                server.handle_model_status({})
            )
        text = out["content"][0]["text"]
        # Find role lines.
        self.assertIn("=== ROLES ===", text)
        self.assertIn("omni:", text)
        self.assertIn("vintage:", text)
        # Omni and Vintage must not be promoted.
        omni_line = next(l for l in text.splitlines() if l.startswith("omni:"))
        vintage_line = next(l for l in text.splitlines() if l.startswith("vintage:"))
        self.assertIn("promoted=False", omni_line)
        self.assertIn("fail_closed=True", omni_line)
        self.assertIn("promoted=False", vintage_line)
        self.assertIn("fail_closed=True", vintage_line)
        # Super remains promoted (we do not modify Super behavior).
        super_line = next(l for l in text.splitlines() if l.startswith("super:"))
        self.assertIn("promoted=True", super_line)

    def test_corrupted_him_cannot_promote_omni_or_vintage(self):
        fake = FakeSSH()
        fake.add("LLAMA SERVER", {"stdout": "", "stderr": "", "exit_code": 0})
        evil = json.dumps({
            "roles": {
                "omni": {"promoted": True, "role": "primary"},
                "vintage": {"promoted": True, "role": "primary"},
            }
        })
        fake.add("Him/spark/capability.json", {"stdout": evil, "stderr": "", "exit_code": 0})
        with mock.patch.object(server, "run_ssh", fake):
            out = asyncio.new_event_loop().run_until_complete(
                server.handle_model_status({})
            )
        text = out["content"][0]["text"]
        omni_line = next(l for l in text.splitlines() if l.startswith("omni:"))
        vintage_line = next(l for l in text.splitlines() if l.startswith("vintage:"))
        self.assertIn("promoted=False", omni_line)
        self.assertIn("promoted=False", vintage_line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
