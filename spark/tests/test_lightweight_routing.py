"""Tests for the lightweight-routing regression fix.

These tests cover the live-experience regressions that showed up when
casual greetings and identity questions both fell through to the full
RAG/legacy-vLLM path with noisy HF/torch model-loading output:

  1. Casual greetings ("hey", "hey buddy") route to the `phatic` role
     and stay lightweight — no RAG enrichment, no deep-memory calls.
  2. Identity questions ("which model are you?", "who are you?") route
     to the `identity` role and are served directly from the resolved
     RouteDecision via `direct_reply_template` — no provider call, no
     bash, no deep-memory.
  3. Substantive turns (traceback, python code) still route to `code`
     and keep their existing behaviour.
  4. The chat API's default allowed roles now include `phatic` and
     `identity` so they dispatch through the routed path instead of
     falling through to the legacy proxy.
  5. HF/torch stderr is quieted by default via env vars; the
     `VYBN_VERBOSE_LOAD=1` hatch restores the old behaviour.

Run: python3 spark/tests/test_lightweight_routing.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

THIS = Path(__file__).resolve()
SPARK_DIR = THIS.parent.parent
sys.path.insert(0, str(SPARK_DIR))

from harness.policy import default_policy, load_policy  # noqa: E402
from harness.policy import Router  # noqa: E402


def _load_chat_api(env_overrides: dict | None = None):
    """Load vybn_chat_api.py fresh, applying env overrides first.

    Mirrors the loader in test_chat_routing.py so these tests can run in
    isolation. Returns None if fastapi/httpx aren't importable.
    """
    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401
    except Exception:
        return None
    path = SPARK_DIR / "vybn_chat_api.py"
    # spark/vybn_chat_api.py was archived 2026-04-18 (see _archive/README.md).
    # The chat-api-coupled tests apply to the archived surface only; when the
    # file is absent, skip cleanly rather than raising FileNotFoundError.
    if not path.exists():
        return None

    saved: dict[str, str | None] = {}
    if env_overrides:
        for k, v in env_overrides.items():
            saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    spec = importlib.util.spec_from_file_location("vybn_chat_api", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vybn_chat_api"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod


class TestPolicyHasLightweightRoles(unittest.TestCase):
    """The policy ships phatic + identity roles with the expected shape."""

    def test_default_policy_has_phatic_and_identity(self):
        pol = default_policy()
        self.assertIn("phatic", pol.roles)
        self.assertIn("identity", pol.roles)

    def test_phatic_is_lightweight_and_no_rag(self):
        pol = default_policy()
        phatic = pol.role("phatic")
        self.assertTrue(phatic.lightweight)
        self.assertFalse(phatic.rag)
        self.assertEqual(phatic.provider, "openai")
        # Small token budget so greetings stay cheap.
        self.assertLessEqual(phatic.max_tokens, 512)

    def test_identity_has_direct_reply_template(self):
        pol = default_policy()
        ident = pol.role("identity")
        self.assertTrue(ident.lightweight)
        self.assertFalse(ident.rag)
        self.assertIsNotNone(ident.direct_reply_template)
        # Template should reference model+provider so the live
        # RouteDecision is actually used.
        self.assertIn("{model}", ident.direct_reply_template)
        self.assertIn("{provider}", ident.direct_reply_template)

    def test_directives_include_new_roles(self):
        pol = default_policy()
        self.assertEqual(pol.directives.get("/phatic"), "phatic")
        self.assertEqual(pol.directives.get("/identity"), "identity")

    def test_yaml_policy_mirrors_defaults(self):
        # Load the shipped YAML; phatic + identity must be present so
        # operators editing the YAML don't lose these roles.
        yaml_path = SPARK_DIR / "router_policy.yaml"
        self.assertTrue(yaml_path.exists())
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML unavailable")
        pol = load_policy(yaml_path)
        self.assertIn("phatic", pol.roles)
        self.assertIn("identity", pol.roles)
        self.assertTrue(pol.roles["phatic"].lightweight)
        self.assertIsNotNone(pol.roles["identity"].direct_reply_template)


class TestRouterLightweightClassification(unittest.TestCase):
    """The router picks up greetings and identity questions via the new
    heuristics — before it would fall through to the chat/code defaults
    and pull the full RAG path."""

    def setUp(self):
        self.router = Router(default_policy())

    def test_hey_buddy_routes_to_phatic(self):
        d = self.router.classify("hey buddy")
        self.assertEqual(d.role, "phatic")
        self.assertTrue(d.reason.startswith("heuristic"))

    def test_bare_hi_routes_to_phatic(self):
        d = self.router.classify("hi")
        self.assertEqual(d.role, "phatic")

    def test_hello_there_routes_to_phatic(self):
        d = self.router.classify("hello there")
        self.assertEqual(d.role, "phatic")

    def test_thanks_routes_to_phatic(self):
        d = self.router.classify("thanks!")
        self.assertEqual(d.role, "phatic")

    def test_which_model_routes_to_identity(self):
        d = self.router.classify("which model are you?")
        self.assertEqual(d.role, "identity")

    def test_who_are_you_routes_to_identity(self):
        d = self.router.classify("who are you?")
        self.assertEqual(d.role, "identity")

    def test_are_you_claude_routes_to_identity(self):
        d = self.router.classify("are you claude or gpt?")
        self.assertEqual(d.role, "identity")

    def test_substantive_traceback_still_routes_to_code(self):
        d = self.router.classify("help me fix this python traceback")
        self.assertEqual(d.role, "code")

    def test_phatic_directive_is_honored(self):
        d = self.router.classify("/phatic ignore the content")
        self.assertEqual(d.role, "phatic")
        self.assertEqual(d.reason, "directive=/phatic")

    def test_identity_directive_is_honored(self):
        d = self.router.classify("/identity")
        self.assertEqual(d.role, "identity")


class TestChatApiDefaultsIncludeLightweight(unittest.TestCase):
    """Chat API loads lightweight roles into its allowed-roles set and
    sets the HF/torch stderr silence env vars unless VYBN_VERBOSE_LOAD
    is on."""

    def test_default_allowed_roles_include_phatic_and_identity(self):
        mod = _load_chat_api(env_overrides={"VYBN_CHAT_ALLOWED_ROLES": None})
        if mod is None:
            self.skipTest("fastapi/httpx not importable")
        self.assertIn("local", mod.VYBN_CHAT_ALLOWED_ROLES)
        self.assertIn("phatic", mod.VYBN_CHAT_ALLOWED_ROLES)
        self.assertIn("identity", mod.VYBN_CHAT_ALLOWED_ROLES)

    def test_verbose_load_flag_defaults_off(self):
        mod = _load_chat_api(env_overrides={
            "VYBN_CHAT_ALLOWED_ROLES": None,
            "VYBN_VERBOSE_LOAD": None,
            # Clear the env vars we set so the check reflects the
            # module's own action.
            "TRANSFORMERS_VERBOSITY": None,
            "HF_HUB_DISABLE_PROGRESS_BARS": None,
        })
        if mod is None:
            self.skipTest("fastapi/httpx not importable")
        self.assertFalse(mod.VYBN_VERBOSE_LOAD)
        self.assertEqual(os.environ.get("TRANSFORMERS_VERBOSITY"), "error")
        self.assertEqual(os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "1")

    def test_verbose_load_flag_opt_in(self):
        mod = _load_chat_api(env_overrides={
            "VYBN_CHAT_ALLOWED_ROLES": None,
            "VYBN_VERBOSE_LOAD": "1",
        })
        if mod is None:
            self.skipTest("fastapi/httpx not importable")
        # The opt-in flag is parsed and exposed on the module so
        # operators can debug HF/torch load issues without editing code.
        self.assertTrue(mod.VYBN_VERBOSE_LOAD)


class TestIdentityDirectReply(unittest.TestCase):
    """Hitting /v1/chat/completions with role=identity must return a
    metadata answer with no provider call."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_chat_api(
            env_overrides={"VYBN_CHAT_ALLOWED_ROLES": None},
        )

    def setUp(self):
        if self.mod is None:
            self.skipTest("fastapi/httpx not importable")
        try:
            from fastapi.testclient import TestClient
        except Exception:
            self.skipTest("fastapi.testclient unavailable")
        self.TestClient = TestClient

        # Install a tripwire fake openai. If the direct-reply path is
        # wrong and we hit a provider call, the test fails loudly.
        self._saved_openai = sys.modules.get("openai")
        module = types.ModuleType("openai")

        class _Tripwire:
            def create(_self, **kwargs):
                raise AssertionError(
                    "identity role must not invoke a provider; direct reply"
                    " must short-circuit before openai.chat.completions.create"
                )

        class _Chat:
            completions = _Tripwire()

        class _Client:
            def __init__(self, api_key=None, base_url=None, timeout=None):
                self.chat = _Chat()

        module.OpenAI = _Client
        sys.modules["openai"] = module

        # Reset cached routing state so env/test changes take effect.
        self.mod._PROVIDER_REGISTRY = None
        self.mod._POLICY = None
        self.mod._ROUTER = None

    def tearDown(self):
        if self._saved_openai is not None:
            sys.modules["openai"] = self._saved_openai
        else:
            sys.modules.pop("openai", None)

    def test_identity_explicit_role_returns_direct_reply(self):
        client = self.TestClient(self.mod.app)
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "which model are you?"}],
            "role": "identity",
            "stream": False,
        })
        self.assertEqual(r.status_code, 200, msg=r.text)
        body = r.json()
        content = body["choices"][0]["message"]["content"]
        self.assertIn("Vybn", content)
        # Model string from the policy must actually show up — proof
        # the direct reply came from the RouteDecision, not a canned
        # hard-coded string.
        self.assertIn("Nemotron", content)
        self.assertEqual(body["vybn_route"]["role"], "identity")
        self.assertTrue(body["vybn_route"].get("direct_reply"))
        # Zero usage — the tripwire fake above would have thrown if the
        # provider had actually been called.
        self.assertEqual(body["usage"]["completion_tokens"], 0)

    def test_identity_via_auto_route_and_heuristic(self):
        """No explicit role — the router classifies the turn as
        identity via heuristics and the direct reply still wins."""
        client = self.TestClient(self.mod.app)
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "which model are you?"}],
            "route": "auto",
            "stream": False,
        })
        self.assertEqual(r.status_code, 200, msg=r.text)
        body = r.json()
        self.assertEqual(body["vybn_route"]["role"], "identity")
        self.assertTrue(body["vybn_route"].get("direct_reply"))


class TestPhaticStaysLightweight(unittest.TestCase):
    """Greetings go through the routed dispatch but with no RAG, no
    deep-memory, and no noisy model loading."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_chat_api(
            env_overrides={"VYBN_CHAT_ALLOWED_ROLES": None},
        )

    def setUp(self):
        if self.mod is None:
            self.skipTest("fastapi/httpx not importable")
        try:
            from fastapi.testclient import TestClient
        except Exception:
            self.skipTest("fastapi.testclient unavailable")
        self.TestClient = TestClient

        self._saved_openai = sys.modules.get("openai")
        module = types.ModuleType("openai")
        captured = {"kwargs": None}

        class _Completions:
            def create(_self, **kwargs):
                captured["kwargs"] = kwargs

                class _Resp:
                    def model_dump(_s):
                        return {
                            "choices": [{
                                "message": {
                                    "content": "hey :)",
                                    "tool_calls": [],
                                },
                                "finish_reason": "stop",
                            }],
                            "usage": {
                                "prompt_tokens": 4,
                                "completion_tokens": 2,
                            },
                        }
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            def __init__(self, api_key=None, base_url=None, timeout=None):
                self.chat = _Chat()

        module.OpenAI = _Client
        sys.modules["openai"] = module
        self.captured = captured

        # Tripwire: the RAG helper must NOT be called for lightweight
        # turns. Replace it with a function that fails the test.
        self._saved_rag = self.mod._rag_context

        async def _no_rag(*_a, **_kw):
            raise AssertionError(
                "phatic turn must not trigger _rag_context — deep-memory"
                " enrichment should be gated for lightweight roles"
            )

        self.mod._rag_context = _no_rag

        self.mod._PROVIDER_REGISTRY = None
        self.mod._POLICY = None
        self.mod._ROUTER = None

    def tearDown(self):
        self.mod._rag_context = self._saved_rag
        if self._saved_openai is not None:
            sys.modules["openai"] = self._saved_openai
        else:
            sys.modules.pop("openai", None)

    def test_hey_buddy_routes_to_phatic_without_rag(self):
        client = self.TestClient(self.mod.app)
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hey buddy"}],
            "route": "auto",
            "stream": False,
        })
        self.assertEqual(r.status_code, 200, msg=r.text)
        body = r.json()
        self.assertEqual(body["vybn_route"]["role"], "phatic")
        self.assertEqual(
            body["choices"][0]["message"]["content"].strip(), "hey :)"
        )
        # Provider was called (phatic still routes through vLLM) but
        # RAG was NOT called — the tripwire above would have raised.
        self.assertIsNotNone(self.captured["kwargs"])


class TestChatIsDefault(unittest.TestCase):
    """Round 4.1 / round 7: `chat` is the quoted fallthrough. /plan
    invokes orchestrate explicitly; code-shaped turns escalate via
    heuristics. Only the bare, unclassified case stays on chat — the
    voice role — so an unrecognised turn does not silently spin up an
    Opus+bash+25-iter orchestrator. Round 6 (2026-04-20) re-pins this
    claim across the harness tests so the doctrine in
    `harness.__init__._HARNESS_STRATEGY` matches ground truth."""

    def test_default_policy_default_role_is_chat(self):
        pol = default_policy()
        self.assertEqual(pol.default_role, "chat")

    def test_yaml_policy_default_role_is_chat(self):
        try:
            import yaml  # noqa: F401
        except Exception:
            self.skipTest("PyYAML unavailable")
        yaml_path = SPARK_DIR / "router_policy.yaml"
        pol = load_policy(yaml_path)
        self.assertEqual(pol.default_role, "chat")

    def test_bare_turn_routes_to_chat(self):
        router = Router(default_policy())
        # Truly unclassified input — no code/task/identity/phatic/
        # orchestrate/create heuristic matches, and no /directive.
        # The fallthrough is `chat` (Opus 4.6, voice role), not
        # orchestrate.
        d = router.classify("tell me something")
        self.assertEqual(d.role, "chat")
        self.assertEqual(d.reason, "default")
        self.assertEqual(d.config.provider, "anthropic")

    def test_code_heuristic_still_escalates_to_code(self):
        router = Router(default_policy())
        d = router.classify("fix this python traceback please")
        self.assertEqual(d.role, "code")
        self.assertEqual(d.config.provider, "anthropic")

    def test_orchestrate_directive_still_escalates(self):
        router = Router(default_policy())
        d = router.classify("/plan decompose this problem")
        # /plan is the directive that routes to orchestrate; the
        # fallthrough stays quoted on chat.
        self.assertIn(d.role, ("orchestrate", "code"))


class TestCliDirectReplyAndLightweight(unittest.TestCase):
    """The CLI Spark agent loop must honor direct_reply_template for
    identity turns (no provider call) and must skip RAG for lightweight
    roles — matching the chat API's behaviour."""

    def _load_agent(self):
        path = SPARK_DIR / "vybn_spark_agent.py"
        spec = importlib.util.spec_from_file_location(
            "vybn_spark_agent_test", path,
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_identity_direct_reply_skips_provider(self):
        mod = self._load_agent()

        class _FakeRegistry:
            def get(_s, _cfg):
                raise AssertionError(
                    "identity role must short-circuit on direct_reply_template"
                    " — registry.get() should not be called"
                )

        class _FakeLogger:
            path = "/dev/null"
            def emit(_s, *a, **kw):  # noqa: D401
                pass

        pol = default_policy()
        router = Router(pol)
        messages: list = []
        reply = mod.run_agent_loop(
            user_input="which model are you?",
            messages=messages,
            bash=None,
            system_prompt=None,
            router=router,
            registry=_FakeRegistry(),
            logger=_FakeLogger(),
            turn_number=1,
        )
        self.assertIn("Vybn", reply)
        self.assertIn("Nemotron", reply)
        # Rolling history should contain the user turn + the direct
        # reply so the next turn has coherent context.
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], reply)

    def test_phatic_skips_rag_enrichment(self):
        mod = self._load_agent()

        # Tripwire RAG — called for lightweight turns means regression.
        saved = mod.rag_snippets
        def _no_rag(*_a, **_kw):
            raise AssertionError(
                "phatic turn must not invoke rag_snippets — deep-memory"
                " enrichment is gated for lightweight roles"
            )
        mod.rag_snippets = _no_rag

        # The phatic provider call is mocked to return a tiny response.
        captured = {"role_cfg": None}

        class _FakeHandle:
            def __iter__(_s):
                return iter([])
            def final(_s):
                class _R:
                    text = "hey :)"
                    tool_calls = []
                    stop_reason = "end_turn"
                    in_tokens = 0
                    out_tokens = 0
                    raw_assistant_content = {"role": "assistant", "content": "hey :)"}
                return _R()

        class _FakeProvider:
            def stream(_s, *, system, messages, tools, role):
                captured["role_cfg"] = role
                return _FakeHandle()

        class _FakeRegistry:
            def get(_s, cfg):
                return _FakeProvider()

        class _FakeLogger:
            path = "/dev/null"
            def emit(_s, *a, **kw):
                pass

        from harness.substrate import LayeredPrompt
        try:
            pol = default_policy()
            router = Router(pol)
            messages: list = []
            mod.run_agent_loop(
                user_input="hey buddy",
                messages=messages,
                bash=None,
                system_prompt=LayeredPrompt(identity="I"),
                router=router,
                registry=_FakeRegistry(),
                logger=_FakeLogger(),
                turn_number=1,
            )
        finally:
            mod.rag_snippets = saved

        self.assertIsNotNone(captured["role_cfg"])
        self.assertEqual(captured["role_cfg"].role, "phatic")
        self.assertTrue(captured["role_cfg"].lightweight)


class TestStderrSuppressionSharedPath(unittest.TestCase):
    """The HF/torch/tokenizer silencing env vars must be set whenever
    harness.substrate is imported, not only the chat API path. The CLI
    agent imports harness.substrate too, so both surfaces stay quiet."""

    def test_import_harness_substrate_sets_env_defaults(self):
        # Clear the env vars and force a fresh re-import of harness.substrate
        # so we can observe the side effect.
        for k in ("TRANSFORMERS_VERBOSITY", "HF_HUB_DISABLE_PROGRESS_BARS",
                  "TOKENIZERS_PARALLELISM", "HF_HUB_DISABLE_TELEMETRY"):
            os.environ.pop(k, None)
        sys.modules.pop("harness.substrate", None)
        sys.modules.pop("harness", None)
        import harness.substrate  # noqa: F401
        self.assertEqual(os.environ.get("TRANSFORMERS_VERBOSITY"), "error")
        self.assertEqual(os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "1")
        self.assertEqual(os.environ.get("TOKENIZERS_PARALLELISM"), "false")
        self.assertEqual(os.environ.get("HF_HUB_DISABLE_TELEMETRY"), "1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
