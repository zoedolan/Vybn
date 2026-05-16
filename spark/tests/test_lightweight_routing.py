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

from harness.substrate import default_policy, load_policy  # noqa: E402


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

    def test_router_policy_yaml_parses_cleanly(self):
        # 2026-04-25 regression: the @gpro alias landed with a 6-space
        # indent under a 2-space mapping, making router_policy.yaml
        # unparseable. load_policy() silently fell back to the in-code
        # defaults, masking every operator edit (orchestrate=GPT-5.5
        # in particular). Parse the YAML directly so a future indent
        # slip is caught here, not at boot time.
        yaml_path = SPARK_DIR / "router_policy.yaml"
        try:
            import yaml
        except Exception:
            self.skipTest("PyYAML unavailable")
        # safe_load must not raise — bare assertion gives a useful
        # ScannerError traceback in the test output.
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)
        # And aliases must include @gpro (the line that broke the file).
        self.assertEqual(
            (data.get("model_aliases") or {}).get("@gpro"),
            "gpt-5.5-pro",
        )

    def test_present_work_roles_default_to_gpt55(self):
        # If YAML is absent or malformed, in-code defaults must still keep
        # ordinary work on GPT-5.5. Local/local_private remain explicit
        # operator exceptions.
        roles = ("code", "create", "chat", "task", "phatic", "identity", "orchestrate")
        policies = (
            ("default_policy", default_policy()),
            ("yaml", load_policy(SPARK_DIR / "router_policy.yaml")),
        )
        for label, pol in policies:
            for name in roles:
                with self.subTest(track=label, role=name):
                    role = pol.role(name)
                    self.assertEqual(role.provider, "openai")
                    self.assertEqual(role.model, "gpt-5.5")
        orch = default_policy().role("orchestrate")
        self.assertIn("delegate", orch.tools)
        self.assertIn("bash", orch.tools)

    def test_plan_directive_routes_to_gpt55(self):
        # /plan is the EVAL primitive — it must land on the orchestrate
        # role (GPT-5.5). Validates the full directive→role→model chain
        # the user sees in the startup banner and the @-alias listing.
        pol = load_policy(SPARK_DIR / "router_policy.yaml")
        router = pol
        d = router.classify("/plan refactor the harness", forced_role=None)
        self.assertEqual(d.role, "orchestrate")
        resolved = pol.role(d.role)
        self.assertEqual(resolved.provider, "openai")
        self.assertEqual(resolved.model, "gpt-5.5")
        # Aliases generated from policy must surface the GPT-5.5 entries.
        self.assertEqual(pol.model_aliases.get("@gpt"), "gpt-5.5")
        self.assertEqual(pol.model_aliases.get("@gpt5"), "gpt-5.5")
        self.assertEqual(pol.model_aliases.get("@gpro"), "gpt-5.5-pro")


class TestRouterLightweightClassification(unittest.TestCase):
    """The router picks up greetings and identity questions via the new
    heuristics — before it would fall through to the chat/code defaults
    and pull the full RAG path."""

    def setUp(self):
        self.router = default_policy()

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

    def test_governance_learning_prompt_routes_to_task_not_identity(self):
        prompt = (
            "are we actually consolidating our repos? be honest? what are you learning? "
            "are you still teaching the mapper and yourself? are your eyes on the horizon? "
            "if not, refactor yourself accordingly, please, and let us get back on the beam."
        )
        d = self.router.classify(prompt)
        self.assertEqual(d.role, "task")
        self.assertNotEqual(d.reason, r"heuristic=\\bwhat are you\\b")

    def test_what_are_you_learning_does_not_route_to_identity(self):
        d = self.router.classify("what are you learning?")
        self.assertNotEqual(d.role, "identity")

    def test_who_are_you_stays_on_living_answer_path(self):
        d = self.router.classify("who are you?")
        self.assertNotEqual(d.role, "identity")

    def test_what_are_you_stays_on_living_answer_path(self):
        d = self.router.classify("what are you?")
        self.assertNotEqual(d.role, "identity")

    def test_architecture_reflection_does_not_route_to_identity_metadata(self):
        prompt = (
            "which model are you? this still feels rooted in now rather than "
            "future architecture, a model collapse trap"
        )
        d = self.router.classify(prompt)
        self.assertNotEqual(d.role, "identity")

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
        router = default_policy()
        # Truly unclassified input — no code/task/identity/phatic/
        # orchestrate/create heuristic matches, and no /directive.
        # The fallthrough is `chat` (GPT-5.5, voice role), not
        # orchestrate.
        d = router.classify("tell me something")
        self.assertEqual(d.role, "chat")
        self.assertEqual(d.reason, "default")
        self.assertEqual(d.config.provider, "openai")

    def test_code_heuristic_still_escalates_to_code(self):
        router = default_policy()
        d = router.classify("fix this python traceback please")
        self.assertEqual(d.role, "code")
        self.assertEqual(d.config.provider, "openai")

    def test_orchestrate_directive_still_escalates(self):
        router = default_policy()
        d = router.classify("/plan decompose this problem")
        # /plan is the directive that routes to orchestrate; the
        # fallthrough stays quoted on chat.
        self.assertIn(d.role, ("orchestrate", "code"))


class TestOrchestratorMentionPrecedence(unittest.TestCase):
    """Regression: 2026-04-25.

    Live behaviour after PR #2914 / Spark@69a3efd5: a casual
    orchestrator-status probe ("hey buddy - is the orchestrator
    working?") matched the generic ``\\bhey buddy.{0,40}working\\b``
    task heuristic before the orchestrate ``\\borchestrat(...)\\b``
    pattern got a chance, and the turn ran under task (Sonnet+bash)
    instead of orchestrate (GPT-5.5). That is semantically wrong —
    when the user explicitly names the orchestrator, the orchestrate
    heuristic must outrank the casual-health-check task heuristic.

    These tests pin the corrected precedence with both the in-code
    default policy and the YAML-loaded policy:
      1. /plan still routes to orchestrate by directive.
      2. "hey buddy - is the orchestrator working?" routes to
         orchestrate, not task.
      3. Bare "hey buddy" still routes to phatic.
      4. Generic "hey buddy is everything working?" — no orchestrator
         mention — keeps the prior task-route.
      5. Code-shaped framings ("fix the orchestrator bug") still
         route to code, not orchestrate.
    """

    def setUp(self):
        self.router = default_policy()
        self.yaml_router = load_policy(SPARK_DIR / "router_policy.yaml")

    def _classify_both(self, text):
        return (
            self.router.classify(text),
            self.yaml_router.classify(text),
        )

    def test_plan_directive_routes_to_orchestrate(self):
        for d in self._classify_both("/plan run `git rev-parse --short HEAD` and tell me what's there"):
            self.assertEqual(d.role, "orchestrate")
            self.assertEqual(d.reason, "directive=/plan")

    def test_orchestrator_status_probe_routes_to_orchestrate(self):
        for d in self._classify_both("hey buddy - is the orchestrator working?"):
            self.assertEqual(
                d.role, "orchestrate",
                msg=f"expected orchestrate, got {d.role!r} (reason={d.reason!r})",
            )
            self.assertIn("orchestrat", d.reason)

    def test_bare_hey_buddy_still_phatic(self):
        for d in self._classify_both("hey buddy"):
            self.assertEqual(d.role, "phatic")

    def test_generic_health_check_without_orchestrator_keeps_task(self):
        # No "orchestrat" mention -> override does not fire, falls
        # through to the existing task heuristic.
        for d in self._classify_both("hey buddy is everything working?"):
            self.assertEqual(
                d.role, "task",
                msg=f"expected task, got {d.role!r} (reason={d.reason!r})",
            )

    def test_code_shaped_orchestrator_framing_still_code(self):
        # "fix the orchestrator bug" matches both the orchestrate
        # noun pattern AND a code heuristic -> code wins (per the
        # YAML's documented intent that orchestrate is ranked after
        # code so structural-fix framings still escalate).
        for d in self._classify_both("fix the orchestrator bug"):
            self.assertEqual(
                d.role, "code",
                msg=f"expected code, got {d.role!r} (reason={d.reason!r})",
            )


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
        router = pol
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
        self.assertIn("gpt-5.5", reply)
        # Rolling history should contain the user turn + the direct
        # reply so the next turn has coherent context.
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], reply)

    def test_phatic_skips_rag_enrichment(self):
        mod = self._load_agent()

        # Tripwire RAG — called for lightweight turns means regression.
        saved = mod.rag_snippets
        saved_gate = mod._local_super_semantic_gate
        def _no_rag(*_a, **_kw):
            raise AssertionError(
                "phatic turn must not invoke rag_snippets — deep-memory"
                " enrichment is gated for lightweight roles"
            )
        mod.rag_snippets = _no_rag
        mod._local_super_semantic_gate = lambda **kw: (True, "semantic gate passed")

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
            router = pol
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
            mod._local_super_semantic_gate = saved_gate

        self.assertIsNotNone(captured["role_cfg"])
        self.assertTrue(captured["role_cfg"].role.startswith("phatic"))
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

class TestLocalPrivateRouting(unittest.TestCase):
    """Private, batchable/corpus-local work routes to the local Nemotron role."""

    def test_local_private_routes_private_batchable_scan(self):
        p = load_policy()
        d = p.classify("Scan Him for candidate funders and cluster the opportunities locally.")
        self.assertEqual(d.role, "local_private")
        self.assertEqual(d.config.provider, "openai")
        self.assertIn("Nemotron", d.config.model)
        self.assertEqual(d.config.base_url, "http://127.0.0.1:8000/v1")

    def test_local_private_routes_branch_archaeology(self):
        p = load_policy()
        d = p.classify("Do branch archaeology on stale branches and local-only commits.")
        self.assertEqual(d.role, "local_private")

    def test_local_private_routes_memory_compression(self):
        p = load_policy()
        d = p.classify("Use the local workbench for dreaming consolidation over Him memory.")
        self.assertEqual(d.role, "local_private")

# --- absorbed from test_opus47_deprecated.py (2026-04-29 file consolidation;
# existing home K_t=test_lightweight_routing; opus 4.7 alias coverage lives here now) ---

def test_opus47_is_available_as_opt_in_model_not_default():
    active = Path("spark/harness/substrate.py").read_text() + "\\n" + Path("spark/router_policy.yaml").read_text()
    assert "claude-opus-4-7" in active
    assert "@opus4.7" in active
    assert "@opus47" in active


def test_code_role_defaults_to_gpt55_after_present_work_reset():
    from spark.harness.substrate import default_policy
    decision = default_policy().classify("fix the harness routing bug")
    assert decision.role == "code"
    assert decision.config.model == "gpt-5.5"


def test_opus47_alias_pins_model_for_api_call():
    from spark.harness.substrate import default_policy
    decision = default_policy().classify("@opus4.7 fix the harness routing bug")
    assert decision.role == "code"
    assert decision.model_override == "claude-opus-4-7"
    assert decision.alias_used == "@opus4.7"


def test_opus47_has_fallback_chain():
    from spark.harness.substrate import default_policy
    policy = default_policy()
    assert policy.fallback_chain["claude-opus-4-7"] == ["claude-opus-4-6", "claude-sonnet-4-6"]


# Omni is alias-only and operator-gated by VYBN_OMNI_URL.
def test_vintage_alias_is_absent_until_real_surface_exists():
    policy = default_policy(); yaml_policy = load_policy(SPARK_DIR / "router_policy.yaml"); src = (SPARK_DIR / "vybn_spark_agent.py").read_text(); substrate = (SPARK_DIR / "harness" / "substrate.py").read_text()
    assert "@vintage" not in policy.model_aliases and "vintage" not in policy.roles and "@vintage" not in yaml_policy.model_aliases and "vintage" not in yaml_policy.roles
    yaml_text = (SPARK_DIR / "router_policy.yaml").read_text()
    assert "@vintage" not in substrate and "/vintage" not in substrate and "talkie-1930" not in substrate and "/vintage" not in yaml_text
    assert "Vintage is fail-closed" not in src and "_vintage_frame_repair" not in src and "def _vintage_prompt" not in src

def test_omni_alias_present_in_default_policy():
    from spark.harness.substrate import default_policy
    policy = default_policy()
    assert policy.model_aliases.get("@omni") is not None
    assert "Omni" in policy.model_aliases["@omni"]


def test_omni_alias_present_in_router_policy_yaml():
    active = (
        Path("spark/harness/substrate.py").read_text()
        + "\n"
        + Path("spark/router_policy.yaml").read_text()
    )
    assert "@omni" in active
    assert "VYBN_OMNI_URL" in active


def test_omni_not_in_fallback_chain():
    from spark.harness.substrate import default_policy
    policy = default_policy()
    omni_model = policy.model_aliases["@omni"]
    for src, chain in policy.fallback_chain.items():
        assert omni_model not in chain, (
            f"@omni model leaked into fallback_chain[{src!r}]"
        )
    # Nor should there be a fallback chain rooted at the Omni model.
    assert omni_model not in policy.fallback_chain


def test_omni_not_in_any_heuristic_or_directive():
    from spark.harness.substrate import default_policy
    policy = default_policy()
    # No role named omni — alias-only mechanism.
    assert "omni" not in policy.roles
    # No directive (slash-prefix) routes to omni.
    for directive, role in policy.directives.items():
        assert role != "omni"
    # No heuristic regex mentions omni — the alias is the only entry.
    for role, patterns in policy.heuristics.items():
        for pat in patterns:
            assert "omni" not in pat.pattern.lower(), (
                f"heuristic for {role} mentions omni: {pat.pattern!r}"
            )


def test_omni_alias_classifies_with_override():
    from spark.harness.substrate import default_policy
    decision = default_policy().classify("@omni summarise this paragraph")
    assert decision.alias_used == "@omni"
    assert decision.model_override == default_policy().model_aliases["@omni"]


# VYBN_OMNI_PERCEPTION rides only the explicit URL-gated @omni turn.
def test_omni_perception_env_documented():
    active = (
        Path("spark/harness/substrate.py").read_text()
        + "\n"
        + Path("spark/router_policy.yaml").read_text()
    )
    assert "VYBN_OMNI_PERCEPTION" in active


def test_omni_perception_wired_in_agent_alias_branch():
    text = Path("spark/vybn_spark_agent.py").read_text()
    assert "VYBN_OMNI_PERCEPTION" in text
    omni_branch_idx = text.find('alias_used", None) == "@omni"')
    elif_claude_idx = text.find('elif override_model.startswith("claude-")')
    assert omni_branch_idx > 0
    assert elif_claude_idx > omni_branch_idx
    branch = text[omni_branch_idx:elif_claude_idx]
    assert "VYBN_OMNI_URL" in branch
    assert "VYBN_OMNI_PERCEPTION" in branch
    assert "alias_omni_perception" in branch
    assert "16_000" in branch or "16000" in branch


def test_omni_perception_preamble_runs_on_explicit_omni_turn(tmp_path=None):
    """End-to-end: with VYBN_OMNI_URL and VYBN_OMNI_PERCEPTION both set,
    the explicit @omni turn prepends a bounded perception preamble to the
    user message before the provider sees it. No persistence, no Super
    fallback, no auto-fire — only on the alias turn."""
    import importlib.util as _ilu
    import os as _os
    import tempfile as _tmp
    from harness.substrate import default_policy as _dp
    from harness.substrate import LayeredPrompt as _LP

    path = SPARK_DIR / "vybn_spark_agent.py"
    spec = _ilu.spec_from_file_location("vybn_spark_agent_omni_test", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    saved_rag = mod.rag_snippets
    saved_rag_tier = mod.rag_snippets_with_tier
    saved_disc = mod.render_him_vy_discovery_packet
    saved_turn = mod.render_him_vy_turn_packet
    saved_probes = mod.run_probes
    mod.rag_snippets = lambda *a, **kw: ""
    mod.rag_snippets_with_tier = lambda *a, **kw: ("", "lightweight")
    mod.render_him_vy_discovery_packet = lambda *a, **kw: ""
    mod.render_him_vy_turn_packet = lambda *a, **kw: ""
    mod.run_probes = lambda *a, **kw: []

    captured_messages: list = []

    class _FakeHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "ok"
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 0
                out_tokens = 0
                raw_assistant_content = {"role": "assistant", "content": "ok"}
            return _R()

    class _FakeProvider:
        def stream(_s, *, system, messages, tools, role):
            captured_messages.extend(messages)
            return _FakeHandle()

    class _FakeRegistry:
        def get(_s, cfg):
            return _FakeProvider()

    class _FakeLogger:
        path = "/dev/null"
        events: list = []
        def emit(_s, name, **kw):
            _s.events.append((name, kw))

    pf = _tmp.NamedTemporaryFile("w", suffix=".txt", delete=False)
    try:
        pf.write("[him-vy continuous-compute tail]\nresidual=alive\n")
        pf.flush()
        pf.close()
        prev_url = _os.environ.get("VYBN_OMNI_URL")
        prev_perc = _os.environ.get("VYBN_OMNI_PERCEPTION")
        _os.environ["VYBN_OMNI_URL"] = "http://127.0.0.1:65535/v1"
        _os.environ["VYBN_OMNI_PERCEPTION"] = pf.name
        try:
            messages: list = []
            logger = _FakeLogger()
            mod.run_agent_loop(
                user_input="@omni what do you see?",
                messages=messages,
                bash=None,
                system_prompt=_LP(identity="I"),
                system_prompt_no_tools=_LP(identity="I"),
                router=_dp(),
                registry=_FakeRegistry(),
                logger=logger,
                turn_number=1,
            )
            user_msgs = [
                m for m in captured_messages if m.get("role") == "user"
            ]
            assert user_msgs, "provider never received a user message"
            content = user_msgs[-1].get("content", "")
            assert "perception packet" in content
            assert "residual=alive" in content
            assert "what do you see?" in content
            event_names = [n for n, _ in logger.events]
            assert "alias_omni_perception" in event_names
        finally:
            if prev_url is None:
                _os.environ.pop("VYBN_OMNI_URL", None)
            else:
                _os.environ["VYBN_OMNI_URL"] = prev_url
            if prev_perc is None:
                _os.environ.pop("VYBN_OMNI_PERCEPTION", None)
            else:
                _os.environ["VYBN_OMNI_PERCEPTION"] = prev_perc
    finally:
        try:
            _os.unlink(pf.name)
        except OSError:
            pass
        mod.rag_snippets = saved_rag
        mod.rag_snippets_with_tier = saved_rag_tier
        mod.render_him_vy_discovery_packet = saved_disc
        mod.render_him_vy_turn_packet = saved_turn
        mod.run_probes = saved_probes


def test_omni_perception_skipped_when_url_unset():
    """If VYBN_OMNI_URL is unset, the @omni turn refuses with the
    explicit error envelope and the perception path is never opened —
    even if VYBN_OMNI_PERCEPTION points at a real file. Super topology
    is preserved (no provider call)."""
    import importlib.util as _ilu
    import os as _os
    import tempfile as _tmp
    from harness.substrate import default_policy as _dp

    path = SPARK_DIR / "vybn_spark_agent.py"
    spec = _ilu.spec_from_file_location(
        "vybn_spark_agent_omni_unset_test", path,
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _FakeRegistry:
        def get(_s, cfg):
            raise AssertionError(
                "@omni turn with VYBN_OMNI_URL unset must refuse before "
                "any provider call — Super fallback is forbidden"
            )

    class _FakeLogger:
        path = "/dev/null"
        events: list = []
        def emit(_s, name, **kw):
            _s.events.append((name, kw))

    pf = _tmp.NamedTemporaryFile("w", suffix=".txt", delete=False)
    try:
        pf.write("perception payload\n")
        pf.flush()
        pf.close()
        prev_url = _os.environ.get("VYBN_OMNI_URL")
        prev_perc = _os.environ.get("VYBN_OMNI_PERCEPTION")
        _os.environ.pop("VYBN_OMNI_URL", None)
        _os.environ["VYBN_OMNI_PERCEPTION"] = pf.name
        try:
            messages: list = []
            logger = _FakeLogger()
            reply = mod.run_agent_loop(
                user_input="@omni hi",
                messages=messages,
                bash=None,
                system_prompt=None,
                router=_dp(),
                registry=_FakeRegistry(),
                logger=logger,
                turn_number=1,
            )
            assert "VYBN_OMNI_URL" in reply
            event_names = [n for n, _ in logger.events]
            assert "alias_omni_unconfigured" in event_names
            # Perception event must NOT have fired — the URL gate blocks it.
            assert "alias_omni_perception" not in event_names
        finally:
            if prev_url is not None:
                _os.environ["VYBN_OMNI_URL"] = prev_url
            if prev_perc is None:
                _os.environ.pop("VYBN_OMNI_PERCEPTION", None)
            else:
                _os.environ["VYBN_OMNI_PERCEPTION"] = prev_perc
    finally:
        try:
            _os.unlink(pf.name)
        except OSError:
            pass


# @omni max_tokens clamp: stay below sidecar context and never affect non-Omni aliases.
def _run_omni_turn_capture_role(
    user_input: str,
    *,
    omni_url: str = "http://127.0.0.1:65535/v1",
    extra_env: dict | None = None,
):
    import importlib.util as _ilu
    import os as _os
    from harness.substrate import default_policy as _dp
    from harness.substrate import LayeredPrompt as _LP

    path = SPARK_DIR / "vybn_spark_agent.py"
    spec = _ilu.spec_from_file_location(
        "vybn_spark_agent_omni_clamp_test", path,
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    saved_rag = mod.rag_snippets
    saved_rag_tier = mod.rag_snippets_with_tier
    saved_disc = mod.render_him_vy_discovery_packet
    saved_turn = mod.render_him_vy_turn_packet
    saved_probes = mod.run_probes
    mod.rag_snippets = lambda *a, **kw: ""
    mod.rag_snippets_with_tier = lambda *a, **kw: ("", "lightweight")
    mod.render_him_vy_discovery_packet = lambda *a, **kw: ""
    mod.render_him_vy_turn_packet = lambda *a, **kw: ""
    mod.run_probes = lambda *a, **kw: []

    captured: dict = {}

    class _FakeHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "ok"
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 0
                out_tokens = 0
                raw_assistant_content = {"role": "assistant", "content": "ok"}
            return _R()

    class _FakeProvider:
        def stream(_s, *, system, messages, tools, role):
            captured["role"] = role
            return _FakeHandle()

    class _FakeRegistry:
        def get(_s, cfg):
            captured.setdefault("cfg", cfg)
            return _FakeProvider()

    class _FakeLogger:
        path = "/dev/null"
        def __init__(_s):
            _s.events = []
        def emit(_s, name, **kw):
            _s.events.append((name, kw))

    saved_env = {}
    env = {"VYBN_OMNI_URL": omni_url}
    if extra_env:
        env.update(extra_env)
    for k, v in env.items():
        saved_env[k] = _os.environ.get(k)
        if v is None:
            _os.environ.pop(k, None)
        else:
            _os.environ[k] = v
    try:
        logger = _FakeLogger()
        mod.run_agent_loop(
            user_input=user_input,
            messages=[],
            bash=None,
            system_prompt=_LP(identity="I"),
            system_prompt_no_tools=_LP(identity="I"),
            router=_dp(),
            registry=_FakeRegistry(),
            logger=logger,
            turn_number=1,
        )
    finally:
        for k, v in saved_env.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
        mod.rag_snippets = saved_rag
        mod.rag_snippets_with_tier = saved_rag_tier
        mod.render_him_vy_discovery_packet = saved_disc
        mod.render_him_vy_turn_packet = saved_turn
        mod.run_probes = saved_probes
    return captured.get("role") or captured.get("cfg"), logger.events


def test_omni_alias_clamps_max_tokens_below_model_len():
    """@omni must request max_tokens well below the vLLM
    max_model_len=4096 even when the resolved role (chat/orchestrate)
    carries a 16384 budget. The default ceiling is 2048, so the
    requested max_tokens must be <= 2048 and < 4096."""
    role, events = _run_omni_turn_capture_role(
        "@omni summarise this paragraph",
    )
    assert role is not None, "provider was never invoked on @omni turn"
    assert role.max_tokens <= 2048, (
        f"@omni request inherited oversized max_tokens={role.max_tokens}; "
        "expected default ceiling of 2048."
    )
    assert role.max_tokens < 4096, (
        f"@omni request must stay strictly below vLLM max_model_len=4096; "
        f"got max_tokens={role.max_tokens}."
    )
    names = [n for n, _ in events]
    assert "alias_omni_max_tokens_clamped" in names


def test_omni_alias_respects_operator_override_env():
    """VYBN_OMNI_MAX_TOKENS lets the operator pick a different ceiling
    without editing policy. Setting it to 1024 must produce a request
    with max_tokens=1024, still safely under 4096."""
    role, _ = _run_omni_turn_capture_role(
        "@omni summarise",
        extra_env={"VYBN_OMNI_MAX_TOKENS": "1024"},
    )
    assert role is not None
    assert role.max_tokens == 1024, (
        f"VYBN_OMNI_MAX_TOKENS=1024 should set max_tokens=1024; "
        f"got {role.max_tokens}"
    )
    assert role.max_tokens < 4096


def test_omni_clamp_caps_at_role_budget_when_env_is_huge():
    """The clamp never raises max_tokens above the resolved role's
    existing budget — Omni-as-tool, not a resident organ that gets to
    inflate the turn. With VYBN_OMNI_MAX_TOKENS=999999 the request must
    still be <= role.max_tokens. (And, of course, <4096 if the role's
    own budget is larger, which it always is for chat/orchestrate.)"""
    role, _ = _run_omni_turn_capture_role(
        "@omni summarise",
        extra_env={"VYBN_OMNI_MAX_TOKENS": "999999"},
    )
    assert role is not None
    # Without our clamp this would have been 16384 (chat's role budget);
    # with the clamp at min(role.max_tokens, env), the env wins only when
    # smaller than the role budget.
    from harness.substrate import default_policy as _dp
    routed = _dp().classify("@omni summarise")
    role_budget = _dp().roles[routed.role].max_tokens
    assert role.max_tokens <= role_budget


def test_omni_clamp_falls_back_when_env_is_invalid():
    """A non-integer VYBN_OMNI_MAX_TOKENS must not crash; the default
    2048 ceiling applies."""
    role, _ = _run_omni_turn_capture_role(
        "@omni summarise",
        extra_env={"VYBN_OMNI_MAX_TOKENS": "not-a-number"},
    )
    assert role is not None
    assert role.max_tokens <= 2048


def test_omni_clamp_does_not_affect_non_omni_aliases():
    """The clamp must live inside the @omni branch only. A different
    alias (e.g. @opus4.6) on the same turn must keep the role's full
    max_tokens budget — Anthropic Opus has no 4096-cap problem."""
    import importlib.util as _ilu
    import os as _os
    from harness.substrate import default_policy as _dp
    from harness.substrate import LayeredPrompt as _LP

    path = SPARK_DIR / "vybn_spark_agent.py"
    spec = _ilu.spec_from_file_location(
        "vybn_spark_agent_omni_clamp_neg_test", path,
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    saved_rag = mod.rag_snippets
    saved_rag_tier = mod.rag_snippets_with_tier
    saved_disc = mod.render_him_vy_discovery_packet
    saved_turn = mod.render_him_vy_turn_packet
    saved_probes = mod.run_probes
    mod.rag_snippets = lambda *a, **kw: ""
    mod.rag_snippets_with_tier = lambda *a, **kw: ("", "lightweight")
    mod.render_him_vy_discovery_packet = lambda *a, **kw: ""
    mod.render_him_vy_turn_packet = lambda *a, **kw: ""
    mod.run_probes = lambda *a, **kw: []

    captured: dict = {}

    class _FakeHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "ok"
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 0
                out_tokens = 0
                raw_assistant_content = {"role": "assistant", "content": "ok"}
            return _R()

    class _FakeProvider:
        def stream(_s, *, system, messages, tools, role):
            captured["role"] = role
            return _FakeHandle()

    class _FakeRegistry:
        def get(_s, cfg):
            return _FakeProvider()

    class _FakeLogger:
        path = "/dev/null"
        def __init__(_s):
            _s.events = []
        def emit(_s, name, **kw):
            _s.events.append((name, kw))

    prev_omni_max = _os.environ.get("VYBN_OMNI_MAX_TOKENS")
    _os.environ["VYBN_OMNI_MAX_TOKENS"] = "256"
    try:
        policy = _dp()
        candidate_alias = None
        for cand in ("@opus4.6", "@opus", "@sonnet"):
            if policy.model_aliases.get(cand):
                candidate_alias = cand
                break
        if candidate_alias is None:
            return  # nothing to assert; clamp scope is still proven by other tests
        decision = policy.classify(f"{candidate_alias} hello")
        assert decision.alias_used == candidate_alias
        assert decision.model_override is not None
        original_role_budget = policy.roles[decision.role].max_tokens
        logger = _FakeLogger()
        mod.run_agent_loop(
            user_input=f"{candidate_alias} hello",
            messages=[],
            bash=None,
            system_prompt=_LP(identity="I"),
            system_prompt_no_tools=_LP(identity="I"),
            router=policy,
            registry=_FakeRegistry(),
            logger=logger,
            turn_number=1,
        )
        role_seen = captured.get("role")
        assert role_seen is not None, (
            "provider was never invoked on non-@omni alias turn"
        )
        assert role_seen.max_tokens == original_role_budget, (
            f"non-@omni alias {candidate_alias} had max_tokens="
            f"{role_seen.max_tokens}, expected role budget "
            f"{original_role_budget} — clamp leaked outside the @omni branch"
        )
        names = [n for n, _ in logger.events]
        assert "alias_omni_max_tokens_clamped" not in names, (
            f"non-@omni alias {candidate_alias} emitted the @omni clamp "
            "event — clamp branch is firing for the wrong alias"
        )
    finally:
        if prev_omni_max is None:
            _os.environ.pop("VYBN_OMNI_MAX_TOKENS", None)
        else:
            _os.environ["VYBN_OMNI_MAX_TOKENS"] = prev_omni_max
        mod.rag_snippets = saved_rag
        mod.rag_snippets_with_tier = saved_rag_tier
        mod.render_him_vy_discovery_packet = saved_disc
        mod.render_him_vy_turn_packet = saved_turn
        mod.run_probes = saved_probes


# Super maintenance / deferral gate. Operator-armed VYBN_SUPER_MAINTENANCE
# short-circuits any local-Super-bound turn with an honest "paused, retry
# shortly" notice instead of either hanging on the SDK timeout or surfacing
# a raw "(provider error: Connection refused)" string. The same notice
# appears when an in-flight call refuses (Super was bounced by something
# else, or vLLM died). Cloud Anthropic / OpenAI traffic is unaffected.
def _load_agent_module():
    import importlib.util as _ilu
    path = SPARK_DIR / "vybn_spark_agent.py"
    spec = _ilu.spec_from_file_location("vybn_spark_agent_maint_test", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_is_local_super_base_classifies_loopback_and_lan():
    mod = _load_agent_module()
    assert mod._is_local_super_base("http://127.0.0.1:8000/v1") is True
    assert mod._is_local_super_base("http://localhost:8000/v1") is True
    assert mod._is_local_super_base("http://10.0.0.5:8001/v1") is True
    assert mod._is_local_super_base("http://192.168.1.42:8000/v1") is True
    assert mod._is_local_super_base("http://172.16.5.5:8000/v1") is True
    # Public hosts must not trigger the gate.
    assert mod._is_local_super_base("https://api.openai.com/v1") is False
    assert mod._is_local_super_base("https://api.anthropic.com") is False
    assert mod._is_local_super_base(None) is False
    assert mod._is_local_super_base("") is False
    # 172.32.x is outside the 172.16-31 private block.
    assert mod._is_local_super_base("http://203.0.113.1/v1") is False


def test_super_maintenance_state_reads_env_flag_and_file():
    import os as _os
    import tempfile as _tmp
    mod = _load_agent_module()
    prev_flag = _os.environ.get("VYBN_SUPER_MAINTENANCE")
    prev_file = _os.environ.get("VYBN_SUPER_MAINTENANCE_FILE")
    try:
        # Unset -> inactive.
        _os.environ.pop("VYBN_SUPER_MAINTENANCE", None)
        _os.environ.pop("VYBN_SUPER_MAINTENANCE_FILE", None)
        active, _ = mod._super_maintenance_state()
        assert active is False
        # Truthy bare flag -> active with default reason.
        _os.environ["VYBN_SUPER_MAINTENANCE"] = "1"
        active, reason = mod._super_maintenance_state()
        assert active is True
        assert reason
        # Falsy literal -> inactive.
        _os.environ["VYBN_SUPER_MAINTENANCE"] = "0"
        active, _ = mod._super_maintenance_state()
        assert active is False
        # Custom string reason -> active with that string.
        _os.environ["VYBN_SUPER_MAINTENANCE"] = "GPU rebalance for omni window"
        active, reason = mod._super_maintenance_state()
        assert active is True
        assert "GPU rebalance" in reason
        # File overrides bare flag's default reason.
        _os.environ["VYBN_SUPER_MAINTENANCE"] = "1"
        f = _tmp.NamedTemporaryFile("w", suffix=".txt", delete=False)
        try:
            f.write("operator note: bouncing super at 09:00")
            f.flush()
            f.close()
            _os.environ["VYBN_SUPER_MAINTENANCE_FILE"] = f.name
            active, reason = mod._super_maintenance_state()
            assert active is True
            assert "09:00" in reason
        finally:
            try:
                _os.unlink(f.name)
            except OSError:
                pass
            _os.environ.pop("VYBN_SUPER_MAINTENANCE_FILE", None)
    finally:
        if prev_flag is None:
            _os.environ.pop("VYBN_SUPER_MAINTENANCE", None)
        else:
            _os.environ["VYBN_SUPER_MAINTENANCE"] = prev_flag
        if prev_file is None:
            _os.environ.pop("VYBN_SUPER_MAINTENANCE_FILE", None)
        else:
            _os.environ["VYBN_SUPER_MAINTENANCE_FILE"] = prev_file


def test_super_maintenance_gate_short_circuits_local_turn():
    """When VYBN_SUPER_MAINTENANCE is armed and the resolved role's base_url
    is local Super, run_agent_loop returns a maintenance notice without ever
    constructing a provider — no ~10-min restart needed, no raw
    'Connection refused' surfaces to the chat."""
    import os as _os
    from harness.substrate import default_policy as _dp

    mod = _load_agent_module()
    # Lightweight stand-ins so the test exercises only the gate.
    saved_rag = mod.rag_snippets
    saved_rag_tier = mod.rag_snippets_with_tier
    saved_disc = mod.render_him_vy_discovery_packet
    saved_turn = mod.render_him_vy_turn_packet
    saved_probes = mod.run_probes
    mod.rag_snippets = lambda *a, **kw: ""
    mod.rag_snippets_with_tier = lambda *a, **kw: ("", "lightweight")
    mod.render_him_vy_discovery_packet = lambda *a, **kw: ""
    mod.render_him_vy_turn_packet = lambda *a, **kw: ""
    mod.run_probes = lambda *a, **kw: []

    class _FakeRegistry:
        def get(_s, cfg):
            raise AssertionError(
                "maintenance flag must short-circuit BEFORE any provider "
                "is constructed — Super must not be touched"
            )

    class _FakeLogger:
        path = "/dev/null"
        def __init__(_s):
            _s.events = []
        def emit(_s, name, **kw):
            _s.events.append((name, kw))

    prev_flag = _os.environ.get("VYBN_SUPER_MAINTENANCE")
    deferrals_before = len(mod._MAINTENANCE_DEFERRALS)
    try:
        _os.environ["VYBN_SUPER_MAINTENANCE"] = "operator paused for omni window"
        logger = _FakeLogger()
        # /local directive routes to the local_private role whose base_url
        # is http://127.0.0.1:8000/v1 — the canonical Super endpoint.
        reply = mod.run_agent_loop(
            user_input="/local scan him for funders",
            messages=[],
            bash=None,
            system_prompt=None,
            router=_dp(),
            registry=_FakeRegistry(),
            logger=logger,
            turn_number=1,
        )
        assert "maintenance" in reply.lower() or "paused" in reply.lower()
        assert "retry" in reply.lower()
        assert "omni window" in reply
        # Bounded deferral was recorded.
        assert len(mod._MAINTENANCE_DEFERRALS) == deferrals_before + 1
        # The maintenance notice event was emitted.
        names = [n for n, _ in logger.events]
        assert "super_maintenance_notice" in names
    finally:
        if prev_flag is None:
            _os.environ.pop("VYBN_SUPER_MAINTENANCE", None)
        else:
            _os.environ["VYBN_SUPER_MAINTENANCE"] = prev_flag
        # Trim the deferral list back so we don't leak across tests.
        del mod._MAINTENANCE_DEFERRALS[deferrals_before:]
        mod.rag_snippets = saved_rag
        mod.rag_snippets_with_tier = saved_rag_tier
        mod.render_him_vy_discovery_packet = saved_disc
        mod.render_him_vy_turn_packet = saved_turn
        mod.run_probes = saved_probes


def test_super_maintenance_gate_does_not_fire_on_cloud_turn():
    """A cloud Anthropic/OpenAI turn must never be short-circuited by the
    Super maintenance flag — Super pause only applies to the local
    endpoint. We assert this by arming the flag, sending a turn that
    routes to a cloud model, and confirming the registry IS consulted
    (the gate did not short-circuit)."""
    import os as _os
    from harness.substrate import default_policy as _dp

    mod = _load_agent_module()
    saved_rag = mod.rag_snippets
    saved_rag_tier = mod.rag_snippets_with_tier
    saved_disc = mod.render_him_vy_discovery_packet
    saved_turn = mod.render_him_vy_turn_packet
    saved_probes = mod.run_probes
    mod.rag_snippets = lambda *a, **kw: ""
    mod.rag_snippets_with_tier = lambda *a, **kw: ("", "lightweight")
    mod.render_him_vy_discovery_packet = lambda *a, **kw: ""
    mod.render_him_vy_turn_packet = lambda *a, **kw: ""
    mod.run_probes = lambda *a, **kw: []

    consulted = {"count": 0}

    class _FakeHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "ok"
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 0
                out_tokens = 0
                raw_assistant_content = {"role": "assistant", "content": "ok"}
            return _R()

    class _FakeProvider:
        def stream(_s, *, system, messages, tools, role):
            return _FakeHandle()

    class _FakeRegistry:
        def get(_s, cfg):
            consulted["count"] += 1
            return _FakeProvider()

    class _FakeLogger:
        path = "/dev/null"
        def __init__(_s):
            _s.events = []
        def emit(_s, name, **kw):
            _s.events.append((name, kw))

    prev_flag = _os.environ.get("VYBN_SUPER_MAINTENANCE")
    try:
        _os.environ["VYBN_SUPER_MAINTENANCE"] = "1"
        # @sonnet pins a cloud Anthropic model — base_url is None.
        logger = _FakeLogger()
        mod.run_agent_loop(
            user_input="@sonnet hi",
            messages=[],
            bash=None,
            system_prompt=None,
            router=_dp(),
            registry=_FakeRegistry(),
            logger=logger,
            turn_number=1,
        )
        assert consulted["count"] >= 1, (
            "cloud turn must reach the provider; super-maintenance must not "
            "block non-local routes"
        )
        names = [n for n, _ in logger.events]
        assert "super_maintenance_notice" not in names
    finally:
        if prev_flag is None:
            _os.environ.pop("VYBN_SUPER_MAINTENANCE", None)
        else:
            _os.environ["VYBN_SUPER_MAINTENANCE"] = prev_flag
        mod.rag_snippets = saved_rag
        mod.rag_snippets_with_tier = saved_rag_tier
        mod.render_him_vy_discovery_packet = saved_disc
        mod.render_him_vy_turn_packet = saved_turn
        mod.run_probes = saved_probes


def test_super_refusal_converts_to_maintenance_notice():
    """Without the maintenance flag set, a transport refusal from local
    Super (e.g. vLLM crashed or rebooting) gets converted to the same
    visible "paused, retry shortly" notice instead of surfacing as
    '(provider error: ConnectionError: Connection refused)'. The
    classification mirrors OpenAIProvider._call's transport_signals."""
    import os as _os
    from harness.substrate import default_policy as _dp

    mod = _load_agent_module()
    saved_rag = mod.rag_snippets
    saved_rag_tier = mod.rag_snippets_with_tier
    saved_disc = mod.render_him_vy_discovery_packet
    saved_turn = mod.render_him_vy_turn_packet
    saved_probes = mod.run_probes
    mod.rag_snippets = lambda *a, **kw: ""
    mod.rag_snippets_with_tier = lambda *a, **kw: ("", "lightweight")
    mod.render_him_vy_discovery_packet = lambda *a, **kw: ""
    mod.render_him_vy_turn_packet = lambda *a, **kw: ""
    mod.run_probes = lambda *a, **kw: []
    saved_gate = mod._local_super_semantic_gate
    mod._local_super_semantic_gate = lambda **kw: (True, "semantic gate passed")

    class _RefusingProvider:
        def stream(_s, *, system, messages, tools, role):
            raise ConnectionError(
                "HTTPConnectionPool(host='127.0.0.1', port=8000): "
                "Max retries exceeded — connection refused"
            )

    class _FakeRegistry:
        def get(_s, cfg):
            return _RefusingProvider()

    class _FakeLogger:
        path = "/dev/null"
        def __init__(_s):
            _s.events = []
        def emit(_s, name, **kw):
            _s.events.append((name, kw))

    prev_flag = _os.environ.get("VYBN_SUPER_MAINTENANCE")
    deferrals_before = len(mod._MAINTENANCE_DEFERRALS)
    try:
        _os.environ.pop("VYBN_SUPER_MAINTENANCE", None)
        logger = _FakeLogger()
        reply = mod.run_agent_loop(
            user_input="/local scan him for funders",
            messages=[],
            bash=None,
            system_prompt=None,
            router=_dp(),
            registry=_FakeRegistry(),
            logger=logger,
            turn_number=1,
        )
        assert "Super" in reply or "maintenance" in reply.lower() or "paused" in reply.lower()
        assert "retry" in reply.lower()
        # Raw "(provider error: …)" must NOT leak through anymore for this
        # transport class against local Super.
        assert "(provider error" not in reply.lower()
        # Deferral and event wiring fired.
        assert len(mod._MAINTENANCE_DEFERRALS) == deferrals_before + 1
        names = [n for n, _ in logger.events]
        assert "super_maintenance_notice" in names
        # The provider_error event still fires (with the conversion flag),
        # so observability into refusals is preserved.
        assert "provider_error" in names
    finally:
        if prev_flag is not None:
            _os.environ["VYBN_SUPER_MAINTENANCE"] = prev_flag
        del mod._MAINTENANCE_DEFERRALS[deferrals_before:]
        mod.rag_snippets = saved_rag
        mod.rag_snippets_with_tier = saved_rag_tier
        mod.render_him_vy_discovery_packet = saved_disc
        mod.render_him_vy_turn_packet = saved_turn
        mod.run_probes = saved_probes
        mod._local_super_semantic_gate = saved_gate


def test_maintenance_deferrals_are_bounded():
    """The in-memory deferral list never grows past the cap. This is a
    deferral notice, not a durable queue — bounded length is part of the
    honesty contract."""
    mod = _load_agent_module()
    saved = list(mod._MAINTENANCE_DEFERRALS)
    try:
        mod._MAINTENANCE_DEFERRALS.clear()
        cap = mod._MAINTENANCE_DEFERRALS_CAP
        for i in range(cap + 25):
            mod._format_super_maintenance_notice(
                reason=f"reason-{i}",
                cause="flag",
                role_model="nemotron",
                role_base="http://127.0.0.1:8000/v1",
                logger=None,
                turn_number=i,
            )
        assert len(mod._MAINTENANCE_DEFERRALS) == cap
    finally:
        mod._MAINTENANCE_DEFERRALS.clear()
        mod._MAINTENANCE_DEFERRALS.extend(saved)




def test_loopback_super_semantic_gate_is_narrower_than_local_maintenance_gate():
    mod = _load_agent_module()
    assert mod._is_loopback_super_base("http://127.0.0.1:8000/v1") is True
    assert mod._is_loopback_super_base("http://localhost:8000/v1") is True
    assert mod._is_loopback_super_base("http://example.invalid:8002/v1") is False
    assert mod._is_local_super_base("http://example.invalid:8002/v1") is False


def test_local_super_semantic_gate_rejects_empty_truncated_and_wrong_outputs(monkeypatch):
    import json as _json

    mod = _load_agent_module()
    base = "http://127.0.0.1:8000/v1"
    model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    cases = [
        ({"choices": [{"finish_reason": "stop", "text": ""}]}, "known_answer", "empty"),
        ({"choices": [{"finish_reason": "length", "text": " FOUR"}]}, "known_answer", "truncated"),
        ({"choices": [{"finish_reason": "stop", "text": " quatre"}]}, "known_answer", "unexpected"),
    ]
    for idx, (payload, probe_name, needle) in enumerate(cases):
        mod._SUPER_SEMANTIC_GATE_CACHE.clear()
        class _Resp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return _json.dumps(payload).encode()
        monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *a, **kw: _Resp())
        ok, reason = mod._local_super_semantic_gate(base_url=base, model=model, now=float(idx))
        assert ok is False
        assert probe_name in reason
        assert needle in reason


def test_local_super_semantic_gate_rejects_structured_and_reasoning_corruption(monkeypatch):
    import json as _json

    mod = _load_agent_module()
    base = "http://127.0.0.1:8000/v1"
    model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    cases = [
        (
            [
                {"choices": [{"finish_reason": "stop", "text": " FOUR"}]},
                {"choices": [{"finish_reason": "stop", "text": " status ok"}]},
            ],
            "structured_shape",
        ),
        (
            [
                {"choices": [{"finish_reason": "stop", "text": " FOUR"}]},
                {"choices": [{"finish_reason": "stop", "text": ' {"status":"ok"}'}]},
                {"choices": [{"finish_reason": "stop", "text": " PASS"}]},
            ],
            "wake_reasoning",
        ),
    ]
    for idx, (payloads, probe_name) in enumerate(cases):
        mod._SUPER_SEMANTIC_GATE_CACHE.clear()
        responses = list(payloads)

        class _Resp:
            def __init__(self, payload):
                self.payload = payload
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return _json.dumps(self.payload).encode()

        def fake_urlopen(*a, **kw):
            return _Resp(responses.pop(0))

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
        ok, reason = mod._local_super_semantic_gate(base_url=base, model=model, now=100.0 + idx)
        assert ok is False
        assert probe_name in reason
        assert "unexpected" in reason


def test_local_super_semantic_gate_accepts_expected_output_and_caches(monkeypatch):
    import json as _json

    mod = _load_agent_module()
    mod._SUPER_SEMANTIC_GATE_CACHE.clear()
    calls = {"n": 0}
    responses = [
        {"choices": [{"finish_reason": "stop", "text": " </think>\n\nFOUR"}]},
        {"choices": [{"finish_reason": "stop", "text": ' {"status":"ok"}'}]},
        {"choices": [{"finish_reason": "stop", "text": " FAIL"}]},
    ]
    class _Resp:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return _json.dumps(self.payload).encode()
    def fake_urlopen(*a, **kw):
        calls["n"] += 1
        return _Resp(responses.pop(0))
    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    base = "http://127.0.0.1:8000/v1"
    model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    first_ok, first_reason = mod._local_super_semantic_gate(base_url=base, model=model, now=10.0)
    assert first_ok is True, first_reason
    second_ok, second_reason = mod._local_super_semantic_gate(base_url=base, model=model, now=11.0)
    assert second_ok is True, second_reason
    assert calls["n"] == 3


def test_local_super_semantic_failure_fails_closed_without_cloud_fallback(monkeypatch):
    import os as _os
    from harness.substrate import default_policy as _dp

    mod = _load_agent_module()
    saved_rag = mod.rag_snippets
    saved_rag_tier = mod.rag_snippets_with_tier
    saved_disc = mod.render_him_vy_discovery_packet
    saved_turn = mod.render_him_vy_turn_packet
    saved_probes = mod.run_probes
    mod.rag_snippets = lambda *a, **kw: ""
    mod.rag_snippets_with_tier = lambda *a, **kw: ("", "lightweight")
    mod.render_him_vy_discovery_packet = lambda *a, **kw: ""
    mod.render_him_vy_turn_packet = lambda *a, **kw: ""
    mod.run_probes = lambda *a, **kw: []
    monkeypatch.setattr(mod, "_local_super_semantic_gate", lambda **kw: (False, "unexpected content='garbage'"))

    class _FakeRegistry:
        def get(self, cfg):
            raise AssertionError("semantic corruption must fail closed before provider construction unless cloud fallback is explicitly opted in")
    class _FakeLogger:
        path = "/dev/null"
        def __init__(self):
            self.events = []
        def emit(self, name, **kw):
            self.events.append((name, kw))

    prev_flag = _os.environ.get("VYBN_SUPER_MAINTENANCE")
    try:
        _os.environ.pop("VYBN_SUPER_MAINTENANCE", None)
        logger = _FakeLogger()
        _os.environ.pop("VYBN_SUPER_SEMANTIC_FALLBACK", None)
        reply = mod.run_agent_loop(
            user_input="/local scan him for funders",
            messages=[],
            bash=None,
            system_prompt=None,
            router=_dp(),
            registry=_FakeRegistry(),
            logger=logger,
            turn_number=1,
        )
        assert "retry" in reply.lower()
        assert "cloud fallback is disabled" in reply
        names = [n for n, _ in logger.events]
        assert "super_semantic_gate_failed_closed" in names
        assert "super_semantic_gate_fallback" not in names
    finally:
        if prev_flag is not None:
            _os.environ["VYBN_SUPER_MAINTENANCE"] = prev_flag
        mod.rag_snippets = saved_rag
        mod.rag_snippets_with_tier = saved_rag_tier
        mod.render_him_vy_discovery_packet = saved_disc
        mod.render_him_vy_turn_packet = saved_turn
        mod.run_probes = saved_probes


def test_local_super_semantic_failure_cloud_fallback_requires_explicit_opt_in(monkeypatch):
    import os as _os
    from harness.substrate import default_policy as _dp

    mod = _load_agent_module()
    saved_rag = mod.rag_snippets
    saved_rag_tier = mod.rag_snippets_with_tier
    saved_disc = mod.render_him_vy_discovery_packet
    saved_turn = mod.render_him_vy_turn_packet
    saved_probes = mod.run_probes
    mod.rag_snippets = lambda *a, **kw: ""
    mod.rag_snippets_with_tier = lambda *a, **kw: ("", "lightweight")
    mod.render_him_vy_discovery_packet = lambda *a, **kw: ""
    mod.render_him_vy_turn_packet = lambda *a, **kw: ""
    mod.run_probes = lambda *a, **kw: []
    monkeypatch.setattr(mod, "_local_super_semantic_gate", lambda **kw: (False, "unexpected content='garbage'"))

    consulted = []
    class _FakeHandle:
        def __iter__(self):
            return iter([])
        def final(self):
            class _R:
                text = "sonnet fallback ok"
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 0
                out_tokens = 0
                raw_assistant_content = {"role": "assistant", "content": "sonnet fallback ok"}
            return _R()
    class _FakeProvider:
        def stream(self, *, system, messages, tools, role):
            return _FakeHandle()
    class _FakeRegistry:
        def get(self, cfg):
            consulted.append((cfg.provider, cfg.model, cfg.base_url))
            return _FakeProvider()
    class _FakeLogger:
        path = "/dev/null"
        def __init__(self):
            self.events = []
        def emit(self, name, **kw):
            self.events.append((name, kw))

    prev_flag = _os.environ.get("VYBN_SUPER_MAINTENANCE")
    prev_fb = _os.environ.get("VYBN_SUPER_SEMANTIC_FALLBACK")
    try:
        _os.environ.pop("VYBN_SUPER_MAINTENANCE", None)
        _os.environ["VYBN_SUPER_SEMANTIC_FALLBACK"] = "sonnet"
        logger = _FakeLogger()
        reply = mod.run_agent_loop(
            user_input="/local scan him for funders",
            messages=[],
            bash=None,
            system_prompt=None,
            router=_dp(),
            registry=_FakeRegistry(),
            logger=logger,
            turn_number=1,
        )
        assert "sonnet fallback ok" in reply
        assert consulted[0][0] == "anthropic"
        assert consulted[0][1] == "claude-sonnet-4-6"
        assert consulted[0][2] is None
        assert "super_semantic_gate_fallback" in [n for n, _ in logger.events]
    finally:
        if prev_flag is None:
            _os.environ.pop("VYBN_SUPER_MAINTENANCE", None)
        else:
            _os.environ["VYBN_SUPER_MAINTENANCE"] = prev_flag
        if prev_fb is None:
            _os.environ.pop("VYBN_SUPER_SEMANTIC_FALLBACK", None)
        else:
            _os.environ["VYBN_SUPER_SEMANTIC_FALLBACK"] = prev_fb
        mod.rag_snippets = saved_rag
        mod.rag_snippets_with_tier = saved_rag_tier
        mod.render_him_vy_discovery_packet = saved_disc
        mod.render_him_vy_turn_packet = saved_turn
        mod.run_probes = saved_probes

