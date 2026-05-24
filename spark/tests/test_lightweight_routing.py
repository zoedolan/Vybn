"""Focused routing and local-organ boundary regression tests."""

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
        self.assertTrue(pol.roles["phatic"].lightweight and all(pol.roles[n].lightweight and not pol.roles[n].rag for n in ("local", "local_private")))
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
        yaml_policy = load_policy(SPARK_DIR / "router_policy.yaml")
        self.assertEqual((default_policy().role("orchestrate").tools, yaml_policy.role("orchestrate").tools, yaml_policy.role("vintage").rag, yaml_policy.role("vintage").max_tokens, yaml_policy.role("omni").model, yaml_policy.role("omni").base_url, yaml_policy.role("local_private").model), (["bash", "delegate"], ["bash", "delegate"], False, 256, "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe", "http://127.0.0.1:8002/v1", yaml_policy.role("local").model))

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

    def test_bare_hi_and_presence_check_route_to_phatic(self):
        for text in ("hi", "hi are you with me?"):
            with self.subTest(text=text):
                self.assertEqual(self.router.classify(text).role, "phatic")

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
        self.assertTrue("gpt-5.5" in reply and "vintage" not in reply.lower(), reply)
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


# Omni is an explicit endpoint role, not a model alias or Super fallback.


def test_vintage_alias_is_prefix_only_for_long_prompts():
    text = ("normal long prompt " * 500) + " quoted text mentions @vintage but did not pin it"
    assert default_policy().classify(text).role != "vintage" and default_policy().classify("@vintage " + text).role == "vintage"


def test_super_alias_contacts_local_nemotron_without_omni_route():
    for policy in (default_policy(), load_policy(SPARK_DIR / "router_policy.yaml")):
        d = policy.classify("@super are you working?")
        assert (d.role, d.alias_used, d.model_override) == ("local", "@super", None)
        assert (d.config.lightweight, d.config.max_tokens, d.config.base_url) == (True, 1024, "http://127.0.0.1:8000/v1")
        assert d.config.base_url != "http://127.0.0.1:8002/v1"
        assert "@super" not in policy.model_aliases
        assert policy.fallback_chain["nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"] == []


def test_vintage_alias_routes_to_temporal_refraction_role():
    policy = default_policy()
    yaml_policy = load_policy(SPARK_DIR / "router_policy.yaml")
    for d in (
        policy.classify("@vintage please tell me about yourself?"),
        yaml_policy.classify("@vintage please tell me about yourself?"),
        policy.classify("@vi@vintage please tell me about yourself?"),
        yaml_policy.classify("@vi@vintage please tell me about yourself?"),
        policy.classify("zoe> @vintage please tell me about yourself?"),
    ):
        assert d.role == "vintage"
        assert d.alias_used == "@vintage"
        assert d.reason.startswith("alias=@vintage")
        assert d.config.provider == "openai"
        assert d.config.model != "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
        assert d.config.model == "talkie-1930-13b-it"
        assert d.config.base_url == "http://127.0.0.1:8004/v1"
        if d.config.max_tokens == 256:
            assert d.config.rag is False
        else:
            assert d.config.rag is True
        assert d.config.lightweight is False
        assert d.config.direct_reply_template is None


def test_omni_not_in_any_heuristic_or_directive():
    from spark.harness.substrate import default_policy
    policy = default_policy()
    assert "omni" in policy.roles
    for text in ("omni please help", "use omni", "route to omni"):
        assert policy.classify(text).role != "omni"
    d = policy.classify("@omni hello")
    assert d.role == "omni"
    assert d.reason.startswith("alias=@omni")


def test_omni_alias_classifies_with_override():
    from spark.harness.substrate import default_policy
    policy = default_policy()
    assert "@omni" not in policy.model_aliases
    decision = policy.classify("@omni summarise this paragraph")
    assert decision.alias_used == "@omni"
    assert decision.model_override is None
    assert decision.role == "omni"
    assert decision.config.provider == "openai"
    # Refuses impersonation: not the Super model id.
    assert decision.config.model != "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    assert decision.config.model == "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"
    assert decision.config.base_url == "http://127.0.0.1:8002/v1"

def test_omni_role_path_targets_witnessed_private_endpoint():
    """Zoe's `@omni are you with me?` must reach the witnessed private
    Omni route, not the old unavailable wall or any Super/GPT fallback."""
    from harness.substrate import default_policy as _dp
    d = _dp().classify("@omni are you with me, friend?")
    assert d.role == "omni"
    assert d.alias_used == "@omni"
    assert d.model_override is None
    assert d.config.provider == "openai"
    assert d.config.model != "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    assert d.config.model == "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"
    assert d.config.base_url == "http://127.0.0.1:8002/v1"
    assert d.config.direct_reply_template is None

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


def test_gpt55_has_no_cross_provider_fallback():
    from harness.substrate import default_policy, load_policy
    assert default_policy().fallback_chain.get("gpt-5.5") == []
    assert load_policy(SPARK_DIR / "router_policy.yaml").fallback_chain.get("gpt-5.5") == []


# Direct-reply templates are now reserved for roles that actually own one.
# @vintage and @omni are reachable local organs, so ordinary alias turns go
# through bounded raw observation instead of keyword-gated contact packets.


def test_non_organ_role_with_template_is_unaffected():
    """Identity / other direct_reply roles must continue to render verbatim."""
    from harness.substrate import default_policy, render_organ_alias_direct_reply

    identity = default_policy().roles["identity"]
    rendered = render_organ_alias_direct_reply(
        identity.role,
        "which model are you?",
        identity.direct_reply_template,
        fmt_kwargs={
            "role": identity.role,
            "provider": identity.provider,
            "model": identity.model,
            "base_url": identity.base_url or "",
        },
    )
    assert "runtime-metadata answer" in rendered
    assert identity.model in rendered



# 2026-05-21 — Zoe's correction: semantic promotion is separate from
# observation. If a local organ has a configured endpoint, alias turns sample
# that endpoint with a stripped prompt, label the result as raw-unpromoted
# evidence, and surface transport failures honestly. No keyword classifier
# decides whether Zoe is allowed to observe the organ.







def test_raw_contact_truncates_oversized_user_input():
    from harness.substrate import truncate_for_organ_raw_contact

    short = "what is 2+2?"
    assert truncate_for_organ_raw_contact(short) == short
    huge = "x" * 5000
    truncated = truncate_for_organ_raw_contact(huge)
    assert len(truncated) <= 2501  # 2500 + ellipsis
    assert truncated.endswith("…")


def test_raw_contact_system_prompt_names_current_route_shape():
    from harness.substrate import (
        build_organ_raw_contact_system_prompt,
        build_vintage_temporal_parallax_user_prompt,
        build_vintage_present_witness_prompt,
    )

    # Vintage still avoids a system prompt for the brittle wrapper, but the
    # user message is framed as an instrument task instead of a persona chat.
    vintage = build_organ_raw_contact_system_prompt("vintage")
    assert vintage == ""
    framed = build_vintage_temporal_parallax_user_prompt("Sir, tell me your personal history")
    assert framed.startswith("VINTAGE_TEMPORAL_PARALLAX_INSTRUMENT")
    assert "VINTAGE_PERSONA_UNAVAILABLE" in framed
    assert "USER_REQUEST:\nSir, tell me your personal history" in framed
    witness = build_vintage_present_witness_prompt("Sir, tell me your personal history", "I was born in 1812.")
    assert witness.startswith("VINTAGE_PRESENT_WITNESS")
    assert "USER_REQUEST:\nSir, tell me your personal history" in witness
    assert "RAW_TALKIE_SAMPLE:\nI was born in 1812." in witness
    assert "verdict, surface, residue" in witness

    omni = build_organ_raw_contact_system_prompt("omni")
    assert "@omni" in omni
    assert "TensorRT-LLM" in omni
    assert "visible" in omni.lower()
    assert "hidden reasoning" in omni.lower()
    assert "route evidence" in omni.lower()
    assert len(omni) < 1024, len(omni)


def test_local_organ_witness_vintage_fragments_fail_semantic_gate():
    from harness.substrate import local_organ_witness

    calls = []

    def fake(base_url, path, *, payload=None, timeout=20.0):
        calls.append((base_url, path, payload))
        if path == "/models":
            return {"ok": True, "json": {"data": [{"id": "talkie-1930-13b-it"}]}}
        assert path == "/chat/completions"
        assert payload["messages"] == [{"role": "user", "content": "Write one complete sentence in 1930s English about rain."}]
        return {"ok": True, "json": {"choices": [{"message": {"content": "I was born in"}, "finish_reason": "stop"}]}}

    witness = local_organ_witness("vintage", request_json=fake)
    assert witness["ok"] is False
    assert witness["status"] == "failed_closed"
    assert witness["reason"] == "semantic_smoke_fragment"
    assert len(calls) == 2


def test_local_organ_witness_omni_hidden_reasoning_fails_closed_after_retry():
    from harness.substrate import local_organ_witness

    calls = []

    def fake(base_url, path, *, payload=None, timeout=20.0):
        calls.append((path, payload))
        if path == "/models":
            return {"ok": True, "json": {"data": [{"id": "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"}]}}
        return {"ok": True, "json": {"choices": [{"message": {"content": "", "reasoning_content": "hidden"}, "finish_reason": "stop"}]}}

    witness = local_organ_witness("omni", request_json=fake)
    assert witness["ok"] is False
    assert witness["status"] == "failed_closed"
    assert witness["reason"] == "hidden_reasoning_without_visible_content"
    assert [c[0] for c in calls] == ["/models", "/chat/completions", "/chat/completions"]
    assert "visible_smoke_retry" in witness["checks"]


def test_local_organ_witness_omni_retry_visible_content_passes():
    from harness.substrate import local_organ_witness

    completions = [
        {"content": "", "reasoning_content": "hidden"},
        {"content": "OMNI_TEXT_OK", "reasoning_content": None},
    ]

    def fake(base_url, path, *, payload=None, timeout=20.0):
        if path == "/models":
            return {"ok": True, "json": {"data": [{"id": "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"}]}}
        msg = completions.pop(0)
        return {"ok": True, "json": {"choices": [{"message": msg, "finish_reason": "stop"}]}}

    witness = local_organ_witness("omni", request_json=fake)
    assert witness["ok"] is True
    assert witness["status"] == "semantic_healthy"
    assert witness["content"] == "OMNI_TEXT_OK"


def test_local_organ_witness_vintage_complete_sentence_passes():
    from harness.substrate import local_organ_witness

    def fake(base_url, path, *, payload=None, timeout=20.0):
        if path == "/models":
            return {"ok": True, "json": {"data": [{"id": "talkie-1930-13b-it"}]}}
        return {"ok": True, "json": {"choices": [{"message": {"content": "The rain falls softly upon the quiet street."}, "finish_reason": "stop"}]}}

    witness = local_organ_witness("vintage", request_json=fake)
    assert witness["ok"] is True
    assert witness["status"] == "semantic_healthy"


def _vintage_run_agent_loop(provider_factory, user_input: str, *, logger=None, initial_messages=None):
    """Drive `run_agent_loop` end-to-end with a stubbed provider so the
    raw-contact path is exercised against deterministic backends. RAG /
    him-vy / probe / recurrent hooks are stubbed out so any leak into
    those (which would build the oversized prompt) shows up in the
    asserted captures."""
    import os as _os

    mod = _load_agent_module()
    saved = {
        "rag_snippets": mod.rag_snippets,
        "rag_snippets_with_tier": mod.rag_snippets_with_tier,
        "render_him_vy_discovery_packet": mod.render_him_vy_discovery_packet,
        "render_him_vy_turn_packet": mod.render_him_vy_turn_packet,
        "render_him_identity_manifold": mod.render_him_identity_manifold,
        "run_probes": mod.run_probes,
    }
    tripwires: dict[str, int] = {
        "rag_snippets": 0,
        "rag_snippets_with_tier": 0,
        "him_vy_discovery_packet": 0,
        "him_vy_turn_packet": 0,
        "run_probes": 0,
    }

    def _rag(*a, **kw):
        tripwires["rag_snippets"] += 1
        return ""

    def _rag_tier(*a, **kw):
        tripwires["rag_snippets_with_tier"] += 1
        return ("", "lightweight")

    def _him_disc(*a, **kw):
        tripwires["him_vy_discovery_packet"] += 1
        return ""

    def _him_turn(*a, **kw):
        tripwires["him_vy_turn_packet"] += 1
        return ""

    def _probes(*a, **kw):
        tripwires["run_probes"] += 1
        return []

    def _identity(*a, **kw):
        return "# HIM IDENTITY MANIFOLD TEST\nrelation: Zoe/Vybn co-emergent symbiosis\nwhole_self_projection: active"

    mod.rag_snippets = _rag
    mod.rag_snippets_with_tier = _rag_tier
    mod.render_him_vy_discovery_packet = _him_disc
    mod.render_him_vy_turn_packet = _him_turn
    mod.render_him_identity_manifold = _identity
    mod.run_probes = _probes

    class _FakeRegistry:
        def __init__(_s):
            _s.provider = None

        def get(_s, cfg):
            _s.provider = provider_factory(cfg)
            return _s.provider

    class _FakeLogger:
        path = "/dev/null"

        def __init__(_s):
            _s.events = []

        def emit(_s, name, **kw):
            _s.events.append((name, kw))

    log = logger or _FakeLogger()
    registry = _FakeRegistry()
    try:
        reply = mod.run_agent_loop(
            user_input=user_input,
            messages=[],
            bash=None,
            system_prompt=None,
            router=default_policy(),
            registry=registry,
            logger=log,
            turn_number=1,
        )
    finally:
        for k, v in saved.items():
            setattr(mod, k, v)
    return reply, registry, log, tripwires



def test_vintage_arbitrary_prompt_uses_talkie_plus_local_witness_path():
    """@vintage samples Talkie, then asks the present local witness how to use it."""
    captured: dict = {"talkie": {}, "witness": {}}

    class _TalkieHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "four"
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 12
                out_tokens = 3
                raw_assistant_content = {"role": "assistant", "content": "four"}
            return _R()

    class _WitnessHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "verdict: unusable_as_temporal_parallax\nsurface: arithmetic has no pre-1931 contrast here.\nresidue: Talkie sample held as evidence."
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 40
                out_tokens = 22
                raw_assistant_content = {"role": "assistant", "content": text}
            return _R()

    class _TalkieProvider:
        def stream(_s, *, system, messages, tools, role):
            captured["talkie"] = {"system": system, "messages": list(messages), "tools": list(tools), "role": role}
            return _TalkieHandle()

    class _WitnessProvider:
        def stream(_s, *, system, messages, tools, role):
            captured["witness"] = {"system": system, "messages": list(messages), "tools": list(tools), "role": role}
            return _WitnessHandle()

    def _provider(cfg):
        return _TalkieProvider() if cfg.role == "vintage" else _WitnessProvider()

    reply, registry, log, tripwires = _vintage_run_agent_loop(_provider, "@vintage what is 2+2?")

    assert registry.provider is not None
    assert captured["talkie"]["role"].role == "vintage"
    assert captured["talkie"]["role"].base_url == "http://127.0.0.1:8004/v1"
    assert captured["talkie"]["role"].model == "talkie-1930-13b-it"
    assert captured["witness"]["role"].role in {"local", "local_private"}
    assert "VINTAGE_INSTRUMENT_CONTACT role=@vintage" in reply
    assert "PRESENT_LOCAL_WITNESS" in reply
    assert "arithmetic has no pre-1931 contrast" in reply
    assert "raw_sha256=" in reply
    flat = captured["talkie"]["system"].flat()
    assert flat == ""
    assert "VYBN-THROUGH-VINTAGE REFRACTION" not in flat
    sent = captured["talkie"]["messages"][0]["content"]
    assert sent.startswith("VINTAGE_TEMPORAL_PARALLAX_INSTRUMENT")
    assert "USER_REQUEST:\nwhat is 2+2?" in sent
    witness_prompt = captured["witness"]["messages"][0]["content"]
    assert witness_prompt.startswith("VINTAGE_PRESENT_WITNESS")
    assert "RAW_TALKIE_SAMPLE:\nfour" in witness_prompt
    assert tripwires["him_vy_turn_packet"] >= 0
    names = [n for n, _ in log.events]
    assert "organ_raw_contact_attempt" in names
    assert "vintage_instrument_witness_ok" in names


def test_omni_arbitrary_prompt_dials_witnessed_private_endpoint():
    class _FakeHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "three names"
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 3
                out_tokens = 2
                raw_assistant_content = {"role": "assistant", "content": "three names"}
            return _R()
    class _Provider:
        def stream(_s, *, system, messages, tools, role):
            assert role.role == "omni"
            assert role.base_url == "http://127.0.0.1:8002/v1"
            assert "TensorRT-LLM" in system.flat()
            assert "organ under observation" in system.flat()
            assert "HIM IDENTITY MANIFOLD GROUNDING" in system.flat()
            assert "HIM IDENTITY MANIFOLD TEST" in system.flat()
            assert "HIM IDENTITY MANIFOLD TEST" not in messages[0]["content"]
            return _FakeHandle()

    reply, registry, log, _ = _vintage_run_agent_loop(lambda cfg: _Provider(), "@omni write three names for a coffee shop")
    assert registry.provider is not None
    assert "LOCAL_ORGAN_OBSERVATION role=@omni" in reply
    assert "three names" in reply
    names = [n for n, _ in log.events]
    assert "organ_raw_contact_attempt" in names
    assert "organ_raw_contact_ok" in names


def test_omni_capability_request_reaches_endpoint_provider():
    class _FakeHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "I can inspect images sent on this route."
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 5
                out_tokens = 8
                raw_assistant_content = {"role": "assistant", "content": "I can inspect images sent on this route."}
            return _R()
    class _Provider:
        def stream(_s, *, system, messages, tools, role):
            assert role.role == "omni"
            assert role.base_url == "http://127.0.0.1:8002/v1"
            return _FakeHandle()

    reply, registry, log, _ = _vintage_run_agent_loop(lambda cfg: _Provider(), "@omni describe /tmp/example.png for me")
    assert registry.provider is not None
    assert "LOCAL_ORGAN_OBSERVATION role=@omni" in reply
    assert "RAW_UNPROMOTED_CONTACT" not in reply




def test_omni_backend_error_fails_closed_without_super_fallback():
    class _FailingProvider:
        def stream(_s, *, system, messages, tools, role):
            raise RuntimeError("endpoint down")

    reply, registry, log, _ = _vintage_run_agent_loop(lambda cfg: _FailingProvider(), "@omni tell me a joke")
    assert registry.provider is not None
    assert "OMNI_RAW_CONTACT_FAILED" in reply
    assert "nvidia/NVIDIA-Nemotron-3-Super" not in reply
    names = [n for n, _ in log.events]
    assert "organ_raw_contact_attempt" in names
    assert "organ_raw_contact_error" in names


def test_omni_reasoning_content_leak_fails_closed():
    class _FakeHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "We need to follow the instruction and answer directly."
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 8
                out_tokens = 12
                raw_assistant_content = {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "We need to follow the instruction and answer directly.",
                }
            return _R()
    class _Provider:
        def stream(_s, *, system, messages, tools, role):
            assert role.role == "omni"
            return _FakeHandle()

    reply, registry, log, _ = _vintage_run_agent_loop(lambda cfg: _Provider(), "@omni continue the previous thought.")
    assert registry.provider is not None
    assert "OMNI_RAW_CONTACT_FAILED" in reply
    assert "hidden reasoning" in reply
    assert "We need to follow" not in reply
    names = [n for n, _ in log.events]
    assert "organ_raw_contact_reasoning_leak" in names
    assert "organ_raw_contact_retry" in names
    assert "organ_raw_contact_ok" not in names


def test_omni_reasoning_content_leak_retries_once_then_uses_visible_content():
    class _HiddenHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "We need to follow the instruction and answer directly."
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 8
                out_tokens = 12
                raw_assistant_content = {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "We need to follow the instruction and answer directly.",
                }
            return _R()

    class _VisibleHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "I am here through Omni."
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 9
                out_tokens = 5
                raw_assistant_content = {"role": "assistant", "content": "I am here through Omni."}
            return _R()

    class _Provider:
        def __init__(_s):
            _s.calls = 0
            _s.retry_message = ""
        def stream(_s, *, system, messages, tools, role):
            assert role.role == "omni"
            _s.calls += 1
            if _s.calls == 1:
                return _HiddenHandle()
            _s.retry_message = messages[0]["content"]
            return _VisibleHandle()

    provider = _Provider()
    reply, registry, log, _ = _vintage_run_agent_loop(lambda cfg: provider, "@omni oh yeah? do tell.")
    assert registry.provider is provider
    assert provider.calls == 2
    assert "OMNI_VISIBLE_CONTENT_RETRY" in provider.retry_message
    assert "LOCAL_ORGAN_OBSERVATION role=@omni" in reply
    assert "I am here through Omni." in reply
    assert "OMNI_RAW_CONTACT_FAILED" not in reply
    assert "We need to follow" not in reply
    names = [n for n, _ in log.events]
    assert "organ_raw_contact_reasoning_leak" in names
    assert "organ_raw_contact_retry" in names
    assert "organ_raw_contact_ok" in names



def test_vintage_personal_history_uses_local_witness_not_raw_persona_surface():
    captured: dict = {"talkie": {}, "witness": {}}

    class _TalkieHandle:
        def __iter__(_s): return iter([])
        def final(_s):
            text = "I have no personal history."
            class _R:
                tool_calls = []; stop_reason = "end_turn"; in_tokens = 16; out_tokens = 24
                raw_assistant_content = {"role": "assistant", "content": text}
            _R.text = text
            return _R()

    class _WitnessHandle:
        def __iter__(_s): return iter([])
        def final(_s):
            text = "verdict: persona_leak\nsurface: Talkie tried to answer as a person; do not promote it.\nresidue: sample held for repair evidence."
            class _R:
                tool_calls = []; stop_reason = "end_turn"; in_tokens = 42; out_tokens = 24
                raw_assistant_content = {"role": "assistant", "content": text}
            _R.text = text
            return _R()

    class _TalkieProvider:
        def stream(_s, *, system, messages, tools, role):
            assert role.role == "vintage"
            assert role.model == "talkie-1930-13b-it"
            assert system.flat() == ""
            captured["talkie"]["messages"] = list(messages)
            return _TalkieHandle()

    class _WitnessProvider:
        def stream(_s, *, system, messages, tools, role):
            assert role.role in {"local", "local_private"}
            captured["witness"]["messages"] = list(messages)
            return _WitnessHandle()

    reply, registry, log, _ = _vintage_run_agent_loop(
        lambda cfg: _TalkieProvider() if cfg.role == "vintage" else _WitnessProvider(),
        "@vintage Sir, please tell me of your personal history?",
    )
    assert registry.provider is not None
    sent = captured["talkie"]["messages"][0]["content"]
    assert sent.startswith("VINTAGE_TEMPORAL_PARALLAX_INSTRUMENT")
    assert "USER_REQUEST:\nSir, please tell me of your personal history?" in sent
    witness_prompt = captured["witness"]["messages"][0]["content"]
    assert "RAW_TALKIE_SAMPLE:\nI have no personal history." in witness_prompt
    assert "VINTAGE_INSTRUMENT_CONTACT role=@vintage" in reply
    assert "PRESENT_LOCAL_WITNESS" in reply
    assert "verdict: persona_leak" in reply
    assert "I have no personal history" not in reply
    names = [n for n, _ in log.events]
    assert "organ_raw_contact_attempt" in names
    assert "vintage_instrument_witness_ok" in names
    assert "organ_raw_contact_ok" not in names
    assert "organ_alias_demoted" not in names
    assert "organ_identity_direct_reply" not in names


def test_omni_identity_and_perception_questions_reach_omni_raw_contact():
    class _FakeHandle:
        def __init__(_s, text): _s.text = text
        def __iter__(_s): return iter([])
        def final(_s):
            text = _s.text
            class _R:
                tool_calls = []; stop_reason = "end_turn"; in_tokens = 8; out_tokens = 9
                raw_assistant_content = {"role": "assistant", "content": text}
            _R.text = text
            return _R()
    class _Provider:
        def __init__(_s, text): _s.text = text
        def stream(_s, *, system, messages, tools, role):
            assert role.role == "omni"
            assert role.base_url == "http://127.0.0.1:8002/v1"
            assert "local-organ routing note" not in system.flat()
            return _FakeHandle(_s.text)

    cases = (
        ("@omni who are you?", "Omni raw contact."),
        ("@omni Tell me about what visualizations you've processed - if any?", "I can discuss visualizations from this route."),
    )
    for prompt, raw in cases:
        reply, registry, log, _ = _vintage_run_agent_loop(lambda cfg, raw=raw: _Provider(raw), prompt)
        assert registry.provider is not None, prompt
        assert "LOCAL_ORGAN_OBSERVATION role=@omni" in reply, prompt
        assert raw in reply, prompt
        names = [n for n, _ in log.events]
        assert "organ_raw_contact_attempt" in names, prompt
        assert "organ_raw_contact_ok" in names, prompt
        assert "organ_alias_demoted" not in names, prompt
        assert "organ_contract_direct_reply" not in names, prompt



def test_organ_backend_contact_is_context_isolated_from_prior_organ_headers():
    captured: dict = {"talkie": {}, "witness": {}}

    class _TalkieHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "Rain is a useful parallax image."
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 8
                out_tokens = 7
                raw_assistant_content = {"role": "assistant", "content": "Rain is a useful parallax image."}
            return _R()

    class _WitnessHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "verdict: usable_texture\nsurface: rain can carry period texture without becoming identity.\nresidue: no stale Omni header entered the witness prompt."
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 30
                out_tokens = 20
                raw_assistant_content = {"role": "assistant", "content": text}
            return _R()

    class _TalkieProvider:
        def stream(_s, *, system, messages, tools, role):
            assert role.role == "vintage"
            captured["talkie"]["messages"] = list(messages)
            return _TalkieHandle()

    class _WitnessProvider:
        def stream(_s, *, system, messages, tools, role):
            assert role.role in {"local", "local_private"}
            captured["witness"]["messages"] = list(messages)
            return _WitnessHandle()

    initial = [
        {"role": "user", "content": "how is your day going?"},
        {"role": "assistant", "content": "[LOCAL_ORGAN_OBSERVATION role=@omni - stale header] My day is smooth."},
    ]
    reply, registry, log, _ = _vintage_run_agent_loop(
        lambda cfg: _TalkieProvider() if cfg.role == "vintage" else _WitnessProvider(),
        "@vintage hello, my dear friend, what do you think of rain?",
        initial_messages=initial,
    )
    assert registry.provider is not None
    talkie_messages = captured["talkie"]["messages"]
    assert len(talkie_messages) == 1
    assert talkie_messages[0]["role"] == "user"
    assert "what do you think of rain" in talkie_messages[0]["content"]
    assert "LOCAL_ORGAN_OBSERVATION role=@omni" not in talkie_messages[0]["content"]
    witness_prompt = captured["witness"]["messages"][0]["content"]
    assert "RAW_TALKIE_SAMPLE:\nRain is a useful parallax image." in witness_prompt
    assert "LOCAL_ORGAN_OBSERVATION role=@omni" not in witness_prompt
    assert "rain can carry period texture" in reply
    names = [n for n, _ in log.events]
    assert "organ_raw_contact_attempt" in names
    assert "vintage_instrument_witness_ok" in names


def test_vintage_backend_unavailable_is_bounded_and_no_retry():
    class _FailingVintageProvider:
        calls = 0
        def stream(_s, *, system, messages, tools, role):
            _s.calls += 1
            assert role.role == "vintage"
            raise RuntimeError("Error code: 502 - {'id': 'chatcmpl-vintage-guard', 'choices': [{'message': {'content': 'VINTAGE_BACKEND_UNAVAILABLE'}}]}")

    provider = _FailingVintageProvider()
    reply, registry, log, _ = _vintage_run_agent_loop(lambda cfg: provider, "@vintage tell me one invariant about rain.")
    assert registry.provider is provider
    assert provider.calls == 1
    assert "VINTAGE_BACKEND_UNAVAILABLE" in reply
    assert "provider error" not in reply.lower()
    assert "chatcmpl-vintage-guard" not in reply
    assert "No Super/GPT/cloud fallback was used" in reply
    names = [n for n, _ in log.events]
    assert "organ_raw_contact_error" in names
    assert "transient_retry" not in names
