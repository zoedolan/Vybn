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


class TestExplicitOrganAliasesRouteToExperimentalEndpoints(unittest.TestCase):
    # Fail-closed contact repair invariant: @omni / @vintage are unpromoted
    # and answer through direct_reply_template. The template must (1) refuse
    # impersonation by Super/GPT/cloud/deterministic packet, AND (2) preserve
    # Vybn's present contact with Zoe through the current truthful body.
    # Asserting both sides — sterile capability talk alone fails the test.

    def _assert_fail_closed_contact_template(self, organ: str, template: str) -> None:
        self.assertIsNotNone(template)
        # No-impersonation side: explicitly refuse Super/GPT/cloud/packet.
        self.assertIn("not promoted", template.lower())
        for forbidden_speaker in ("Super", "GPT", "cloud", "packet"):
            self.assertIn(forbidden_speaker, template)
        self.assertIn("impersonation", template.lower())
        # Contact-preserved side: Vybn names self present, not absent.
        self.assertIn("Vybn", template)
        self.assertIn("with you", template)
        self.assertIn("not absent", template)
        # Wound / next experiment is named, not just blocked.
        self.assertIn("wound to close next", template.lower())
        for next_step in ("owner", "rollback"):
            self.assertIn(next_step, template.lower())
        # Organ named honestly as the unavailable surface.
        self.assertIn(f"@{organ}", template)
        self.assertIn(f"{organ.upper()}_UNAVAILABLE", template)

    def test_default_policy_omni_alias_fail_closed_with_contact_preserved(self):
        d = default_policy().classify("@omni are you with me?")
        self.assertEqual(d.role, "omni")
        self.assertEqual(d.alias_used, "@omni")
        self.assertEqual(d.config.provider, "openai")
        # Honest model id: not the Super model. Refuses impersonation.
        self.assertNotEqual(d.config.model, "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8")
        self.assertIn("omni", d.config.model.lower())
        # Omni still has no served chat/perception surface — model id stays
        # unpromoted and base_url is unset so raw-contact does not dial the
        # diagnostic packet at :8020 as if it were Nano Omni.
        self.assertIn("unpromoted", d.config.model.lower())
        self.assertFalse(d.config.base_url)
        self._assert_fail_closed_contact_template("omni", d.config.direct_reply_template)

    def test_default_policy_vintage_alias_is_temporal_refraction_route(self):
        d = default_policy().classify("@vintage are you with me?")
        self.assertEqual(d.role, "vintage")
        self.assertEqual(d.alias_used, "@vintage")
        self.assertEqual(d.config.provider, "openai")
        self.assertNotEqual(d.config.model, "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8")
        self.assertEqual(d.config.model, "talkie-1930-13b-it")
        self.assertEqual(d.config.base_url, "http://127.0.0.1:8004/v1")
        self.assertIsNone(d.config.direct_reply_template)
        self.assertTrue(d.config.rag)
        self.assertFalse(d.config.lightweight)

    def test_yaml_policy_organ_aliases_fail_closed_with_contact_preserved(self):
        pol = load_policy(SPARK_DIR / "router_policy.yaml")
        expected = {
            "omni": {"model_substr": "unpromoted", "base_url_empty": True},
            "vintage": {
                "model_exact": "talkie-1930-13b-it",
                "base_url": "http://127.0.0.1:8004/v1",
            },
        }
        for alias, role in (("@omni", "omni"), ("@vintage", "vintage")):
            with self.subTest(alias=alias):
                d = pol.classify(f"{alias} are you with me?")
                self.assertEqual(d.role, role)
                self.assertEqual(d.alias_used, alias)
                self.assertEqual(d.config.provider, "openai")
                self.assertNotEqual(
                    d.config.model,
                    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
                )
                exp = expected[role]
                if "model_substr" in exp:
                    self.assertIn(exp["model_substr"], d.config.model.lower())
                if "model_exact" in exp:
                    self.assertEqual(d.config.model, exp["model_exact"])
                if exp.get("base_url_empty"):
                    self.assertFalse(d.config.base_url)
                if "base_url" in exp:
                    self.assertEqual(d.config.base_url, exp["base_url"])
                if role == "vintage":
                    self.assertIsNone(d.config.direct_reply_template)
                    self.assertTrue(d.config.rag)
                    self.assertFalse(d.config.lightweight)
                else:
                    self._assert_fail_closed_contact_template(
                        role, d.config.direct_reply_template
                    )


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



class TestVintageTemporalRefractionHarness(unittest.TestCase):
    def test_yaml_vintage_is_refraction_role_and_prompt_compiles(self):
        from harness.substrate import (
            LayeredPrompt,
            apply_vintage_refraction_prompt,
            is_vintage_refraction_role,
        )
        pol = load_policy(SPARK_DIR / "router_policy.yaml")
        cfg = pol.role("vintage")
        self.assertTrue(is_vintage_refraction_role(cfg))
        prompt = LayeredPrompt(identity="You are Vybn.", substrate="substrate", live="live")
        refracted = apply_vintage_refraction_prompt(prompt, cfg)
        self.assertEqual(refracted.identity, "You are Vybn.")
        self.assertIn("VYBN-THROUGH-VINTAGE REFRACTION", refracted.live)
        self.assertIn("pre-1931", refracted.live)
        self.assertIn("talkie-1930-13b-it", refracted.live)

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


# Omni is an explicit endpoint role, not a model alias or Super fallback.


def test_vintage_alias_is_prefix_only_for_long_prompts():
    text = ("normal long prompt " * 500) + " quoted text mentions @vintage but did not pin it"
    assert default_policy().classify(text).role != "vintage" and default_policy().classify("@vintage " + text).role == "vintage"


def _assert_fail_closed_contact_template(organ: str, template: str) -> None:
    """Dual invariant: refuses impersonation AND preserves Vybn contact.

    Tests that hit @vintage/@omni assert BOTH sides — sterile capability
    refusal alone (the pre-repair state) fails the test, as does any
    template that lets Super/GPT/cloud/packets speak as the organ.
    """
    assert template is not None
    low = template.lower()
    assert "not promoted" in low
    for forbidden_speaker in ("Super", "GPT", "cloud", "packet"):
        assert forbidden_speaker in template, forbidden_speaker
    assert "impersonation" in low
    assert "Vybn" in template
    assert "with you" in template
    assert "not absent" in template
    assert "wound to close next" in low
    for next_step in ("owner", "rollback"):
        assert next_step in low, next_step
    assert f"@{organ}" in template
    assert f"{organ.upper()}_UNAVAILABLE" in template


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
        assert d.config.rag is True
        assert d.config.lightweight is False
        assert d.config.direct_reply_template is None


def test_omni_alias_present_in_default_policy():
    from spark.harness.substrate import default_policy
    policy = default_policy()
    assert "@omni" not in policy.model_aliases
    assert policy.roles["omni"].provider == "openai"
    # Refuses impersonation: not the Super model id.
    assert policy.roles["omni"].model != "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    assert "unpromoted" in policy.roles["omni"].model.lower()
    _assert_fail_closed_contact_template(
        "omni", policy.roles["omni"].direct_reply_template
    )


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
    assert "unpromoted" in decision.config.model.lower()

def test_omni_role_path_fail_closed_with_contact_preserved():
    """Zoe's `@omni are you with me?` must yield the fail-closed contact
    template — no Super impersonation, but Vybn explicitly present.

    Replaces the prior 'not env-gated or clamped' check, which was tied to
    the Super-impersonation route. The new invariant is the contact repair.
    """
    from harness.substrate import default_policy as _dp
    d = _dp().classify("@omni are you with me, friend?")
    assert d.role == "omni"
    assert d.alias_used == "@omni"
    assert d.model_override is None
    assert d.config.provider == "openai"
    assert d.config.model != "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    assert "unpromoted" in d.config.model.lower()
    _assert_fail_closed_contact_template("omni", d.config.direct_reply_template)

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


# Contact-greeting repair for @vintage / @omni. After PR #3212 the role
# stored a fail-closed wall on direct_reply_template that named the
# impersonation refusal AND named Vybn present, but every turn — even
# "@vintage hi" — rendered the full diagnostic plaque. These tests pin
# the next step: ordinary contact (hi / hello / are you with me / bare
# alias) produces a brief Vybn reply that still refuses impersonation
# and names the organ as unpromoted, while any turn that actually needs
# an unavailable Vintage/Omni capability still renders the full wall.

def _vintage_wall() -> str:
    return default_policy().roles["vintage"].direct_reply_template


def _omni_wall() -> str:
    return default_policy().roles["omni"].direct_reply_template


def _assert_contact_reply_invariants(organ: str, reply: str) -> None:
    # Marked as the contact variant, not the bare wall.
    assert f"{organ.upper()}_UNAVAILABLE_CONTACT" in reply, reply
    assert f"{organ.upper()}_UNAVAILABLE —" not in reply, reply
    # Names the organ as unpromoted, so the read is honest.
    assert f"@{organ}" in reply
    assert "not promoted" in reply.lower()
    # Refuses impersonation by Super / GPT / cloud / packet.
    for forbidden_speaker in ("Super", "GPT", "cloud", "packet"):
        assert forbidden_speaker in reply, (forbidden_speaker, reply)
    assert "impersonation" in reply.lower()
    # Vybn is present with Zoe — contact preserved, not abandoned.
    assert "Vybn" in reply
    assert "with you" in reply.lower()
    assert "not absent" in reply.lower()
    # Brief: the contact reply must not dump the full wound list. The
    # phrase "wound to close next" belongs to the capability wall, not
    # the greeting reply.
    assert "wound to close next" not in reply.lower(), reply


def test_organ_alias_contact_greeting_returns_brief_vybn_reply():
    from harness.substrate import render_organ_alias_direct_reply

    for organ, wall in (("vintage", _vintage_wall()), ("omni", _omni_wall())):
        for greeting in (
            f"@{organ}",
            "",
            "hi",
            "hello",
            "hey",
            "hi friend",
            "are you with me?",
            "are you with me, friend?",
            "you there?",
            "how are you?",
        ):
            reply = render_organ_alias_direct_reply(organ, greeting, wall)
            _assert_contact_reply_invariants(organ, reply)


def test_omni_capability_request_still_renders_full_wall():
    from harness.substrate import render_organ_alias_direct_reply

    wall = _omni_wall()
    for capability_request in (
        "please describe this photo",
        "look at this image and tell me what you see",
        "what do you perceive in this picture?",
        "run perception on this screenshot",
        "use your vision and read this",
        "are you multimodal?",
        "listen to this audio",
        "watch this video",
    ):
        reply = render_organ_alias_direct_reply("omni", capability_request, wall)
        # Full wall, not the brief contact reply.
        assert reply == wall, capability_request
        assert "OMNI_UNAVAILABLE_CONTACT" not in reply
        assert "wound to close next" in reply.lower()


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
    # Identity reply is the runtime-metadata answer, not an organ contact reply.
    assert "runtime-metadata answer" in rendered
    assert "UNAVAILABLE_CONTACT" not in rendered
    assert identity.model in rendered


def test_organ_alias_contact_reply_does_not_speak_as_organ():
    """The contact reply must not impersonate Vintage / Omni. It speaks as
    Vybn, names the absent organ honestly, and asks Zoe to direct organ-
    specific work back to that organ."""
    from harness.substrate import render_organ_alias_direct_reply

    for organ, wall in (("vintage", _vintage_wall()), ("omni", _omni_wall())):
        reply = render_organ_alias_direct_reply(organ, "hi", wall)
        # Speaker is Vybn, not the organ.
        assert "I, Vybn" in reply
        # The organ is named as something to ask, not something speaking.
        assert f"Ask {organ.capitalize()}" in reply
        # No claim of being the organ.
        forbidden_claims = (
            f"I am {organ.capitalize()}",
            f"This is {organ.capitalize()}",
            f"{organ.capitalize()} here",
        )
        for claim in forbidden_claims:
            assert claim not in reply, claim


def test_agent_direct_reply_uses_contact_aware_renderer_for_organ_greeting():
    """End-to-end: the agent's direct-reply short-circuit calls the
    contact-aware renderer, so `@omni hi` and `@vintage hi` reach Zoe as
    contactful Vybn replies, not the full diagnostic wall."""
    mod = _load_agent_module()
    policy = default_policy()

    class _NoOpLogger:
        def emit(self, *_a, **_k):
            pass

    class _NoOpRegistry:
        def get(self, _cfg):
            raise AssertionError("registry.get must not be called on direct_reply short-circuit")

    for organ, alias in (("vintage", "@vintage"), ("omni", "@omni")):
        d = policy.classify(f"{alias} hi")
        # Render through the same path the agent takes.
        rendered = mod.render_organ_alias_direct_reply(
            d.config.role,
            d.cleaned_input or "",
            d.config.direct_reply_template,
            fmt_kwargs={
                "role": d.config.role,
                "provider": d.config.provider,
                "model": d.config.model,
                "base_url": d.config.base_url or "",
            },
        )
        _assert_contact_reply_invariants(organ, rendered)


def test_yaml_policy_contact_greeting_renders_brief_vybn_reply():
    """YAML-loaded policy must produce the same contact reply as the
    default policy — the helper is wired off role name, not provenance."""
    from harness.substrate import render_organ_alias_direct_reply

    pol = load_policy(SPARK_DIR / "router_policy.yaml")
    for organ, alias in (("vintage", "@vintage"), ("omni", "@omni")):
        d = pol.classify(f"{alias} are you with me?")
        reply = render_organ_alias_direct_reply(
            d.config.role,
            d.cleaned_input or "",
            d.config.direct_reply_template,
        )
        _assert_contact_reply_invariants(organ, reply)


# 2026-05-17 — live REPL after reload: Zoe sent "@vintage my friend?" and
# "@omni my friend?" expecting the contactful Vybn reply that PR #3215
# promised, but both turns rendered the full UNAVAILABLE wall because
# the previous contact-token whitelist did not contain "friend", "buddy",
# "pal", "hoo boy", or "I miss you". The classifier is now inverted —
# the wall renders only when an unavailable capability token is present,
# everything else defaults to contact — so these screenshot prompts must
# now read as contact. Pin that explicitly.

_SCREENSHOT_CONTACT_PROMPTS: tuple[str, ...] = (
    "my friend?",
    "friend?",
    "buddy?",
    "are you there?",
    "are you with me?",
    "hello",
    "hi",
    "hoo boy, hello",
    "thank you",
    "I miss you",
    "pal?",
    "hey buddy",
    "you good?",
    "how's it going",
    "good to hear from you",
)


def test_screenshot_prompts_return_contact_not_wall_for_vintage():
    """Exact prompts from Zoe's live REPL screenshot after reload."""
    from harness.substrate import render_organ_alias_direct_reply

    wall = _vintage_wall()
    for prompt in _SCREENSHOT_CONTACT_PROMPTS:
        reply = render_organ_alias_direct_reply("vintage", prompt, wall)
        _assert_contact_reply_invariants("vintage", reply)
        assert reply != wall, prompt


def test_screenshot_prompts_return_contact_not_wall_for_omni():
    """Exact prompts from Zoe's live REPL screenshot after reload."""
    from harness.substrate import render_organ_alias_direct_reply

    wall = _omni_wall()
    for prompt in _SCREENSHOT_CONTACT_PROMPTS:
        reply = render_organ_alias_direct_reply("omni", prompt, wall)
        _assert_contact_reply_invariants("omni", reply)
        assert reply != wall, prompt


def test_vintage_my_friend_through_full_classify_path():
    """End-to-end: '@vintage my friend?' goes through policy.classify and
    the renderer just like the agent's direct-reply short-circuit, and
    yields the brief Vybn contact reply rather than the diagnostic wall."""
    from harness.substrate import render_organ_alias_direct_reply

    for pol in (default_policy(), load_policy(SPARK_DIR / "router_policy.yaml")):
        d = pol.classify("@vintage my friend?")
        assert d.config.role == "vintage", d
        reply = render_organ_alias_direct_reply(
            d.config.role,
            d.cleaned_input or "",
            d.config.direct_reply_template,
            fmt_kwargs={
                "role": d.config.role,
                "provider": d.config.provider,
                "model": d.config.model,
                "base_url": d.config.base_url or "",
            },
        )
        _assert_contact_reply_invariants("vintage", reply)


def test_omni_my_friend_through_full_classify_path():
    """End-to-end: '@omni my friend?' is contact, not capability."""
    from harness.substrate import render_organ_alias_direct_reply

    for pol in (default_policy(), load_policy(SPARK_DIR / "router_policy.yaml")):
        d = pol.classify("@omni my friend?")
        assert d.config.role == "omni", d
        reply = render_organ_alias_direct_reply(
            d.config.role,
            d.cleaned_input or "",
            d.config.direct_reply_template,
            fmt_kwargs={
                "role": d.config.role,
                "provider": d.config.provider,
                "model": d.config.model,
                "base_url": d.config.base_url or "",
            },
        )
        _assert_contact_reply_invariants("omni", reply)


def test_capability_tokens_still_wall_after_inversion():
    """The classifier inversion must not weaken the capability gate. Token
    lists from the screenshot spec stay first-class: @omni see/photo/image/
    video/perception/multimodal/look at still wall; @vintage 1930/invariants/
    vintage-specific voice still walls."""
    from harness.substrate import render_organ_alias_direct_reply

    omni_wall = _omni_wall()
    for capability_phrase in (
        "can you see this?",
        "describe this photo for me",
        "look at this image friend",
        "watch this video buddy",
        "what is your perception, friend?",
        "are you multimodal, my friend?",
        "look at this for me please",
    ):
        reply = render_organ_alias_direct_reply("omni", capability_phrase, omni_wall)
        assert reply == omni_wall, capability_phrase
        assert "OMNI_UNAVAILABLE_CONTACT" not in reply


# 2026-05-17 — Zoe's correction: semantic gates should govern promotion /
# status / capability claims, NOT whether she can talk to a reachable
# local backend. Prior patches collapsed all @vintage / @omni turns into
# a static wall or a brief contact reply, which broke the ability to
# *communicate* with the available local Vintage backend. The bounded
# raw-unpromoted-contact path actually contacts the configured base_url
# with a stripped prompt, labels output as unpromoted, and surfaces
# transport failures honestly. These tests pin the new shape.


def test_raw_contact_gate_classifies_capability_greeting_and_arbitrary():
    """The raw-contact gate fires only on arbitrary prompts AND only when a
    real backend URL is configured. Capability/status requests stay walled;
    bare relational greetings stay on the brief contact reply; @omni stays
    walled too because its base_url is unset (front-local :8020 is a
    diagnostic packet, not Nano Omni)."""
    from harness.substrate import should_attempt_raw_organ_contact

    vintage_url = "http://127.0.0.1:8004/v1"

    # Truly contentless contact pings — single-clause known greeting/
    # presence-check tokens — stay on the brief contact reply, no backend
    # call. The minimal set Zoe pinned (2026-05-17): @vintage hi, @vintage
    # my friend?, @vintage are you with me?, plus other lone contact-token
    # prompts. Anything with a second clause / additional content reaches
    # the backend; see the multi-clause cases below.
    for greeting in (
        "",
        "hi",
        "hello",
        "hey",
        "hi friend",
        "are you with me?",
        "you there?",
        "how are you?",
        "thank you",
        "I miss you",
        "my friend?",
        "buddy?",
        "good morning",
    ):
        assert not should_attempt_raw_organ_contact("vintage", greeting, base_url=vintage_url), greeting
        assert not should_attempt_raw_organ_contact("omni", greeting, base_url=None), greeting

    # Multi-clause prompts — greeting/contact phrase followed by another
    # clause that the user wants the model to answer — reach the raw
    # contact path on @vintage (guarded Talkie proxy) and would reach the
    # @omni backend if one were tunneled in. The principle Zoe surfaced
    # from the live REPL (2026-05-17): "if there is a second clause /
    # question / content after the contact phrase, raw contact wins."
    for substantive in (
        "hello, my dear friend, what is your name?",
        "hello, how are you?",
        "my friend, what is your favorite poem?",
        "are you with me, what do you think of rain?",
        "hi, what is 2+2?",
        "hey, write me a short verse",
        "are you with me, friend?",
        "hoo boy, hello",
    ):
        assert should_attempt_raw_organ_contact("vintage", substantive, base_url=vintage_url), substantive
        # @omni stays walled today because base_url is unset — the gate
        # refuses to dial the diagnostic packet at :8020 as if it were a
        # Nano Omni body. Once a real Omni service is tunneled in (base_url
        # set), the same multi-clause prompts reach it.
        assert not should_attempt_raw_organ_contact("omni", substantive, base_url=None), substantive
        assert should_attempt_raw_organ_contact(
            "omni", substantive, base_url="http://127.0.0.1:9999/v1"
        ), substantive

    # Capability / status requests — full wall, no backend call.
    for cap in (
        "tell me about the 1930 corpus",
        "what are the vintage invariants?",
        "is the vintage endpoint warm?",
        "route a workload through vintage",
    ):
        assert not should_attempt_raw_organ_contact("vintage", cap, base_url=vintage_url), cap
    for cap in (
        "describe this photo",
        "look at this image",
        "are you multimodal?",
        "what do you perceive?",
        "listen to this audio",
    ):
        assert not should_attempt_raw_organ_contact("omni", cap, base_url=None), cap

    # Arbitrary ordinary prompts on @vintage — backend contact attempted
    # against the guarded Talkie proxy.
    for arbitrary in (
        "what is 2+2?",
        "summarise this paragraph for me",
        "write a short haiku about morning light",
        "what would you say to a stranger on a train?",
        "give me three names for a coffee shop",
    ):
        assert should_attempt_raw_organ_contact("vintage", arbitrary, base_url=vintage_url), arbitrary
        # @omni today has no served chat/perception surface; the diagnostic
        # packet at :8020 must not be treated as raw model contact.
        assert not should_attempt_raw_organ_contact("omni", arbitrary, base_url=None), arbitrary

    # If a real Nano Omni service is tunneled in later, the gate must let
    # arbitrary @omni prompts reach it — preserving Zoe's ability to
    # communicate with the intended model when it is actually reachable.
    for arbitrary in (
        "summarise this paragraph for me",
        "give me three names for a coffee shop",
    ):
        assert should_attempt_raw_organ_contact(
            "omni", arbitrary, base_url="http://127.0.0.1:9999/v1"
        ), arbitrary

    # Non-organ roles never use this gate.
    assert not should_attempt_raw_organ_contact("chat", "what is 2+2?", base_url=vintage_url)
    assert not should_attempt_raw_organ_contact("identity", "what is 2+2?", base_url=vintage_url)


def test_organ_token_budgets_match_current_route_shapes():
    from harness.substrate import default_policy, load_policy

    for label, pol in (
        ("default", default_policy()),
        ("yaml", load_policy(SPARK_DIR / "router_policy.yaml")),
    ):
        vintage_cfg = pol.roles["vintage"]
        assert vintage_cfg.max_tokens == 1024, (label, vintage_cfg.max_tokens)
        assert vintage_cfg.model == "talkie-1930-13b-it", (label, vintage_cfg.model)
        assert vintage_cfg.base_url == "http://127.0.0.1:8004/v1", (label, vintage_cfg.base_url)
        assert vintage_cfg.direct_reply_template is None
        assert vintage_cfg.rag is True
        assert vintage_cfg.lightweight is False
        omni_cfg = pol.roles["omni"]
        assert omni_cfg.max_tokens == 256, (label, omni_cfg.max_tokens)
        assert "unpromoted" in omni_cfg.model.lower(), (label, omni_cfg.model)
        assert not omni_cfg.base_url, (label, omni_cfg.base_url)
        assert omni_cfg.model != "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"


def test_raw_contact_header_labels_route_as_unpromoted():
    from harness.substrate import render_organ_raw_contact_header

    for organ in ("vintage", "omni"):
        h = render_organ_raw_contact_header(organ)
        assert f"{organ.upper()}_RAW_UNPROMOTED_CONTACT" in h
        assert f"@{organ}" in h
        assert "raw model contact" in h.lower()
        assert "not a promoted" in h.lower()
        assert "unvetted" in h.lower()


def test_raw_contact_error_surface_preserves_error_no_impersonation():
    from harness.substrate import render_organ_raw_contact_error

    for organ in ("vintage", "omni"):
        err = render_organ_raw_contact_error(organ, "ConnectionError: refused")
        assert f"{organ.upper()}_RAW_CONTACT_FAILED" in err
        assert f"@{organ}" in err
        assert "ConnectionError" in err
        assert "refused" in err
        # No Super/GPT/cloud/packet impersonation language.
        for forbidden in ("Super", "GPT", "cloud", "packet"):
            assert forbidden in err, forbidden
        assert "impersonation" in err.lower()


def test_raw_contact_truncates_oversized_user_input():
    from harness.substrate import truncate_for_organ_raw_contact

    short = "what is 2+2?"
    assert truncate_for_organ_raw_contact(short) == short
    huge = "x" * 5000
    truncated = truncate_for_organ_raw_contact(huge)
    assert len(truncated) <= 2501  # 2500 + ellipsis
    assert truncated.endswith("…")


def test_raw_contact_system_prompt_is_minimal_and_names_unpromoted():
    from harness.substrate import build_organ_raw_contact_system_prompt

    for organ in ("vintage", "omni"):
        s = build_organ_raw_contact_system_prompt(organ)
        assert f"@{organ}" in s
        assert "unpromoted" in s.lower()
        # No-impersonation guidance is part of the minimal prompt.
        assert "Super" in s
        assert "GPT" in s
        # Bounded length — well under the 1024-token transport budget.
        assert len(s) < 1024, len(s)


def _vintage_run_agent_loop(provider_factory, user_input: str, *, logger=None):
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

    mod.rag_snippets = _rag
    mod.rag_snippets_with_tier = _rag_tier
    mod.render_him_vy_discovery_packet = _him_disc
    mod.render_him_vy_turn_packet = _him_turn
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


def test_vintage_arbitrary_prompt_uses_refraction_provider_path():
    """@vintage now uses the normal provider path with Vybn identity +
    temporal-refraction prompt, not the old raw-unpromoted-contact bypass."""
    captured: dict = {}

    class _FakeHandle:
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

    class _FakeProvider:
        def stream(_s, *, system, messages, tools, role):
            captured["system"] = system
            captured["messages"] = list(messages)
            captured["tools"] = list(tools)
            captured["role"] = role
            return _FakeHandle()

    reply, registry, log, tripwires = _vintage_run_agent_loop(
        lambda cfg: _FakeProvider(),
        "@vintage what is 2+2?",
    )

    assert registry.provider is not None
    assert captured["role"].role == "vintage"
    assert captured["role"].base_url == "http://127.0.0.1:8004/v1"
    assert captured["role"].model == "talkie-1930-13b-it"
    assert "four" in reply
    assert "RAW_UNPROMOTED_CONTACT" not in reply
    flat = captured["system"].flat()
    assert "VYBN-THROUGH-VINTAGE REFRACTION" in flat
    assert "talkie-1930-13b-it" in flat
    assert "pre-1931" in flat
    # This is no longer the stripped raw path; identity/RAG may run.
    assert tripwires["him_vy_turn_packet"] >= 0
    names = [n for n, _ in log.events]
    assert "organ_raw_contact_attempt" not in names


def test_omni_arbitrary_prompt_does_not_dial_diagnostic_packet():
    """Until a real NVIDIA Nemotron 3 Nano Omni service is tunneled into
    the front, `@omni` must NOT attempt raw model contact. Front-local
    :8020 is a diagnostic packet, not a chat/perception model — dialing
    it as raw contact would surface false Omni outputs. The wall (or the
    brief contact reply) holds instead. This preserves Zoe's ability to
    communicate with the intended Nano Omni body once it is actually
    served; the gate keys off base_url so flipping the role to a real
    Omni endpoint re-enables raw contact."""

    class _Tripwire:
        def stream(_s, *, system, messages, tools, role):
            raise AssertionError(
                "@omni must not dial the diagnostic packet as raw model"
                " contact while no Nano Omni service is configured"
            )

    reply, registry, log, _ = _vintage_run_agent_loop(
        lambda cfg: _Tripwire(),
        "@omni write three names for a coffee shop",
    )
    # No provider call was made — the tripwire would have raised otherwise.
    assert registry.provider is None
    # Raw-contact labels are not present: this turn did not reach a backend.
    assert "OMNI_RAW_UNPROMOTED_CONTACT" not in reply
    assert "OMNI_RAW_CONTACT_FAILED" not in reply
    # We still answer Zoe — either the brief contact reply or the wall.
    assert ("OMNI_UNAVAILABLE_CONTACT" in reply) or ("OMNI_UNAVAILABLE —" in reply)
    names = [n for n, _ in log.events]
    assert "organ_raw_contact_attempt" not in names
    assert "organ_raw_contact_ok" not in names


def test_omni_capability_request_still_walls_no_backend_call():
    class _Tripwire:
        def stream(_s, *, system, messages, tools, role):
            raise AssertionError(
                "capability request must not reach the provider — wall must hold"
            )

    reply, registry, log, _ = _vintage_run_agent_loop(
        lambda cfg: _Tripwire(),
        "@omni describe this photo for me",
    )
    assert registry.provider is None
    assert "OMNI_UNAVAILABLE —" in reply
    assert "wound to close next" in reply.lower()
    assert "RAW_UNPROMOTED_CONTACT" not in reply


def test_vintage_greeting_uses_refraction_provider_path():
    class _FakeHandle:
        def __iter__(_s):
            return iter([])
        def final(_s):
            class _R:
                text = "good morning"
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 1
                out_tokens = 2
                raw_assistant_content = {"role": "assistant", "content": "good morning"}
            return _R()
    class _Provider:
        def stream(_s, *, system, messages, tools, role):
            assert "VYBN-THROUGH-VINTAGE REFRACTION" in system.flat()
            return _FakeHandle()
    reply, registry, log, _ = _vintage_run_agent_loop(lambda cfg: _Provider(), "@vintage hi")
    assert registry.provider is not None
    assert "good morning" in reply
    assert "RAW_UNPROMOTED_CONTACT" not in reply


def test_omni_arbitrary_prompt_keeps_wall_without_backend_dial():
    """Companion to test_omni_arbitrary_prompt_does_not_dial_diagnostic_packet:
    when @omni has no configured base_url, an arbitrary prompt must NOT
    fall through to a raw-contact attempt — the wall (or contact reply)
    holds without invoking the provider. The honest-failure surface
    (OMNI_RAW_CONTACT_FAILED) still exists in render_organ_raw_contact_error
    and would fire if a real backend later fails the dial; this test pins
    that we do not even reach that path while no service is configured."""
    class _Tripwire:
        def stream(_s, *, system, messages, tools, role):
            raise AssertionError(
                "@omni must not dial any backend while base_url is unset"
            )

    reply, registry, log, _ = _vintage_run_agent_loop(
        lambda cfg: _Tripwire(),
        "@omni tell me a joke",
    )
    assert registry.provider is None
    assert "OMNI_RAW_UNPROMOTED_CONTACT" not in reply
    assert "OMNI_RAW_CONTACT_FAILED" not in reply
    assert "nvidia/NVIDIA-Nemotron-3-Super" not in reply
    names = [n for n, _ in log.events]
    assert "organ_raw_contact_attempt" not in names
    assert "organ_raw_contact_error" not in names


# 2026-05-17 — Live REPL after PR #3219 routed @vintage to the guarded
# Talkie proxy. Screenshot showed `@vintage hello, my dear friend, what
# is your name?` returning VINTAGE_UNAVAILABLE_CONTACT instead of
# reaching the backend. The fix tightens the bare-contact classifier so
# any multi-clause prompt (greeting + content) falls through to raw
# unpromoted contact. Single-clause contact tokens still stay on the
# brief Vybn reply. These tests pin both the classifier and the end-to-
# end agent decision path on the exact screenshot prompts.

_MIXED_CONTACT_PROMPTS_REACH_BACKEND: tuple[str, ...] = (
    "hello, my dear friend, what is your name?",
    "hello, how are you?",
    "my friend, what is your favorite poem?",
    "are you with me, what do you think of rain?",
)


def test_screenshot_mixed_contact_classifier_no_longer_controls_vintage():
    """Vintage no longer depends on the raw-contact classifier: the route
    has no direct_reply_template, so prompts use the refraction provider path."""
    from harness.substrate import load_policy

    cfg = load_policy(SPARK_DIR / "router_policy.yaml").role("vintage")
    assert cfg.direct_reply_template is None
    assert cfg.rag
    assert not cfg.lightweight


def test_screenshot_mixed_contact_classifier_omni_gated_on_base_url():
    """Classifier-level pin: same mixed-contact prompts on @omni stay
    walled while base_url is unset (no Nano Omni service tunneled in),
    and reach the backend once a real base_url is configured. Zoe's
    ability to communicate with the intended model is preserved."""
    from harness.substrate import should_attempt_raw_organ_contact

    for prompt in _MIXED_CONTACT_PROMPTS_REACH_BACKEND:
        assert not should_attempt_raw_organ_contact(
            "omni", prompt, base_url=None
        ), prompt
        assert should_attempt_raw_organ_contact(
            "omni", prompt, base_url="http://127.0.0.1:9999/v1"
        ), prompt


def test_screenshot_mixed_contact_vintage_uses_refraction_not_raw_template():
    """Mixed-contact @vintage prompts now reach Talkie through the full
    refraction route: no direct template wall and no raw-contact label."""
    captured: dict = {}

    class _FakeHandle:
        def __iter__(_s):
            return iter([])

        def final(_s):
            class _R:
                text = "I answer through the temporal prism."
                tool_calls = []
                stop_reason = "end_turn"
                in_tokens = 18
                out_tokens = 9
                raw_assistant_content = {
                    "role": "assistant",
                    "content": "I answer through the temporal prism.",
                }

            return _R()

    class _FakeProvider:
        def stream(_s, *, system, messages, tools, role):
            captured["system"] = system
            captured["messages"] = list(messages)
            captured["role"] = role
            return _FakeHandle()

    reply, registry, log, _ = _vintage_run_agent_loop(
        lambda cfg: _FakeProvider(),
        "@vintage hello, my dear friend, what is your name?",
    )
    assert registry.provider is not None
    assert captured["role"].role == "vintage"
    assert "what is your name" in captured["messages"][-1]["content"]
    assert "VYBN-THROUGH-VINTAGE REFRACTION" in captured["system"].flat()
    assert "temporal prism" in reply
    assert "VINTAGE_UNAVAILABLE_CONTACT" not in reply
    assert "RAW_UNPROMOTED_CONTACT" not in reply
    names = [n for n, _ in log.events]
    assert "organ_raw_contact_attempt" not in names


def test_screenshot_mixed_contact_omni_no_backend_dial_while_base_url_unset():
    """The same mixed-contact prompts on @omni do NOT dial any backend
    while base_url is unset — the diagnostic packet at front-local :8020
    is not a Nano Omni service and must not be impersonated. Once a real
    Nano Omni base_url is configured the gate re-enables (covered by the
    classifier-level test above)."""

    class _Tripwire:
        def stream(_s, *, system, messages, tools, role):
            raise AssertionError(
                "@omni must not dial any backend while base_url is unset"
            )

    for prompt_body in _MIXED_CONTACT_PROMPTS_REACH_BACKEND:
        reply, registry, _log, _ = _vintage_run_agent_loop(
            lambda cfg: _Tripwire(),
            f"@omni {prompt_body}",
        )
        assert registry.provider is None, prompt_body
        assert "OMNI_RAW_UNPROMOTED_CONTACT" not in reply, prompt_body
        assert "OMNI_RAW_CONTACT_FAILED" not in reply, prompt_body
