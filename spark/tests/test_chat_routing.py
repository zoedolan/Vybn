"""Third-pass tests: role-based dispatch in vybn_chat_api.

Covers:
  - ChatRequest accepts `role` / `route` fields without breaking clients.
  - _resolve_role honours explicit role, directive prefix, and the
    classify flag; unknown roles are rejected.
  - _strip_directive only removes the leading directive token.
  - VYBN_CHAT_ALLOWED_ROLES gates chat-side dispatch; anthropic roles
    are never dispatched from chat-api even when requested.
  - /v1/route introspection returns the expected shape.
  - Legacy clients (no role, no route) keep byte-identical behavior.
  - A routed /local dispatch uses OpenAIProvider + LayeredPrompt and
    returns an OpenAI-shaped completion with `vybn_route` metadata.

Run: python3 spark/tests/test_chat_routing.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

THIS = Path(__file__).resolve()
SPARK_DIR = THIS.parent.parent
sys.path.insert(0, str(SPARK_DIR))

from harness.policy import RoleConfig, default_policy  # noqa: E402


def _load_chat_api(env_overrides: dict | None = None):
    """Load vybn_chat_api.py as a fresh module. Applies env overrides
    before exec so module-level constants (VYBN_CHAT_ROUTING,
    VYBN_CHAT_ALLOWED_ROLES, etc.) pick them up."""
    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401
    except Exception:
        return None
    path = SPARK_DIR / "vybn_chat_api.py"

    saved: dict[str, str | None] = {}
    if env_overrides:
        for k, v in env_overrides.items():
            saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # Use the real module name so Pydantic can resolve forward refs
    # like `list[Message]` via sys.modules. Register before exec.
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


class TestChatRequestShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_chat_api()

    def setUp(self):
        if self.mod is None:
            self.skipTest("fastapi/httpx not importable in this env")

    def test_chat_request_accepts_role_and_route(self):
        req = self.mod.ChatRequest(
            messages=[self.mod.Message(role="user", content="hi")],
            role="local",
            route="auto",
        )
        self.assertEqual(req.role, "local")
        self.assertEqual(req.route, "auto")

    def test_chat_request_legacy_clients_unchanged(self):
        # No role, no route — must still validate.
        req = self.mod.ChatRequest(
            messages=[self.mod.Message(role="user", content="hi")],
        )
        self.assertIsNone(req.role)
        self.assertIsNone(req.route)
        self.assertTrue(req.rag)  # legacy default preserved

    def test_routing_ok_flag(self):
        self.assertTrue(self.mod._ROUTING_OK)


class TestResolveRole(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_chat_api()

    def setUp(self):
        if self.mod is None:
            self.skipTest("fastapi/httpx not importable in this env")

    def test_explicit_role(self):
        name, cfg, reason = self.mod._resolve_role(
            explicit_role="local", last_user_text="whatever", classify=False,
        )
        self.assertEqual(name, "local")
        self.assertEqual(cfg.provider, "openai")
        self.assertTrue(cfg.base_url)
        self.assertEqual(reason, "explicit=local")

    def test_unknown_role_rejected(self):
        name, cfg, reason = self.mod._resolve_role(
            explicit_role="does_not_exist", last_user_text="", classify=False,
        )
        self.assertIsNone(name)
        self.assertIsNone(cfg)
        self.assertIn("unknown_role", reason)

    def test_directive_prefix_detected(self):
        name, cfg, reason = self.mod._resolve_role(
            explicit_role=None,
            last_user_text="/local what's the weather",
            classify=False,
        )
        self.assertEqual(name, "local")
        self.assertEqual(reason, "directive=/local")

    def test_classify_disabled_means_no_role(self):
        # Input that heuristics would classify as "code" — but classify
        # is off, so we get no role and the caller uses the legacy path.
        name, _, reason = self.mod._resolve_role(
            explicit_role=None,
            last_user_text="fix this python traceback",
            classify=False,
        )
        self.assertIsNone(name)
        self.assertEqual(reason, "classify_disabled")

    def test_classify_enabled_returns_heuristic_role(self):
        name, cfg, reason = self.mod._resolve_role(
            explicit_role=None,
            last_user_text="fix this python traceback",
            classify=True,
        )
        self.assertEqual(name, "code")
        self.assertTrue(reason.startswith("heuristic"))

    def test_empty_input_returns_no_role(self):
        name, _, reason = self.mod._resolve_role(
            explicit_role=None, last_user_text="   ", classify=True,
        )
        self.assertIsNone(name)
        self.assertEqual(reason, "empty_input")


class TestStripDirective(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_chat_api()

    def setUp(self):
        if self.mod is None:
            self.skipTest("fastapi/httpx not importable in this env")

    def test_strip_leading_directive(self):
        directives = {"/local": "local", "/chat": "chat"}
        out = self.mod._strip_directive("/local hello there", directives)
        self.assertEqual(out, "hello there")

    def test_leaves_content_without_directive_alone(self):
        directives = {"/local": "local"}
        out = self.mod._strip_directive("just hi", directives)
        self.assertEqual(out, "just hi")

    def test_handles_directive_only(self):
        directives = {"/local": "local"}
        out = self.mod._strip_directive("/local", directives)
        self.assertEqual(out, "")

    def test_does_not_strip_mid_sentence(self):
        directives = {"/local": "local"}
        out = self.mod._strip_directive("ask /local later", directives)
        self.assertEqual(out, "ask /local later")


class TestAllowedRolesGate(unittest.TestCase):
    """Anthropic roles must never be dispatched by the chat API even
    if the caller explicitly requests them — the server falls back to
    the legacy vLLM path instead of silently making a cloud call."""

    def test_default_allowed_roles_is_local_only(self):
        mod = _load_chat_api(env_overrides={"VYBN_CHAT_ALLOWED_ROLES": None})
        if mod is None:
            self.skipTest("fastapi/httpx not importable")
        self.assertEqual(mod.VYBN_CHAT_ALLOWED_ROLES, {"local"})

    def test_allowed_roles_env_override(self):
        mod = _load_chat_api(env_overrides={
            "VYBN_CHAT_ALLOWED_ROLES": "local,orchestrate",
        })
        if mod is None:
            self.skipTest("fastapi/httpx not importable")
        self.assertEqual(mod.VYBN_CHAT_ALLOWED_ROLES, {"local", "orchestrate"})

    def test_classify_off_by_default(self):
        mod = _load_chat_api(env_overrides={"VYBN_CHAT_ROUTING": None})
        if mod is None:
            self.skipTest("fastapi/httpx not importable")
        self.assertFalse(mod.VYBN_CHAT_ROUTING)


class TestRouteIntrospection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_chat_api()

    def setUp(self):
        if self.mod is None:
            self.skipTest("fastapi/httpx not importable in this env")
        try:
            from fastapi.testclient import TestClient
        except Exception:
            self.skipTest("fastapi.testclient unavailable")
        self.TestClient = TestClient

    def test_route_endpoint_explicit_local(self):
        client = self.TestClient(self.mod.app)
        r = client.post("/v1/route", json={
            "messages": [{"role": "user", "content": "test"}],
            "role": "local",
        })
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["role"], "local")
        self.assertEqual(body["provider"], "openai")
        self.assertTrue(body["dispatchable"])
        self.assertIn("local", body["allowed_roles"])

    def test_route_endpoint_rejects_anthropic_role_for_dispatch(self):
        """Explicit role=chat resolves, but chat-api must mark it
        non-dispatchable because anthropic providers aren't allowed."""
        client = self.TestClient(self.mod.app)
        r = client.post("/v1/route", json={
            "messages": [{"role": "user", "content": "hi"}],
            "role": "chat",
        })
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["role"], "chat")
        self.assertEqual(body["provider"], "anthropic")
        self.assertFalse(body["dispatchable"])

    def test_route_endpoint_default_no_role(self):
        """With no role, no route=auto, and classify disabled, the
        response indicates we'd fall through to the legacy path."""
        client = self.TestClient(self.mod.app)
        r = client.post("/v1/route", json={
            "messages": [{"role": "user", "content": "fix this python traceback"}],
        })
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIsNone(body["role"])
        self.assertFalse(body["dispatchable"])
        self.assertEqual(body["reason"], "classify_disabled")

    def test_route_endpoint_classify_auto(self):
        client = self.TestClient(self.mod.app)
        r = client.post("/v1/route", json={
            "messages": [{"role": "user", "content": "/local quick check"}],
            "route": "auto",
        })
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["role"], "local")
        self.assertEqual(body["reason"], "directive=/local")


class TestRoutedDispatch(unittest.TestCase):
    """End-to-end routed chat through a stub OpenAI client."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_chat_api()

    def setUp(self):
        if self.mod is None:
            self.skipTest("fastapi/httpx not importable in this env")
        try:
            from fastapi.testclient import TestClient
        except Exception:
            self.skipTest("fastapi.testclient unavailable")
        self.TestClient = TestClient
        self._saved_openai = sys.modules.get("openai")

        # Install a fake openai module so OpenAIProvider uses it.
        module = types.ModuleType("openai")
        captured = {"kwargs": None}

        class _Completions:
            def create(self_inner, **kwargs):
                captured["kwargs"] = kwargs

                class _Resp:
                    def model_dump(_self):
                        return {
                            "choices": [{
                                "message": {
                                    "content": "local says hi",
                                    "tool_calls": [],
                                },
                                "finish_reason": "stop",
                            }],
                            "usage": {
                                "prompt_tokens": 7,
                                "completion_tokens": 3,
                            },
                        }
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            def __init__(self, api_key=None, base_url=None, timeout=None):
                self.api_key = api_key
                self.base_url = base_url
                self.timeout = timeout
                self.chat = _Chat()

        module.OpenAI = _Client
        sys.modules["openai"] = module
        self.captured = captured

    def tearDown(self):
        if self._saved_openai is not None:
            sys.modules["openai"] = self._saved_openai
        else:
            sys.modules.pop("openai", None)

    def test_routed_local_completion(self):
        # Force a fresh provider registry so the fake openai module is
        # picked up (the registry caches provider instances, but the
        # OpenAI client is constructed per-call via `from openai import OpenAI`).
        self.mod._PROVIDER_REGISTRY = None
        self.mod._POLICY = None
        self.mod._ROUTER = None
        client = self.TestClient(self.mod.app)
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "role": "local",
            "stream": False,
        })
        self.assertEqual(r.status_code, 200, msg=r.text)
        body = r.json()
        self.assertEqual(body["choices"][0]["message"]["content"], "local says hi")
        self.assertEqual(body["vybn_route"]["role"], "local")
        self.assertEqual(body["vybn_route"]["provider"], "openai")
        self.assertEqual(body["usage"]["prompt_tokens"], 7)
        self.assertEqual(body["usage"]["completion_tokens"], 3)

        # The provider received a messages list with a system message
        # built from LayeredPrompt.flat().
        sent = self.captured["kwargs"]
        self.assertIsNotNone(sent)
        self.assertEqual(sent["messages"][0]["role"], "system")
        # Role policy for `local` has rag=True but the stubbed fake
        # doesn't populate deep_memory, so live text is empty and the
        # flattened system prompt equals SYSTEM_PROMPT.
        self.assertTrue(sent["messages"][0]["content"])

    def test_directive_prefix_stripped_from_user_turn(self):
        self.mod._PROVIDER_REGISTRY = None
        self.mod._POLICY = None
        self.mod._ROUTER = None
        client = self.TestClient(self.mod.app)
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "/local what's up"}],
            "route": "auto",
            "stream": False,
        })
        self.assertEqual(r.status_code, 200, msg=r.text)
        body = r.json()
        self.assertEqual(body["vybn_route"]["role"], "local")

        sent = self.captured["kwargs"]
        # The last user message should be just "what's up"
        user_msgs = [m for m in sent["messages"] if m["role"] == "user"]
        self.assertEqual(user_msgs[-1]["content"], "what's up")

    def test_explicit_anthropic_role_falls_back_to_legacy(self):
        """Asking for role=chat from the chat-api must NOT invoke
        Anthropic (no paid call from the chat surface). The request
        falls through to the legacy vLLM path; since there's no vLLM
        running in the test, we expect a connection-type failure, not
        a silent cloud call."""
        self.mod._PROVIDER_REGISTRY = None
        self.mod._POLICY = None
        self.mod._ROUTER = None
        client = self.TestClient(self.mod.app)
        # Capture whether our fake openai client saw a call.
        self.captured["kwargs"] = None

        # Stub _http_client so the legacy path fails fast without a
        # real network round-trip. It doesn't matter what error it
        # raises — the point is that we got past route resolution
        # without dispatching through the fake openai.
        class _FakeHttp:
            async def post(self, *a, **kw):
                raise RuntimeError("legacy-path-reached")

            def build_request(self, *a, **kw):
                raise RuntimeError("legacy-path-reached")

        saved = self.mod._http_client
        self.mod._http_client = _FakeHttp()
        try:
            # TestClient re-raises exceptions from the endpoint by
            # default; the sentinel "legacy-path-reached" is what we
            # want to see — proof that we went through the legacy code
            # path instead of the anthropic route. Any raised
            # exception carrying that sentinel satisfies the check.
            reached = False
            try:
                client.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "role": "chat",
                    "stream": False,
                })
            except Exception as exc:
                if "legacy-path-reached" in str(exc):
                    reached = True
        finally:
            self.mod._http_client = saved

        # Fake openai must not have been touched
        self.assertIsNone(self.captured["kwargs"])
        # And the legacy path was reached
        self.assertTrue(reached, "legacy vLLM path was not reached")


class TestLegacyBackwardCompat(unittest.TestCase):
    """Legacy clients that don't know about routing must behave
    byte-identically: no routing, no role hints, and the old code path
    is followed."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_chat_api()

    def setUp(self):
        if self.mod is None:
            self.skipTest("fastapi/httpx not importable")

    def test_resolve_role_returns_none_for_legacy_payload(self):
        # No explicit role + classify disabled = no routing decision.
        name, cfg, reason = self.mod._resolve_role(
            explicit_role=None,
            last_user_text="anything at all",
            classify=False,
        )
        self.assertIsNone(name)
        self.assertIsNone(cfg)

    def test_chat_request_rag_default_true(self):
        req = self.mod.ChatRequest(
            messages=[self.mod.Message(role="user", content="x")],
        )
        self.assertTrue(req.rag)
        self.assertTrue(req.stream)


if __name__ == "__main__":
    unittest.main(verbosity=2)
