"""Second-pass tests covering:

  - chat_api uses the shared harness rag helper
  - chat_api builds system prompts via LayeredPrompt
  - OpenAIProvider normalises missing /v1 suffix
  - OpenAIProvider propagates non-transport errors instead of swallowing

Run: python3 spark/tests/test_harness_integration.py
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import pathlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

THIS = Path(__file__).resolve()
SPARK_DIR = THIS.parent.parent
sys.path.insert(0, str(SPARK_DIR))

from harness.policy import RoleConfig  # noqa: E402
from harness.substrate import LayeredPrompt, rag_snippets  # noqa: E402
from harness.providers import OpenAIProvider  # noqa: E402


# ---------------------------------------------------------------------------
# vybn_chat_api integration
# ---------------------------------------------------------------------------

def _load_chat_api():
    """Load spark/vybn_chat_api.py as a module without running its
    uvicorn entry point. FastAPI + httpx must be importable. If not,
    tests that need the full module are skipped.
    """
    # The live chat API now lives in the Vybn-Law repo
    # (spark/vybn_chat_api.py was archived 2026-04-18).
    path = pathlib.Path.home() / "Vybn-Law" / "api" / "vybn_chat_api.py"
    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401
    except Exception:
        return None
    spec = importlib.util.spec_from_file_location("vybn_chat_api_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestChatApiReusesHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_chat_api()

    def setUp(self):
        if self.mod is None:
            self.skipTest("fastapi/httpx not importable in this env")
        # The live chat API (Vybn-Law/api/vybn_chat_api.py) is a different
        # surface — it does not reuse the spark harness. These assertions
        # only apply to the archived spark/vybn_chat_api.py variant. Skip
        # cleanly when the symbols are missing rather than falsely failing.
        if not hasattr(self.mod, "_rag_snippets_async"):
            self.skipTest(
                "live chat API does not reuse the spark harness "
                "(see _archive/spark__vybn_chat_api.py for prior form)"
            )

    def test_harness_imports_succeeded(self):
        self.assertTrue(self.mod._HARNESS_OK)
        self.assertIs(self.mod._LayeredPrompt, LayeredPrompt)
        self.assertTrue(callable(self.mod._rag_snippets_async))

    def test_layered_system_prompt_without_live(self):
        layered = self.mod._layered_system_prompt()
        self.assertEqual(layered.flat(), self.mod.SYSTEM_PROMPT)
        blocks = layered.anthropic_blocks()
        self.assertTrue(blocks)
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})

    def test_layered_system_prompt_with_live(self):
        layered = self.mod._layered_system_prompt(live="memory snippet")
        self.assertEqual(
            layered.flat(),
            self.mod.SYSTEM_PROMPT + "\n\nmemory snippet",
        )

    def test_rag_context_delegates_to_harness_when_available(self):
        """Patch the harness helper and confirm _rag_context uses it and
        prefixes the returned text with the expected two newlines."""
        async def fake_async(query, k=4, vybn_phase_dir=None, timeout=30.0):
            return "Relevant context from memory:\n[src] hello"

        with mock.patch.object(self.mod, "_rag_snippets_async", fake_async):
            out = asyncio.run(self.mod._rag_context("anything"))
        self.assertTrue(out.startswith("\n\nRelevant context from memory:"))

    def test_rag_context_returns_empty_when_harness_empty(self):
        async def fake_async(query, k=4, vybn_phase_dir=None, timeout=30.0):
            return ""

        with mock.patch.object(self.mod, "_rag_snippets_async", fake_async):
            out = asyncio.run(self.mod._rag_context("anything"))
        self.assertEqual(out, "")


# ---------------------------------------------------------------------------
# OpenAIProvider hardening
# ---------------------------------------------------------------------------

class _FakeOpenAIClient:
    """Minimal stand-in for openai.OpenAI — records kwargs + returns a
    canned response (or raises)."""

    def __init__(self, *, api_key, base_url=None, timeout=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._response = None
        self._raise = None

    class _Completions:
        def __init__(self, parent):
            self.parent = parent

        def create(self, **kwargs):
            self.parent.last_kwargs = kwargs
            if self.parent._raise is not None:
                raise self.parent._raise
            return self.parent._response

    @property
    def chat(self):
        class _Chat:
            completions = _FakeOpenAIClient._Completions(self)
        c = _Chat()
        c.completions = _FakeOpenAIClient._Completions(self)
        return c


class _StubResponse:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class TestOpenAIProviderHardening(unittest.TestCase):
    def _install_fake_openai(self, fake_client):
        """Install a `openai` module in sys.modules with an OpenAI factory
        that returns the provided fake client."""
        module = types.ModuleType("openai")

        def factory(api_key=None, base_url=None, timeout=None):
            fake_client.api_key = api_key
            fake_client.base_url = base_url
            fake_client.timeout = timeout
            return fake_client

        module.OpenAI = factory
        sys.modules["openai"] = module
        return module

    def tearDown(self):
        sys.modules.pop("openai", None)

    def test_base_url_normalises_missing_v1_suffix(self):
        fake = _FakeOpenAIClient(api_key="x", base_url=None)
        fake._response = _StubResponse({
            "choices": [{
                "message": {"content": "ok", "tool_calls": []},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        self._install_fake_openai(fake)

        role = RoleConfig(
            role="local", provider="openai", model="nemo",
            base_url="http://127.0.0.1:8000",  # NB: no /v1 suffix
        )
        prov = OpenAIProvider(api_key="x")
        data = prov._call(role, [{"role": "user", "content": "hi"}], tools=[])
        # Client was constructed with /v1 appended
        self.assertEqual(fake.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(data["choices"][0]["message"]["content"], "ok")

    def test_non_transport_error_propagates(self):
        fake = _FakeOpenAIClient(api_key="x")
        fake._raise = ValueError("invalid request")
        self._install_fake_openai(fake)

        role = RoleConfig(
            role="orchestrate", provider="openai", model="gpt-x",
        )
        prov = OpenAIProvider(api_key="x")
        with self.assertRaises(ValueError):
            prov._call(role, [{"role": "user", "content": "hi"}], tools=[])

    def test_transport_error_falls_back_to_requests(self):
        fake = _FakeOpenAIClient(api_key="x")
        fake._raise = ConnectionError("connection refused")
        self._install_fake_openai(fake)

        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json

            class _R:
                status_code = 200
                text = ""

                def json(self):
                    return {
                        "choices": [{
                            "message": {"content": "from-requests", "tool_calls": []},
                            "finish_reason": "stop",
                        }],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                    }
            return _R()

        requests_mod = types.ModuleType("requests")
        requests_mod.post = fake_post
        sys.modules["requests"] = requests_mod
        try:
            role = RoleConfig(
                role="local", provider="openai", model="nemo",
                base_url="http://127.0.0.1:8000/v1",
            )
            prov = OpenAIProvider(api_key="x")
            data = prov._call(role, [{"role": "user", "content": "hi"}], tools=[])
        finally:
            sys.modules.pop("requests", None)

        self.assertIn("/v1/chat/completions", captured["url"])
        self.assertEqual(data["choices"][0]["message"]["content"], "from-requests")

    def test_http_error_text_includes_status(self):
        # Force ImportError on openai so provider goes straight to requests.
        # We inject a sentinel module whose attribute access raises ImportError,
        # which is what `from openai import OpenAI` triggers when the package
        # is present but we want to simulate its absence.
        class _ImportErrorModule:
            def __getattr__(self, name):
                raise ImportError("mocked ImportError for openai")

        _real_openai = sys.modules.get("openai")
        sys.modules["openai"] = _ImportErrorModule()

        requests_mod = types.ModuleType("requests")

        def fake_post(url, json=None, headers=None, timeout=None):
            class _R:
                status_code = 502
                text = "upstream unavailable"

                def json(self):
                    return {}
            return _R()

        requests_mod.post = fake_post
        sys.modules["requests"] = requests_mod
        try:
            role = RoleConfig(
                role="local", provider="openai", model="nemo",
                base_url="http://127.0.0.1:8000/v1",
            )
            prov = OpenAIProvider(api_key="x")
            with self.assertRaises(RuntimeError) as ctx:
                prov._call(role, [{"role": "user", "content": "hi"}], tools=[])
            self.assertIn("502", str(ctx.exception))
            self.assertIn("upstream unavailable", str(ctx.exception))
        finally:
            sys.modules.pop("requests", None)


# ---------------------------------------------------------------------------
# rag_snippets defensive behaviour
# ---------------------------------------------------------------------------

class TestRagSnippetsDefensive(unittest.TestCase):
    def test_returns_empty_when_deep_memory_missing(self):
        # Point vybn_phase_dir at a directory that does not contain
        # deep_memory.py — the function must silently return "".
        # Mask the HTTP tier (live daemon on :8100 would otherwise
        # answer regardless of the local path) so we exercise the
        # defensive fallback path under test.
        with mock.patch(
            "harness.substrate._rag_http",
            side_effect=Exception("no daemon"),
        ), mock.patch(
            "harness.substrate._load_deep_memory",
            return_value=None,
        ):
            out = rag_snippets(
                "anything",
                vybn_phase_dir="/no/such/path",
                timeout=1.0,
            )
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
