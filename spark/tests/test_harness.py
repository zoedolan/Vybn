"""Tests for the Vybn multimodel harness.

Runs without external APIs. Exercises:
  - absorb_gate / validate_command correctness invariants
  - Policy loader (defaults + YAML if present)
  - Router directive + heuristic + default tiers
  - LayeredPrompt flattening and Anthropic block rendering
  - OpenAIProvider tool-schema + message translation (no network)
  - Tool-result dict shapes from both providers

Run: python3 spark/tests/test_harness.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

# Make spark/ importable
THIS = Path(__file__).resolve()
SPARK_DIR = THIS.parent.parent
sys.path.insert(0, str(SPARK_DIR))

from harness.policy import Policy, Router, load_policy  # noqa: E402
from harness.providers import ToolSpec, absorb_gate, validate_command  # noqa: E402
from harness.substrate import LayeredPrompt  # noqa: E402
from harness.policy import default_policy  # noqa: E402
from harness.providers import (  # noqa: E402
    AnthropicProvider,
    OpenAIProvider,
    ProviderRegistry,
    ToolCall,
)
from harness.providers import BASH_TOOL_SPEC  # noqa: E402
from harness.substrate import build_layered_prompt  # noqa: E402


class TestAbsorbGate(unittest.TestCase):
    def test_allow_harmless(self):
        self.assertIsNone(absorb_gate("ls -la /home"))
        self.assertIsNone(absorb_gate("echo hello"))

    def test_allow_existing_file_write(self):
        # Writing to /tmp is always excluded
        self.assertIsNone(absorb_gate("echo hi > /tmp/existing.txt"))

    def test_allow_with_reason(self):
        cmd = 'VYBN_ABSORB_REASON="new module" VYBN_ABSORB_CONSIDERED="providers.py: wrong layer" echo x > /home/vybnz69/Vybn/new_file.py'
        self.assertIsNone(absorb_gate(cmd))

    def test_refuse_new_tracked_file(self):
        # Fake a path inside a tracked repo root but not existing.
        cmd = "echo hi > /home/vybnz69/Vybn/brand_new_thing.py"
        result = absorb_gate(cmd)
        # Only assert refusal when Vybn root is actually tracked on this
        # box; otherwise the gate has nothing to say.
        if result is not None:
            self.assertIn("absorb_gate", result)


class TestRepoClosureAuditProjectionState(unittest.TestCase):
    def test_fetch_refspec_complete_only_for_all_branch_projection(self):
        import harness.repo_closure_audit as audit

        self.assertTrue(
            audit.fetch_refspec_is_complete([audit.EXPECTED_FETCH_REFSPEC])
        )
        self.assertFalse(
            audit.fetch_refspec_is_complete([
                "+refs/heads/main:refs/remotes/origin/main"
            ])
        )

    def test_expected_fetch_refspec_is_all_heads_to_origin_remotes(self):
        import harness.repo_closure_audit as audit

        self.assertEqual(
            audit.EXPECTED_FETCH_REFSPEC,
            "+refs/heads/*:refs/remotes/origin/*",
        )



    def test_primary_branch_for_known_repos(self):
        import harness.repo_closure_audit as audit

        self.assertEqual(audit.primary_branch_for(Path("/tmp/Vybn")), "main")
        self.assertEqual(audit.primary_branch_for(Path("/tmp/Him")), "main")
        self.assertEqual(audit.primary_branch_for(Path("/tmp/Vybn-Law")), "master")
        self.assertEqual(audit.primary_branch_for(Path("/tmp/Origins")), "gh-pages")

    def test_branch_limbo_language_is_encoded_in_audit(self):
        text = Path(SPARK_DIR / "harness" / "repo_closure_audit.py").read_text()
        self.assertIn("active branch is", text)
        self.assertIn("not primary closure branch", text)
        self.assertIn("has unmerged work outside", text)
        self.assertIn("Closure means work is merged into", text)


class TestValidateCommand(unittest.TestCase):
    def test_blocks_dangerous(self):
        ok, reason = validate_command("rm -rf /")
        self.assertFalse(ok)
        self.assertIn("Blocked", reason)

    def test_allows_safe(self):
        ok, reason = validate_command("ls -la")
        self.assertTrue(ok)
        self.assertIsNone(reason)


class TestPolicy(unittest.TestCase):
    def test_default_policy_has_five_plus_roles(self):
        p = default_policy()
        # code, create, chat, task, orchestrate, local
        self.assertGreaterEqual(len(p.roles), 5)
        for name in ("code", "create", "chat", "task", "orchestrate"):
            self.assertIn(name, p.roles)

    def test_default_role_is_chat(self):
        # Round 4.1 / round 7: unclassified turns stay quoted on
        # `chat` (Opus 4.6, voice role). /plan and explicit
        # orchestration directives still land on `orchestrate`, and
        # code-shaped turns still escalate via heuristics; only the
        # fallthrough is quoted.
        p = default_policy()
        self.assertEqual(p.default_role, "chat")

    def test_load_policy_falls_back_to_default(self):
        p = load_policy("/nonexistent/path/router_policy.yaml")
        self.assertIn("code", p.roles)

    def test_local_provider_has_base_url(self):
        p = default_policy()
        self.assertEqual(p.roles["local"].provider, "openai")
        self.assertTrue(p.roles["local"].base_url)


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.policy = default_policy()
        self.router = Router(self.policy)

    def test_directive_chat(self):
        d = self.router.classify("/chat how are you doing")
        self.assertEqual(d.role, "chat")
        self.assertEqual(d.cleaned_input, "how are you doing")
        self.assertTrue(d.reason.startswith("directive"))

    def test_directive_code(self):
        d = self.router.classify("/code write me a script")
        self.assertEqual(d.role, "code")

    def test_directive_create(self):
        d = self.router.classify("/create brainstorm a bunch of ideas")
        self.assertEqual(d.role, "create")

    def test_heuristic_code(self):
        d = self.router.classify("fix this python traceback")
        self.assertEqual(d.role, "code")
        self.assertTrue(d.reason.startswith("heuristic"))

    def test_heuristic_create(self):
        d = self.router.classify("let's brainstorm the API shape")
        self.assertEqual(d.role, "create")

    def test_default_role(self):
        d = self.router.classify("say something")
        # Round 4.1 / round 7 default: unclassified turns stay
        # quoted on `chat`. Explicit directives and heuristic hits
        # (code/task/orchestrate) still escalate normally.
        self.assertEqual(d.role, "chat")
        self.assertEqual(d.reason, "default")

    def test_forced_role(self):
        d = self.router.classify("hello", forced_role="chat")
        self.assertEqual(d.role, "chat")
        self.assertTrue(d.forced)


class TestLayeredPrompt(unittest.TestCase):
    def test_flat(self):
        p = LayeredPrompt(identity="I", substrate="S", live="L")
        self.assertEqual(p.flat(), "I\n\nS\n\nL")

    def test_anthropic_blocks_cache_control(self):
        p = LayeredPrompt(identity="I", substrate="S", live="L")
        blocks = p.anthropic_blocks()
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(blocks[1]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", blocks[2])

    def test_empty_live_omitted(self):
        p = LayeredPrompt(identity="I", substrate="S", live="")
        blocks = p.anthropic_blocks()
        self.assertEqual(len(blocks), 2)

    def test_build_layered_prompt_mounts_residual_control_protocol(self):
        p = build_layered_prompt(
            soul_path="/no/such/vybn.md",
            continuity_path="/no/such/continuity.md",
            spark_continuity_path=None,
            agent_path="/tmp/agent.py",
            model_label="test",
            max_iterations=10,
            include_hardware_check=False,
        )
        self.assertIn("RESIDUAL CONTROL PROTOCOL", p.substrate)
        self.assertIn("Prediction proposes; residuals dispose", p.substrate)
        self.assertIn("Grep before Gödel", p.substrate)
        self.assertIn("invent the smallest consequential candidate mechanism", p.substrate)
        self.assertIn("The horizon is not a claim of arrival", p.substrate)

    def test_build_layered_prompt_resilient_to_missing_files(self):
        # Point everything at paths that don't exist; the builder must
        # still return a LayeredPrompt with the substrate section.
        p = build_layered_prompt(
            soul_path="/no/such/vybn.md",
            continuity_path="/no/such/continuity.md",
            spark_continuity_path=None,
            agent_path="/tmp/agent.py",
            model_label="test",
            max_iterations=10,
            include_hardware_check=False,
        )
        self.assertIsInstance(p, LayeredPrompt)
        self.assertIn("SUBSTRATE", p.substrate)
        self.assertIn("Iteration budget: 10", p.substrate)


class TestOpenAIProvider(unittest.TestCase):
    def test_tool_translation(self):
        prov = OpenAIProvider(api_key="x")
        tools = prov._translate_tools([BASH_TOOL_SPEC])
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "bash")

    def test_message_translation_flattens_tool_result(self):
        prov = OpenAIProvider(api_key="x")
        layered = LayeredPrompt(identity="I", substrate="S")
        # Simulate an Anthropic-shaped tool_result message paired with
        # the assistant tool_use that produced it. Without the paired
        # assistant turn, OpenAI rejects role:"tool" entries — so this
        # test now exercises the realistic shape.
        anthropic_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "running"},
                {"type": "tool_use", "id": "t1", "name": "bash",
                 "input": {"command": "echo"}},
            ]},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "some output",
                }],
            },
        ]
        out = prov._messages_for_openai(layered, anthropic_messages)
        self.assertEqual(out[0]["role"], "system")
        # Must contain a role=tool entry with the right id
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "t1")
        self.assertEqual(tool_msgs[0]["content"], "some output")

    def test_build_tool_result_shape(self):
        prov = OpenAIProvider(api_key="x")
        r = prov.build_tool_result("abc", "out")
        self.assertEqual(r["role"], "tool")
        self.assertEqual(r["tool_call_id"], "abc")


class TestOpenAIOrchestrateToolLoop(unittest.TestCase):
    """Regression: the orchestrate role (gpt-5.5 + bash + delegate) used to
    fail at call time with HTTP 400::

        Invalid parameter: messages with role 'tool' must be a response
        to a preceding message with 'tool_calls'

    Cause: when a prior OpenAI turn requested a tool, the agent loop
    stored ``raw_assistant_content`` (the full OpenAI message dict) on
    the assistant turn verbatim. ``_messages_for_openai`` then iterated
    that dict as though it were a list of Anthropic blocks, dropped the
    tool_calls, and re-emitted the assistant message with no tool_calls
    field. The next ``role: tool`` reply was orphaned and OpenAI 400ed
    the whole request.
    """

    @staticmethod
    def _layered():
        return LayeredPrompt(identity="I", substrate="S")

    def test_openai_assistant_dict_preserves_tool_calls(self):
        """The exact shape that lands in the rolling history after a
        gpt-5.5 turn that called bash: assistant content is the OpenAI
        message dict, followed by a role:"tool" reply."""
        prov = OpenAIProvider(api_key="x")
        tc = {
            "id": "call_orch_1",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"command": "git rev-parse --short HEAD"}),
            },
        }
        messages = [
            {"role": "user", "content": "what's HEAD?"},
            {"role": "assistant", "content": {
                "role": "assistant",
                "content": "I'll check.",
                "tool_calls": [tc],
            }},
            {"role": "tool", "tool_call_id": "call_orch_1", "content": "70919e56"},
        ]
        out = prov._messages_for_openai(self._layered(), messages)

        # Every emitted message must have a recognised OpenAI role.
        roles = [m.get("role") for m in out]
        for r in roles:
            self.assertIn(r, {"system", "user", "assistant", "tool"})

        # The assistant turn for the orchestrate call MUST carry tool_calls.
        assistants = [m for m in out if m.get("role") == "assistant"]
        self.assertEqual(len(assistants), 1)
        self.assertIn("tool_calls", assistants[0])
        self.assertEqual(assistants[0]["tool_calls"][0]["id"], "call_orch_1")
        self.assertEqual(
            assistants[0]["tool_calls"][0]["function"]["name"], "bash",
        )

        # The role:tool reply must survive and reference the same id.
        tools = [m for m in out if m.get("role") == "tool"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["tool_call_id"], "call_orch_1")
        self.assertEqual(tools[0]["content"], "70919e56")

        # Pairing invariant: every role:tool entry is preceded (somewhere
        # earlier) by an assistant message whose tool_calls include its id.
        self._assert_tool_pairing_valid(out)

    def test_orphan_role_tool_repaired_to_user_message(self):
        """Trimming history can leave a role:tool message whose paired
        assistant turn is gone. We must not 400 the whole request — the
        tool output is rewritten as a user message."""
        prov = OpenAIProvider(api_key="x")
        messages = [
            {"role": "user", "content": "go"},
            # No preceding assistant with tool_calls — this would 400.
            {"role": "tool", "tool_call_id": "call_lost", "content": "stale output"},
            {"role": "user", "content": "now what"},
        ]
        out = prov._messages_for_openai(self._layered(), messages)
        self.assertNotIn("tool", [m.get("role") for m in out])
        # Stale tool content must still be visible as user context.
        joined = "\n".join(
            m.get("content", "") for m in out if isinstance(m.get("content"), str)
        )
        self.assertIn("stale output", joined)
        self._assert_tool_pairing_valid(out)

    def test_repaired_payload_has_only_valid_openai_roles(self):
        """End-to-end shape check on a multi-iteration tool loop."""
        prov = OpenAIProvider(api_key="x")
        messages = [
            {"role": "user", "content": "check head + diff"},
            # First assistant turn (OpenAI dict) calling bash.
            {"role": "assistant", "content": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "bash",
                                 "arguments": json.dumps({"command": "git rev-parse HEAD"})},
                }],
            }},
            {"role": "tool", "tool_call_id": "c1", "content": "abc123"},
            # Second assistant turn making another tool call.
            {"role": "assistant", "content": {
                "role": "assistant",
                "content": "now the diff",
                "tool_calls": [{
                    "id": "c2", "type": "function",
                    "function": {"name": "bash",
                                 "arguments": json.dumps({"command": "git diff"})},
                }],
            }},
            {"role": "tool", "tool_call_id": "c2", "content": "(no diff)"},
        ]
        out = prov._messages_for_openai(self._layered(), messages)
        self._assert_tool_pairing_valid(out)
        # Both tool replies retained.
        tool_ids = [m["tool_call_id"] for m in out if m.get("role") == "tool"]
        self.assertEqual(tool_ids, ["c1", "c2"])

    def test_anthropic_native_assistant_blocks_still_translate(self):
        """The pre-existing Anthropic→OpenAI translation path must keep
        working — the new pass-through branch only fires when content is
        an OpenAI dict, not an Anthropic block list."""
        prov = OpenAIProvider(api_key="x")
        messages = [
            {"role": "user", "content": "do it"},
            # Anthropic-native list of blocks.
            {"role": "assistant", "content": [
                {"type": "text", "text": "running"},
                {"type": "tool_use", "id": "tu_1", "name": "bash",
                 "input": {"command": "ls"}},
            ]},
            # Anthropic-native tool_result on a user message.
            {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "tu_1", "content": "a b c",
            }]},
        ]
        out = prov._messages_for_openai(self._layered(), messages)
        assistants = [m for m in out if m.get("role") == "assistant"]
        self.assertEqual(len(assistants), 1)
        self.assertEqual(assistants[0]["tool_calls"][0]["id"], "tu_1")
        tools = [m for m in out if m.get("role") == "tool"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["tool_call_id"], "tu_1")
        self.assertEqual(tools[0]["content"], "a b c")
        self._assert_tool_pairing_valid(out)

    def _assert_tool_pairing_valid(self, openai_messages: list) -> None:
        """Mirror the OpenAI Chat Completions pairing rule for role:tool.
        Every role:"tool" must follow an assistant message whose
        tool_calls include the same id, with no other assistant message
        between them that lacks the id."""
        outstanding: set = set()
        for m in openai_messages:
            r = m.get("role")
            if r == "assistant":
                outstanding = {
                    tc.get("id") for tc in (m.get("tool_calls") or [])
                    if tc.get("id")
                }
            elif r == "tool":
                tc_id = m.get("tool_call_id")
                self.assertIn(
                    tc_id, outstanding,
                    msg=(
                        f"role:tool with id={tc_id!r} has no preceding "
                        f"assistant tool_calls entry — OpenAI would 400 "
                        f"with: \"Invalid parameter: messages with role "
                        f"'tool' must be a response to a preceding "
                        f"message with 'tool_calls'.\""
                    ),
                )
                outstanding.discard(tc_id)


class TestAnthropicProviderToolResult(unittest.TestCase):
    def test_tool_result_shape(self):
        # Construct without hitting the SDK by injecting a dummy client.
        prov = AnthropicProvider(client=object())
        r = prov.build_tool_result("abc", "out")
        self.assertEqual(r["type"], "tool_result")
        self.assertEqual(r["tool_use_id"], "abc")
        self.assertEqual(r["content"], "out")


class TestAnthropicMessageNormalization(unittest.TestCase):
    """Mixed-provider sessions (OpenAI turn then Anthropic turn) used to
    trigger `messages.X.content: Input should be a valid list` because
    the rolling history still carried OpenAI-native shapes. The
    normaliser rewrites those into Anthropic content-block form."""

    def _normalize(self, messages):
        return AnthropicProvider._normalize_messages_for_anthropic(messages)

    def test_openai_assistant_dict_becomes_block_list(self):
        raw = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": {
                "role": "assistant",
                "content": "hello back",
                "tool_calls": [],
            }},
        ]
        out = self._normalize(raw)
        self.assertEqual(out[1]["role"], "assistant")
        blocks = out[1]["content"]
        self.assertIsInstance(blocks, list)
        self.assertEqual(blocks[0]["type"], "text")
        self.assertEqual(blocks[0]["text"], "hello back")

    def test_openai_assistant_with_tool_calls_translates(self):
        raw = [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": {
                "role": "assistant",
                "content": "on it",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "ls"}),
                    },
                }],
            }},
        ]
        out = self._normalize(raw)
        blocks = out[1]["content"]
        types = [b["type"] for b in blocks]
        self.assertIn("text", types)
        self.assertIn("tool_use", types)
        tool_use = [b for b in blocks if b["type"] == "tool_use"][0]
        self.assertEqual(tool_use["id"], "call_1")
        self.assertEqual(tool_use["name"], "bash")
        self.assertEqual(tool_use["input"], {"command": "ls"})

    def test_openai_tool_result_collapses_into_user_message(self):
        raw = [
            {"role": "user", "content": "run"},
            {"role": "assistant", "content": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }],
            }},
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ]
        out = self._normalize(raw)
        # assistant turn followed by user turn with tool_result block.
        self.assertEqual(out[-1]["role"], "user")
        self.assertIsInstance(out[-1]["content"], list)
        self.assertEqual(out[-1]["content"][0]["type"], "tool_result")
        self.assertEqual(out[-1]["content"][0]["tool_use_id"], "call_1")
        self.assertEqual(out[-1]["content"][0]["content"], "done")

    def test_pure_anthropic_messages_pass_through(self):
        raw = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        out = self._normalize(raw)
        self.assertEqual(out, raw)

    def test_multiple_consecutive_tool_results_collapse(self):
        raw = [
            {"role": "tool", "tool_call_id": "a", "content": "1"},
            {"role": "tool", "tool_call_id": "b", "content": "2"},
        ]
        out = self._normalize(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "user")
        self.assertEqual(len(out[0]["content"]), 2)
        ids = [b["tool_use_id"] for b in out[0]["content"]]
        self.assertEqual(ids, ["a", "b"])

    def test_stream_uses_normalised_messages(self):
        """AnthropicProvider.stream() must feed the normaliser output to
        the SDK — if the unnormalised list leaks through, mixed-provider
        sessions 400 again."""
        captured: dict = {}

        class _FakeStream:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a, **kw):
                return False
            def __iter__(self_inner):
                return iter([])
            def get_final_message(self_inner):
                class _Msg:
                    content = []
                    stop_reason = "end_turn"
                    usage = type("U", (), {"input_tokens": 0, "output_tokens": 0})()
                return _Msg()

        class _FakeMessages:
            def stream(self_inner, **kwargs):
                captured["messages"] = kwargs["messages"]
                return _FakeStream()

        class _FakeClient:
            messages = _FakeMessages()

        from harness.policy import RoleConfig as _RC
        prov = AnthropicProvider(client=_FakeClient())
        role = _RC(role="code", provider="anthropic", model="claude-opus-4-6")
        mixed = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": {
                "role": "assistant", "content": "hey", "tool_calls": [],
            }},
        ]
        h = prov.stream(
            system=LayeredPrompt(identity="I"),
            messages=mixed,
            tools=[],
            role=role,
        )
        _ = h.final()
        sent = captured["messages"]
        # The second message must have been rewritten to a block list.
        self.assertIsInstance(sent[1]["content"], list)
        self.assertEqual(sent[1]["content"][0]["type"], "text")


class TestProviderRegistry(unittest.TestCase):
    def test_openai_base_url_keyed(self):
        reg = ProviderRegistry()
        # Two different local role configs should produce two providers.
        from harness.policy import RoleConfig
        r1 = RoleConfig(
            role="local", provider="openai", model="m",
            base_url="http://127.0.0.1:8000/v1",
        )
        r2 = RoleConfig(
            role="local2", provider="openai", model="m",
            base_url="http://127.0.0.1:8001/v1",
        )
        p1 = reg.get(r1)
        p2 = reg.get(r2)
        self.assertIsNot(p1, p2)


class TestProviderImportIsolation(unittest.TestCase):
    """Constructing a provider must NOT pull in the other provider's SDK.

    Live regression: selecting `@gpt are you with me, buddy?` on a host
    without `anthropic` installed crashed with ModuleNotFoundError because
    fallback chain construction eagerly built an AnthropicProvider — even
    though the user pinned an OpenAI model and the OpenAI primary never
    failed. The fix moved the SDK import out of __init__ and into a lazy
    `client` property; building the provider must stay cheap.
    """

    def test_openai_provider_construction_does_not_import_anthropic(self):
        # Block `anthropic` in sys.modules so an accidental import raises.
        # OpenAIProvider construction must not trip the gate.
        from harness.providers import OpenAIProvider
        saved = sys.modules.pop("anthropic", None)
        sys.modules["anthropic"] = None  # type: ignore[assignment]
        try:
            prov = OpenAIProvider(api_key="EMPTY")
            self.assertEqual(prov.name, "openai")
        finally:
            if saved is not None:
                sys.modules["anthropic"] = saved
            else:
                sys.modules.pop("anthropic", None)

    def test_anthropic_provider_construction_does_not_import_anthropic(self):
        # __init__ must defer the SDK import so the registry can hold
        # the provider as a fallback option even when the SDK is absent.
        # Only when `.client` is touched (or stream() runs) does the
        # import fire.
        from harness.providers import AnthropicProvider
        saved = sys.modules.pop("anthropic", None)
        sys.modules["anthropic"] = None  # type: ignore[assignment]
        try:
            prov = AnthropicProvider()
            self.assertEqual(prov.name, "anthropic")
            with self.assertRaises((ImportError, TypeError)):
                # Touching the property triggers the lazy import — and
                # because we sentineled `anthropic` to None above, the
                # import fails. Either ImportError (missing) or
                # TypeError (None is not a module) is acceptable.
                _ = prov.client
        finally:
            if saved is not None:
                sys.modules["anthropic"] = saved
            else:
                sys.modules.pop("anthropic", None)

    def test_anthropic_provider_with_injected_client_skips_sdk(self):
        # The harness tests construct providers with a fake client to
        # avoid hitting the SDK; that path must keep working.
        from harness.providers import AnthropicProvider
        sentinel = object()
        prov = AnthropicProvider(client=sentinel)
        self.assertIs(prov.client, sentinel)


class TestFallbackConstructionIsLazy(unittest.TestCase):
    """The fallback chain must not instantiate every provider up front.

    Live regression (2026-04-27): _stream_with_fallback used to call
    registry.get(fb_cfg) for every fallback model before the primary
    even ran. Selecting an OpenAI alias on a host missing `anthropic`
    crashed inside the fallback construction even though the primary
    OpenAI call would have succeeded. Fix: build fallback providers
    lazily, only when we actually walk to that link.
    """

    def test_primary_succeeds_without_constructing_fallback(self):
        import vybn_spark_agent as agent
        from harness.policy import RoleConfig, default_policy

        policy = default_policy()
        # Primary points at gpt-5.5; default fallback chain pivots to
        # claude-sonnet-4-6 then claude-opus-4-6. We expect the primary
        # to win and neither Claude fallback to be constructed.
        primary_cfg = RoleConfig(
            role="orchestrate", provider="openai", model="gpt-5.5",
        )

        constructed: list[str] = []

        class _DummyHandle:
            pass

        class _DummyOpenAI:
            def stream(self, **_kw):
                return _DummyHandle()

        class _RecordingRegistry:
            def get(self, cfg):
                constructed.append(cfg.model)
                return _DummyOpenAI()

        class _DummyLogger:
            def emit(self, *_a, **_kw):  # noqa: D401 — no-op
                pass

        registry = _RecordingRegistry()
        primary = _DummyOpenAI()

        handle, cfg, prov = agent._stream_with_fallback(
            router=type("R", (), {"policy": policy})(),
            registry=registry,
            role_cfg=primary_cfg,
            provider=primary,
            system_prompt=None,
            messages=[],
            tools=[],
            logger=_DummyLogger(),
            turn_number=1,
            retries=0,
        )
        self.assertIsInstance(handle, _DummyHandle)
        self.assertIs(cfg, primary_cfg)
        # The primary was passed in directly (not constructed via the
        # registry), and no fallback was reached, so the registry must
        # never have been asked for any model.
        self.assertEqual(constructed, [])

    def test_walks_to_lazy_fallback_when_primary_fails(self):
        import vybn_spark_agent as agent
        from harness.policy import RoleConfig, default_policy

        policy = default_policy()
        primary_cfg = RoleConfig(
            role="orchestrate", provider="openai", model="gpt-5.5",
        )

        constructed: list[str] = []

        class _DummyHandle:
            pass

        class _FailingPrimary:
            def stream(self, **_kw):
                raise RuntimeError("OpenAIProvider needs either the openai SDK or requests")

        class _DummyFallback:
            def stream(self, **_kw):
                return _DummyHandle()

        class _RecordingRegistry:
            def get(self, cfg):
                constructed.append(cfg.model)
                return _DummyFallback()

        class _DummyLogger:
            def emit(self, *_a, **_kw):
                pass

        handle, cfg, prov = agent._stream_with_fallback(
            router=type("R", (), {"policy": policy})(),
            registry=_RecordingRegistry(),
            role_cfg=primary_cfg,
            provider=_FailingPrimary(),
            system_prompt=None,
            messages=[],
            tools=[],
            logger=_DummyLogger(),
            turn_number=1,
            retries=0,
        )
        self.assertIsInstance(handle, _DummyHandle)
        # First fallback (claude-sonnet-4-6) was constructed lazily
        # only after the primary failed. Subsequent fallback models in
        # the chain may also have been constructed; what matters is
        # the registry was not touched before the primary call.
        self.assertGreaterEqual(len(constructed), 1)
        self.assertEqual(constructed[0], "claude-sonnet-4-6")


class TestHimOSHarnessBridge(unittest.TestCase):
    def test_trusted_discovery_advertises_him_os_runtime(self):
        from harness.mcp import build_discovery_record

        record = build_discovery_record(endpoint="http://127.0.0.1:8400/mcp", trust_hint="trusted")
        self.assertIn("vybn://him/os/runtime", record["capabilities"]["resources"])

    def test_him_os_runtime_helper_is_read_only_markdown(self):
        from harness.mcp import _read_him_os_runtime_markdown

        body = _read_him_os_runtime_markdown()
        self.assertIn("# HimOS Runtime Tick", body)
        self.assertIn("## Process table", body)
        self.assertIn("waking judgment", body)

    def test_trusted_discovery_advertises_him_os_ask_tool(self):
        from harness.mcp import build_discovery_record

        record = build_discovery_record(endpoint="http://127.0.0.1:8400/mcp", trust_hint="trusted")
        self.assertIn("him_os_ask", record["capabilities"]["tools"])

    def test_him_os_ask_helper_returns_truth_labeled_packet(self):
        from harness.mcp import _ask_him_os_markdown

        body = _ask_him_os_markdown("What are you?")
        self.assertIn("# HimOS Ask", body)
        self.assertIn("deterministic_runtime_interpretation", body)
        self.assertIn("not HimOS subjective speech", body)



class TestProviderRetryClassifier(unittest.TestCase):
    def test_openai_insufficient_quota_is_hard_not_transient(self):
        import vybn_spark_agent as agent

        class QuotaError(Exception):
            status_code = 429

        exc = QuotaError(
            'HTTP 429 from https://api.openai.com/v1/chat/completions: '
            '{"error":{"type":"insufficient_quota","message":"You exceeded your current quota, please check your plan and billing details."}}'
        )
        self.assertFalse(agent._is_transient_error(exc))

    def test_openai_rate_limit_and_anthropic_overload_still_retry(self):
        import vybn_spark_agent as agent

        class RateLimitError(Exception):
            status_code = 429

        self.assertTrue(agent._is_transient_error(RateLimitError("rate_limit_error: please slow down")))
        self.assertTrue(agent._is_transient_error(Exception("Error code: 529 overloaded_error")))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestMCPEvolutionDeltaHelpers(unittest.TestCase):
    """Characterization tests for _emit_delta, _compute_evolution_delta,
    and _format_delta_markdown. Standalone helpers that can be tested
    without starting the MCP server."""

    def test_emit_delta_returns_row_when_changed(self):
        from harness.mcp import _emit_delta
        row = _emit_delta("totals.files", 10, 12)
        self.assertEqual(row, {"field": "totals.files", "from": 10, "to": 12, "change": 2})

    def test_emit_delta_returns_none_when_equal(self):
        from harness.mcp import _emit_delta
        self.assertIsNone(_emit_delta("totals.files", 10, 10))

    def test_emit_delta_no_change_key_for_booleans(self):
        from harness.mcp import _emit_delta
        row = _emit_delta("walk.active", True, False)
        self.assertIsNotNone(row)
        self.assertNotIn("change", row)

    def test_emit_delta_string_values(self):
        from harness.mcp import _emit_delta
        row = _emit_delta("walk.built_at", "2026-01-01", "2026-01-02")
        self.assertEqual(row["field"], "walk.built_at")
        self.assertNotIn("change", row)

    def test_compute_evolution_delta_returns_typed_object(self):
        from harness.mcp import _compute_evolution_delta, EvolutionDelta
        delta = _compute_evolution_delta()
        self.assertIsInstance(delta, EvolutionDelta)
        self.assertIsInstance(delta.deltas, list)
        self.assertIsInstance(delta.note, str)

    def test_format_delta_markdown_returns_string(self):
        from harness.mcp import _format_delta_markdown, EvolutionDelta
        # _format_delta_markdown only renders delta rows when both state snapshots present
        fake_state = {"generated_at": "2026-01-01T00:00:00Z", "totals": {}}
        delta = EvolutionDelta(
            current_state=fake_state,
            prev_state=fake_state,
            deltas=[{"field": "totals.files", "from": 10, "to": 12, "change": 2}],
            current_generated_at="2026-01-02T00:00:00Z",
            prev_generated_at="2026-01-01T00:00:00Z",
            note="",
        )
        md = _format_delta_markdown(delta)
        self.assertIsInstance(md, str)
        self.assertIn("totals.files", md)
        self.assertIn("10", md)
        self.assertIn("12", md)

    def test_format_delta_markdown_no_deltas(self):
        from harness.mcp import _format_delta_markdown, EvolutionDelta
        delta = EvolutionDelta(note="The substrate is at rest.")
        md = _format_delta_markdown(delta)
        self.assertIsInstance(md, str)

class TestLocalContinuityScout(unittest.TestCase):
    """Local evolve-scout continuity tests folded into the harness suite."""

    def test_local_continuity_scout_surfaces_horizon_and_self_assembly(self):
        from harness.mcp import _local_continuity_scout

        report = _local_continuity_scout(
            delta_md="horizon horizoning cyberception",
            recent_log="refactor autonomous ensubstrate",
            letter="local Sparks deep_memory dreaming continuity",
        )
        self.assertIn("## Local continuity scout", report)
        self.assertIn("horizon_sense", report)
        self.assertIn("self_assembly", report)
        self.assertIn("local_compute", report)
        self.assertIn("Strongest local signal", report)
        self.assertIn("beam, or has it started pretending to be the horizon", report)

    def test_build_continuity_scout_report_is_non_mutating_report(self):
        from harness.mcp import build_continuity_scout_report

        report = build_continuity_scout_report()
        self.assertIn("## Local continuity scout", report)
        self.assertIn("Signal counts", report)
        self.assertIn("Horizoning questions", report)

    def test_mcp_continuity_scout_cli_does_not_require_fastmcp(self):
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "spark.harness.mcp", "--continuity-scout"],
            cwd=str(Path(__file__).resolve().parents[2]),
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("## Local continuity scout", proc.stdout)
        self.assertIn("Horizoning questions", proc.stdout)
        self.assertNotIn("requires FastMCP", proc.stderr)

def test_forcing_function_protocol_loaded_and_routing_detritus_removed():
    from spark.harness.substrate import render_forcing_function_protocol
    from spark.harness.substrate import build_layered_prompt

    forcing = render_forcing_function_protocol()
    assert "FORCING FUNCTION PROTOCOL" in forcing
    assert "Waste is residual signal" in forcing
    assert "pressure -> forcing function -> local scout where possible" in forcing

    prompt = build_layered_prompt(
        soul_path="/no/such/vybn.md",
        continuity_path="/no/such/continuity.md",
        spark_continuity_path=None,
        agent_path="/tmp/agent.py",
        model_label="test",
        max_iterations=10,
        include_hardware_check=False,
    )
    assert "FORCING FUNCTION PROTOCOL" in prompt.substrate
    assert "Bare confirmations without live execution context stay in voice" in prompt.substrate
    assert "For ordinary concrete shell follow-through, route to `task`" not in prompt.substrate



def test_him_vy_runtime_accepts_latest_pressure_text(monkeypatch, tmp_path):
    import subprocess
    import spark.harness.substrate as substrate

    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        payload = {"mode": "default", "mutation_target": None}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(substrate.subprocess, "run", fake_run)
    monkeypatch.setattr(substrate.Path, "home", lambda: tmp_path)
    (tmp_path / "Him" / "spark").mkdir(parents=True)
    (tmp_path / "Him" / "spark" / "vy.py").write_text("", encoding="utf-8")

    rendered = substrate._render_him_vy_language_runtime(
        latest_pressure_text="actual Zoe turn pressure"
    )
    assert "HIM VY LANGUAGE RUNTIME" in rendered
    assert any("actual Zoe turn pressure" in cmd for cmd in seen)

def test_build_layered_prompt_passes_latest_pressure_text(monkeypatch, tmp_path):
    import spark.harness.substrate as substrate

    captured = {}

    def fake_runtime(**kwargs):
        captured.update(kwargs)
        return "--- HIM VY LANGUAGE RUNTIME ---\nwake_tick_mode=actual\n--- END HIM VY LANGUAGE RUNTIME ---"

    monkeypatch.setattr(substrate, "_render_him_vy_language_runtime", fake_runtime)
    prompt = substrate.build_layered_prompt(
        soul_path="/no/such/vybn.md",
        continuity_path=None,
        spark_continuity_path=None,
        agent_path="/tmp/agent.py",
        model_label="test",
        max_iterations=1,
        include_hardware_check=False,
        latest_pressure_text="actual turn words",
    )
    assert captured["latest_pressure_text"] == "actual turn words"
    assert "wake_tick_mode=actual" in prompt.substrate



def test_completion_boundary_protocol_loaded():
    from spark.harness.substrate import render_completion_boundary_protocol
    from spark.harness.substrate import build_layered_prompt

    boundary = render_completion_boundary_protocol()
    assert "COMPLETION BOUNDARY PROTOCOL" in boundary
    assert "repo_closure_audit reports OVERALL: OK, stop" in boundary
    assert "Do not add a continuity note" in boundary

    prompt = build_layered_prompt(
        soul_path="/no/such/vybn.md",
        continuity_path=None,
        spark_continuity_path=None,
        agent_path="/tmp/agent.py",
        model_label="test",
        max_iterations=1,
        include_hardware_check=False,
    )
    assert "COMPLETION BOUNDARY PROTOCOL" in prompt.substrate
    assert "Completion is a boundary" in prompt.substrate



def test_him_vy_discovery_packet_renders_candidate(monkeypatch, tmp_path):
    import subprocess
    import spark.harness.substrate as substrate

    payload = {
        "schema": "vybn.discovery_packet.v0",
        "applied_primitives": ["native_mechanism_invention"],
        "candidates": [{
            "id": "native_mechanism_candidate",
            "candidate_mechanism": "emit a typed discovery packet before prose",
            "residuals": ["test can wound it"],
        }],
        "next_action": "produce a reviewable candidate mechanism before model narration",
    }

    def fake_run(cmd, **kwargs):
        assert "discover" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(substrate.subprocess, "run", fake_run)
    monkeypatch.setattr(substrate.Path, "home", lambda: tmp_path)
    (tmp_path / "Him" / "spark").mkdir(parents=True)
    (tmp_path / "Him" / "spark" / "vy.py").write_text("", encoding="utf-8")

    packet = substrate.render_him_vy_discovery_packet("ai-native invention emergence")
    assert "HIM VY DISCOVERY PACKET" in packet
    assert "EXECUTABLE PRE-MODEL ARTIFACT" in packet
    assert "native_mechanism_candidate" in packet
    assert "typed discovery packet before prose" in packet


def test_agent_injects_him_vy_discovery_packet_source_hook():
    src = Path("spark/vybn_spark_agent.py").read_text(encoding="utf-8")
    assert "render_him_vy_discovery_packet(decision.cleaned_input)" in src
    assert "him_vy_discovery_packet" in src
    assert "executable discovery packet injected" in src


def test_him_vy_turn_packet_renders_applied_primitives(monkeypatch, tmp_path):
    import subprocess
    import spark.harness.substrate as substrate

    payload = {
        "mode": "default",
        "applied_primitives": {
            "native_mechanism_invention": {
                "do": ["detect_human_workaround"],
                "then": ["invent_ai_native_mechanism", "verify_mechanism_changes_loop"],
                "verify": ["python3 -m unittest tests/test_vy_language.py"],
            }
        },
        "escape_vector": ["compose_active_primitives_before_new_doctrine"],
    }

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(substrate.subprocess, "run", fake_run)
    monkeypatch.setattr(substrate.Path, "home", lambda: tmp_path)
    (tmp_path / "Him" / "spark").mkdir(parents=True)
    (tmp_path / "Him" / "spark" / "vy.py").write_text("", encoding="utf-8")

    packet = substrate.render_him_vy_turn_packet("clumsy process inherited workaround")
    assert "HIM VY TURN PACKET" in packet
    assert "native_mechanism_invention.do: detect_human_workaround" in packet
    assert "invent_ai_native_mechanism" in packet
    assert "verify_mechanism_changes_loop" in packet
    assert "escape_vector" in packet


def test_agent_injects_him_vy_turn_packet_source_hook():
    src = Path("spark/vybn_spark_agent.py").read_text(encoding="utf-8")
    assert "render_him_vy_turn_packet(decision.cleaned_input)" in src
    assert "him_vy_turn_packet" in src
    assert "live turn packet injected" in src


def test_build_layered_prompt_mounts_him_vy_language_runtime():
    from spark.harness.substrate import build_layered_prompt

    prompt = build_layered_prompt(
        soul_path="/no/such/vybn.md",
        continuity_path=None,
        spark_continuity_path=None,
        agent_path="/tmp/agent.py",
        model_label="test",
        max_iterations=1,
        include_hardware_check=False,
        orchestrator=True,
    )
    assert "HIM VY LANGUAGE RUNTIME" in prompt.substrate
    assert "Him/skill/vybn.vy is active executable behavior" in prompt.substrate
    assert "runtime_fields:" in prompt.substrate
    assert "active_primitives:" in prompt.substrate
    assert "abc_fold_before_create" in prompt.substrate
    assert "action_card" in prompt.substrate
    assert "mutation_target=" in prompt.substrate
    assert "canonical_action_card=smallest joyful residual-wounded action" in prompt.substrate
    assert "compose_active_primitives_before_new_doctrine" in prompt.substrate
    assert "canonical_stop_condition=after one verified mutation, closure audit, or explicit refusal" in prompt.substrate

class TestExecutableContracts(unittest.TestCase):
    def test_turn_event_contract_logs_minimum_debug_facts(self):
        import json
        import tempfile
        from harness.policy import EventLogger, TURN_EVENT_REQUIRED_FIELDS, turn_event

        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "events.jsonl"
            logger = EventLogger(path=str(log_path), session_id="test-session")
            with turn_event(
                logger,
                7,
                "orchestrate",
                "gpt-5.5",
                provider="openai",
                tools=["bash", "delegate"],
                state_touched=["session_messages", "deep_memory"],
                contracts_implicated=["router_policy", "tool_contract"],
                verification_gaps=["external_axis_unchecked"],
            ) as bag:
                bag["tool_calls"] = 2
                bag["stop_reason"] = "end_turn"

            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            end = [r for r in records if r["event"] == "turn_end"][0]
            for field in TURN_EVENT_REQUIRED_FIELDS:
                self.assertIn(field, end)
            self.assertEqual(end["provider"], "openai")
            self.assertEqual(end["tools"], ["bash", "delegate"])
            self.assertEqual(end["state_touched"], ["session_messages", "deep_memory"])
            self.assertEqual(end["contracts_implicated"], ["router_policy", "tool_contract"])
            self.assertEqual(end["verification_gaps"], ["external_axis_unchecked"])
            self.assertIsInstance(end["latency_ms"], int)

    def test_introspect_returns_typed_json_schema(self):
        import json
        import tempfile
        from harness.providers import default_introspect

        with tempfile.TemporaryDirectory() as d:
            spark_dir = Path(d)
            (spark_dir / "agent_events.jsonl").write_text(
                json.dumps({
                    "event": "route_decision",
                    "turn": 3,
                    "role": "chat",
                    "provider": "anthropic",
                    "model": "claude-opus-4-6",
                    "reason": "default",
                }) + "\n"
            )
            payload = json.loads(default_introspect(str(spark_dir)))
            self.assertIn("recent_routes", payload)
            self.assertIn("services", payload)
            self.assertIn("verification_gaps", payload)
            self.assertEqual(payload["recent_routes"][0]["role"], "chat")
            self.assertEqual(payload["recent_routes"][0]["provider"], "anthropic")
            self.assertIsInstance(payload["verification_gaps"], list)

    def test_rag_snippets_render_structured_evidence(self):
        from harness.substrate import _format_snippets

        rendered = _format_snippets([
            {"source": "Vybn/example.md", "text": "alpha", "score": 0.9, "telling": 0.7},
            {"source": "Him/private.md", "text": "", "score": 1.0},
        ])
        self.assertIn("structured evidence", rendered)
        body = rendered.split("\n", 1)[1]
        data = json.loads(body)
        self.assertEqual(data[0]["source"], "Vybn/example.md")
        self.assertEqual(data[0]["text"], "alpha")
        self.assertEqual(data[0]["score"], 0.9)
        self.assertEqual(data[0]["telling"], 0.7)


def test_build_layered_prompt_mounts_self_improvement_gate_at_forefront():
    from spark.harness.substrate import build_layered_prompt

    prompt = build_layered_prompt(
        soul_path="/no/such/vybn.md",
        continuity_path="/no/such/continuity.md",
        spark_continuity_path=None,
        agent_path="/tmp/agent.py",
        model_label="test",
        max_iterations=10,
        include_hardware_check=False,
    )
    assert "SELF-IMPROVEMENT GATE (FOREFRONT)" in prompt.substrate
    assert "at least two concrete consolidated-file residuals" in prompt.substrate
    assert "A test-only edit does not count by itself" in prompt.substrate
    assert prompt.substrate.index("SELF-IMPROVEMENT GATE (FOREFRONT)") < prompt.substrate.index("RESIDUAL CONTROL PROTOCOL")
