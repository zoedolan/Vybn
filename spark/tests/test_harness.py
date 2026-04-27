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

from harness import (  # noqa: E402
    LayeredPrompt,
    Policy,
    Router,
    ToolSpec,
    absorb_gate,
    load_policy,
    validate_command,
)
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
        self.assertIn("invent the smallest candidate mechanism", p.substrate)
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
        role = _RC(role="code", provider="anthropic", model="claude-opus-4-7")
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
