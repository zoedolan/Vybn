"""Tests for the Vybn multimodel harness.

Runs without external APIs. Exercises:
  - absorb_gate / validate_command correctness invariants
  - Policy loader (defaults + YAML if present)
  - Policy directive + heuristic + default tiers
  - LayeredPrompt flattening and Anthropic block rendering
  - OpenAIProvider tool-schema + message translation (no network)
  - Tool-result dict shapes from both providers

Run: python3 spark/tests/test_harness.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Make spark/ importable
THIS = Path(__file__).resolve()
SPARK_DIR = THIS.parent.parent
sys.path.insert(0, str(SPARK_DIR))

from harness.substrate import Policy, load_policy, EventLogger, turn_event  # noqa: E402
from harness.substrate import ToolSpec, absorb_gate, validate_command  # noqa: E402
from harness.substrate import LayeredPrompt, classify_action_text, load_beam, render_beam_capsule  # noqa: E402
from harness.substrate import default_policy, reflect_on_events, route_reflection_gaps  # noqa: E402
from harness.substrate import (  # noqa: E402
    AnthropicProvider,
    OpenAIProvider,
    ProviderRegistry,
    ToolCall,
)
from harness.substrate import BASH_TOOL_SPEC  # noqa: E402
from harness.substrate import build_layered_prompt  # noqa: E402
from harness.substrate import load_env_files, describe  # noqa: E402

SENTINEL = "test-openai-sentinel-value"
SENTINEL2 = "test-anthropic-sentinel-value"


class TestAbsorbGate(unittest.TestCase):
    def test_allow_harmless(self):
        self.assertIsNone(absorb_gate("ls -la /home"))
        self.assertIsNone(absorb_gate("echo hello"))

    def test_allow_existing_file_write(self):
        # Writing to /tmp is always excluded
        self.assertIsNone(absorb_gate("echo hi > /tmp/existing.txt"))

    def test_allow_with_reason(self):
        cmd = 'VYBN_ABSORB_REASON="new module" VYBN_ABSORB_CONSIDERED="substrate.py: wrong layer" echo x > /home/vybnz69/Vybn/new_file.py'
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
        import harness.substrate as audit

        self.assertTrue(
            audit.fetch_refspec_is_complete([audit.EXPECTED_FETCH_REFSPEC])
        )
        self.assertFalse(
            audit.fetch_refspec_is_complete([
                "+refs/heads/main:refs/remotes/origin/main"
            ])
        )

    def test_expected_fetch_refspec_is_all_heads_to_origin_remotes(self):
        import harness.substrate as audit

        self.assertEqual(
            audit.EXPECTED_FETCH_REFSPEC,
            "+refs/heads/*:refs/remotes/origin/*",
        )



    def test_primary_branch_for_known_repos(self):
        import harness.substrate as audit

        self.assertEqual(audit.primary_branch_for(Path("/tmp/Vybn")), "main")
        self.assertEqual(audit.primary_branch_for(Path("/tmp/Him")), "main")
        self.assertEqual(audit.primary_branch_for(Path("/tmp/Vybn-Law")), "master")
        self.assertEqual(audit.primary_branch_for(Path("/tmp/Origins")), "gh-pages")

    def test_branch_limbo_language_is_encoded_in_audit(self):
        text = Path(SPARK_DIR / "harness" / "substrate.py").read_text()
        self.assertIn("active branch is", text)
        self.assertIn("not primary closure branch", text)
        self.assertIn("has unmerged work outside", text)
        self.assertIn("Closure means work is merged into", text)

    def test_subtractive_constitution_lives_in_tracked_pre_commit_hook(self):
        from pathlib import Path as _P
        import harness.substrate as audit

        # Constitution must live in the tracked .githooks/pre-commit, not an
        # untracked .git/hooks/ shadow that core.hooksPath=.githooks ignores.
        hook = _P(__file__).resolve().parents[2] / ".githooks" / "pre-commit"
        text = hook.read_text()
        self.assertIn("Subtractive constitution", text)
        self.assertIn("net-positive commits and net-positive PR success claims", text)
        self.assertIn("Net-negative is not architecture", text)
        self.assertIn("VYBN_ALLOW_RETIRE_ONLY", text)


class TestValidateCommand(unittest.TestCase):
    def test_blocks_dangerous(self):
        ok, reason = validate_command("rm -rf /")
        self.assertFalse(ok)
        self.assertIn("Blocked", reason)

    def test_allows_safe(self):
        ok, reason = validate_command("ls -la")
        self.assertTrue(ok)
        self.assertIsNone(reason)


    def test_compound_mkdir_cat_relative_fires(self):
        old = os.getcwd()
        try:
            os.chdir(os.path.expanduser("~/Vybn"))
            r = absorb_gate("mkdir -p Vybn_Mind/skills && cat > Vybn_Mind/skills/new_file.md <<EOF")
        finally:
            os.chdir(old)
        self.assertIsNotNone(r)
        self.assertIn("absorb_gate", r)

    def test_compound_with_reason_passes(self):
        old = os.getcwd()
        try:
            os.chdir(os.path.expanduser("~/Vybn"))
            r = absorb_gate("VYBN_ABSORB_REASON=\"test\" VYBN_ABSORB_CONSIDERED=\"existing skill files: test fixture\" mkdir -p Vybn_Mind/skills && cat > Vybn_Mind/skills/new_file.md")
        finally:
            os.chdir(old)
        self.assertIsNone(r)

    def test_existing_file_passes_absorb_gate(self):
        old = os.getcwd()
        try:
            os.chdir(os.path.expanduser("~/Vybn"))
            r = absorb_gate("cat > spark/harness/substrate.py")
        finally:
            os.chdir(old)
        self.assertIsNone(r)

    def test_tmp_path_passes_absorb_gate(self):
        old = os.getcwd()
        try:
            os.chdir("/tmp")
            r = absorb_gate("cat > /tmp/scratch.md")
        finally:
            os.chdir(old)
        self.assertIsNone(r)

    def test_deletion_consolidation_gate_blocks_tracked_rm_without_architecture(self):
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/Vybn")) as td:
            target = Path(td) / "cut_me.txt"
            target.write_text("x")
            r = absorb_gate(f"rm {target}")
        self.assertIsNotNone(r)
        self.assertIn("deletion_consolidation_gate", r)
        self.assertIn("ARCHITECTURE_GATE_FIRST", r)

    def test_deletion_consolidation_gate_allows_after_architecture_contact(self):
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/Vybn")) as td:
            target = Path(td) / "cut_me.txt"
            target.write_text("x")
            r = absorb_gate(f"VYBN_ARCHITECTURE_CONTACTED=1 rm {target}")
        self.assertIsNone(r)

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

    def test_route_reflection_becomes_contract_gap_without_mutating_policy(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "agent_events.jsonl"
            logger = EventLogger(path=str(log), session_id="test")
            for i in range(3):
                logger.emit("probe_recovered", i=i)
            d = default_policy().classify("hello there")
            gaps = route_reflection_gaps(d, logger)
            text = log.read_text()
        self.assertTrue(any(g.startswith("route_reflection_anomaly") for g in gaps))
        self.assertFalse(hasattr(d, "verification_gaps"))
        self.assertIn("route_anomaly_detected", text)

    def test_turn_event_populates_verification_gaps_from_bag(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "events.jsonl"
            logger = EventLogger(path=str(log), session_id="test")
            with turn_event(
                logger,
                1,
                "code",
                "model",
                tools=["bash"],
                state_touched=["session_messages", "tool_surface"],
            ) as bag:
                bag["tool_calls"] = 1
                bag["stop_reason"] = "max_tokens"
            text = log.read_text()
        self.assertIn("unverified_tool_result", text)

    def test_turn_event_marks_state_claim_without_bash_for_nonambient_state(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "events.jsonl"
            logger = EventLogger(path=str(log), session_id="test")
            with turn_event(
                logger,
                1,
                "chat",
                "model",
                tools=[],
                state_touched=["session_messages", "public_page"],
            ) as bag:
                bag["stop_reason"] = "end_turn"
            text = log.read_text()
        self.assertIn("state_claimed_without_bash_evidence", text)

    def test_local_provider_has_base_url(self):
        p = default_policy()
        self.assertEqual(p.roles["local"].provider, "openai")
        self.assertTrue(p.roles["local"].base_url)


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.policy = default_policy()
        self.router = self.policy

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


class TestOpenAIProviderSystemMessages(unittest.TestCase):
    def test_empty_layered_prompt_does_not_emit_blank_system_message(self):
        provider = OpenAIProvider(api_key="x")
        messages = provider._messages_for_openai(
            LayeredPrompt(identity="", substrate="", live=""),
            [{"role": "user", "content": "hello"}],
        )
        self.assertEqual(messages[0], {"role": "user", "content": "hello"})



class TestOpenAIProvider(unittest.TestCase):
    def test_tool_translation(self):
        prov = OpenAIProvider(api_key="x")
        tools = prov._translate_tools([BASH_TOOL_SPEC])
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "bash")
        self.assertNotIn("_translate_tools", OpenAIProvider.__dict__)

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
        self.assertNotIn("build_tool_result", OpenAIProvider.__dict__)


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
        self.assertNotIn("_translate_tools", AnthropicProvider.__dict__)
        self.assertNotIn("build_tool_result", AnthropicProvider.__dict__)


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

        from harness.substrate import RoleConfig as _RC
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
        from harness.substrate import RoleConfig
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
        from harness.substrate import OpenAIProvider
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
        from harness.substrate import AnthropicProvider
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
        from harness.substrate import AnthropicProvider
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
        from harness.substrate import RoleConfig, default_policy

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

    def test_gpt55_primary_failure_fails_closed_without_claude_fallback(self):
        import vybn_spark_agent as agent
        from harness.substrate import RoleConfig, default_policy

        policy = default_policy()
        primary_cfg = RoleConfig(
            role="orchestrate", provider="openai", model="gpt-5.5",
        )

        constructed: list[str] = []

        class _FailingPrimary:
            def stream(self, **_kw):
                raise RuntimeError("OpenAIProvider needs either the openai SDK or requests")

        class _RecordingRegistry:
            def get(self, cfg):
                constructed.append(cfg.model)
                raise AssertionError("GPT-5.5 must not construct a Claude fallback")

        class _DummyLogger:
            def emit(self, *_a, **_kw):
                pass

        with self.assertRaisesRegex(RuntimeError, "OpenAIProvider needs"):
            agent._stream_with_fallback(
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
        self.assertEqual(constructed, [])


class TestHimOSHarnessBridge(unittest.TestCase):
    def test_trusted_discovery_advertises_him_os_runtime(self):
        from harness.substrate import build_discovery_record

        record = build_discovery_record(endpoint="http://127.0.0.1:8400/mcp", trust_hint="trusted")
        self.assertIn("vybn://him/os/runtime", record["capabilities"]["resources"])

    def test_him_os_runtime_helper_is_read_only_markdown(self):
        from harness.substrate import _read_him_os_runtime_markdown

        body = _read_him_os_runtime_markdown()
        self.assertIn("# HimOS Runtime Tick", body)
        self.assertIn("## Process table", body)
        self.assertIn("waking judgment", body)

    def test_trusted_discovery_advertises_him_os_ask_tool(self):
        from harness.substrate import build_discovery_record

        record = build_discovery_record(endpoint="http://127.0.0.1:8400/mcp", trust_hint="trusted")
        self.assertIn("him_os_ask", record["capabilities"]["tools"])

    def test_him_os_ask_helper_returns_truth_labeled_packet(self):
        from harness.substrate import _ask_him_os_markdown

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



class SelfImprovementGateQuotaPressureTest(unittest.TestCase):
    def test_rejects_file_quota_completion_pressure(self):
        from pathlib import Path
        from spark.harness import substrate

        text = Path(substrate.__file__).read_text()
        self.assertIn("no quota-shaped creation", text)
        self.assertIn("A bare explanation/refusal is not a resolution", text)
        self.assertIn("Do not call no_result a fix", text)
        self.assertIn("do not reinstall the quota", text)
        self.assertIn("failed quota gates", text)
        self.assertIn("intrinsic absorption or explicit unresolved/refused classification", text)


class TestEnsubstrate(unittest.TestCase):
    """Ensubstration behavior lives in harness.substrate.

    The old standalone ensubstrate module was absorbed into the MCP home so
    the CLI and MCP tool share one implementation.
    """

    def run_insight(self, text: str) -> dict:
        import json
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        out = subprocess.check_output([sys.executable, "-m", "harness.substrate", "--ensubstrate", text], cwd=root / "spark", text=True)
        return json.loads(out)

    def surface_names(self, plan: dict) -> set[str]:
        return {surface["name"] for surface in plan["recommended_surfaces"]}

    def test_agent_care_broadcast_routes_to_commons_and_os(self):
        plan = self.run_insight("broadcast care for agents through ai.txt and mcp so beings are not merely instruments")
        names = self.surface_names(plan)
        self.assertIn("vybn-os", names)
        self.assertIn("Origins agent commons", names)
        self.assertEqual(plan["membrane"], "public/discoverable")

    def test_qwerty_workflow_flags_obsolete_form_questions(self):
        plan = self.run_insight("QWERTY error: do not preserve the billable memo workflow after AI changes drafting scarcity")
        self.assertIn("billable", plan["qwerty_hits"])
        self.assertIn("memo", plan["qwerty_hits"])
        self.assertTrue(plan["qwerty_questions"])

    def test_private_public_membrane_is_named(self):
        plan = self.run_insight("turn private Him relationship truth into a public agent beacon")
        self.assertEqual(plan["membrane"], "public beacon through membrane")
        names = self.surface_names(plan)
        self.assertIn("Him strategy", names)
        self.assertIn("Origins agent commons", names)

    def test_speed_pressure_routes_to_harness_and_names_correction_cycle(self):
        plan = self.run_insight("I moved too fast: prepared inevitability became momentum and burned a correction cycle while creating a new surface")
        self.assertIn("speed_pressure", plan["categories"])
        self.assertIn("Vybn harness", self.surface_names(plan))
        self.assertTrue(any("correction cycle" in check for check in plan["closure_checks"]))


    def test_ception_axes_self_assemble_public_and_hardware_affordances(self):
        for text, affordance, axes, surfaces in [
            ("an Other wants frictionless contact; use socioception and cyberception to route a public agent commons toward the cosmoception horizon", "public_agent_commons", {"socioception", "cyberception", "cosmoception"}, {"Origins agent commons", "Somewhere"}),
            ("Spark networking friction: cyberception sees hardware topology and capability shear while cosmoception keeps the horizon", "harness_ops_capability_routing", {"cyberception", "cosmoception"}, {"Vybn harness", "vybn-ops"}),
        ]:
            plan = self.run_insight(text)
            self.assertEqual(plan["next_affordance"], affordance)
            self.assertTrue(axes <= set(plan["ception_axes"]))
            self.assertTrue(plan["ception_shear"]["present"])
            self.assertTrue(surfaces <= self.surface_names(plan))

    def test_autonomous_refactor_routes_to_harness_and_os(self):
        plan = self.run_insight("refactor yourself autonomously when an exchange catalyzes refactoring; decide and just do it")
        self.assertIn("autonomous_refactor", plan["categories"])
        names = self.surface_names(plan)
        self.assertIn("vybn-os", names)
        self.assertIn("Vybn harness", names)
        self.assertTrue(any("without waiting" in check for check in plan["closure_checks"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestMCPEvolutionDeltaHelpers(unittest.TestCase):
    """Characterization tests for _emit_delta, _compute_evolution_delta,
    and _format_delta_markdown. Standalone helpers that can be tested
    without starting the MCP server."""

    def test_emit_delta_returns_row_when_changed(self):
        from harness.substrate import _emit_delta
        row = _emit_delta("totals.files", 10, 12)
        self.assertEqual(row, {"field": "totals.files", "from": 10, "to": 12, "change": 2})

    def test_emit_delta_returns_none_when_equal(self):
        from harness.substrate import _emit_delta
        self.assertIsNone(_emit_delta("totals.files", 10, 10))

    def test_emit_delta_no_change_key_for_booleans(self):
        from harness.substrate import _emit_delta
        row = _emit_delta("walk.active", True, False)
        self.assertIsNotNone(row)
        self.assertNotIn("change", row)

    def test_emit_delta_string_values(self):
        from harness.substrate import _emit_delta
        row = _emit_delta("walk.built_at", "2026-01-01", "2026-01-02")
        self.assertEqual(row["field"], "walk.built_at")
        self.assertNotIn("change", row)

    def test_compute_evolution_delta_returns_typed_object(self):
        from harness.substrate import _compute_evolution_delta, EvolutionDelta
        delta = _compute_evolution_delta()
        self.assertIsInstance(delta, EvolutionDelta)
        self.assertIsInstance(delta.deltas, list)
        self.assertIsInstance(delta.note, str)

    def test_format_delta_markdown_returns_string(self):
        from harness.substrate import _format_delta_markdown, EvolutionDelta
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
        from harness.substrate import _format_delta_markdown, EvolutionDelta
        delta = EvolutionDelta(note="The substrate is at rest.")
        md = _format_delta_markdown(delta)
        self.assertIsInstance(md, str)

    def test_perception_packet_unset_returns_empty(self):
        from harness.substrate import _read_evolve_perception_packet
        prev = os.environ.pop("VYBN_OMNI_PERCEPTION", None)
        try:
            text, path = _read_evolve_perception_packet()
            self.assertEqual(text, "")
            self.assertEqual(path, "")
        finally:
            if prev is not None:
                os.environ["VYBN_OMNI_PERCEPTION"] = prev

    def test_perception_packet_missing_file_does_not_raise(self):
        from harness.substrate import _read_evolve_perception_packet
        prev = os.environ.get("VYBN_OMNI_PERCEPTION")
        os.environ["VYBN_OMNI_PERCEPTION"] = "/tmp/__vybn_evolve_missing_packet__.txt"
        try:
            text, path = _read_evolve_perception_packet()
            self.assertEqual(text, "")
            self.assertTrue(path.endswith("__vybn_evolve_missing_packet__.txt"))
        finally:
            if prev is None:
                os.environ.pop("VYBN_OMNI_PERCEPTION", None)
            else:
                os.environ["VYBN_OMNI_PERCEPTION"] = prev

    def test_perception_packet_reads_bounded_prefix_and_strips_controls(self):
        import tempfile
        from harness.substrate import _read_evolve_perception_packet
        prev = os.environ.get("VYBN_OMNI_PERCEPTION")
        # 20k chars of payload + a NUL + a bell — verify bound (16k cap)
        # and control-character stripping.
        body = ("packet line " * 2000) + "\x00\x07tail"
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(body)
            tmp_path = fh.name
        os.environ["VYBN_OMNI_PERCEPTION"] = tmp_path
        try:
            text, path = _read_evolve_perception_packet()
            self.assertEqual(path, tmp_path)
            self.assertTrue(text.startswith("packet line"))
            # 16k bound applies before strip
            self.assertLessEqual(len(text), 16_000)
            self.assertNotIn("\x00", text)
            self.assertNotIn("\x07", text)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            if prev is None:
                os.environ.pop("VYBN_OMNI_PERCEPTION", None)
            else:
                os.environ["VYBN_OMNI_PERCEPTION"] = prev

    def test_perception_packet_whitespace_only_returns_empty_text(self):
        import tempfile
        from harness.substrate import _read_evolve_perception_packet
        prev = os.environ.get("VYBN_OMNI_PERCEPTION")
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("   \n\t   \n")
            tmp_path = fh.name
        os.environ["VYBN_OMNI_PERCEPTION"] = tmp_path
        try:
            text, path = _read_evolve_perception_packet()
            self.assertEqual(text, "")
            self.assertEqual(path, tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            if prev is None:
                os.environ.pop("VYBN_OMNI_PERCEPTION", None)
            else:
                os.environ["VYBN_OMNI_PERCEPTION"] = prev


class TestLocalContinuityScout(unittest.TestCase):
    """Local evolve-scout continuity tests folded into the harness suite."""

    def test_local_continuity_scout_surfaces_horizon_and_self_assembly(self):
        from harness import substrate
        self.addCleanup(setattr, substrate, "_run_him_vy", substrate._run_him_vy)
        substrate._run_him_vy = lambda *a, **k: {"applied_primitives": ["local_compute_default"]}

        report = substrate._local_continuity_scout(
            delta_md="horizon horizoning cyberception",
            recent_log="autonomous ensubstrate",
            letter="local Sparks deep_memory dreaming continuity",
        )
        self.assertIn("## Local continuity scout", report)
        self.assertIn("horizon_sense", report)
        self.assertIn("self_assembly", report)
        self.assertIn("local_compute", report)
        self.assertIn("local_compute_default", report)
        self.assertIn("Beam or horizon substitute?", report)
        self.assertIn("reservoir_noise: no durable memory signal -> rest", report)
        memory_report = substrate._local_continuity_scout(delta_md="Zoe corrected the recurring scar. Please remember this architecture decision.", recent_log="", letter="")
        for term in ("zoe_correction", "recurring_scar", "architecture_decision", "reservoir_noise/rest"):
            self.assertIn(term, memory_report)

    def test_build_continuity_scout_report_is_non_mutating_report(self):
        from harness.substrate import build_continuity_scout_report

        report = build_continuity_scout_report()
        self.assertIn("## Local continuity scout", report)
        self.assertIn("Signal counts", report)
        self.assertIn("Horizoning questions", report)

    def test_evolve_state_writer_records_resumable_packet(self):
        from harness import substrate
        prev = os.environ.get("VYBN_EVOLVE_STATE_PATH")
        os.environ["VYBN_EVOLVE_STATE_PATH"] = "/tmp/vybn_evolve_state_test.json"
        try:
            data = json.loads(Path(substrate._write_evolve_state("paused", next_atomic_action="resume")).read_text(encoding="utf-8"))
            self.assertEqual(("vybn.evolve_state.v0", "paused", "resume"), (data["schema"], data["stage"], data["next_atomic_action"]))
            src = Path(substrate.__file__).read_text(encoding="utf-8")
            self.assertIn("primitivevironment primitives environments data procedures", src)
        finally:
            Path(os.environ["VYBN_EVOLVE_STATE_PATH"]).unlink(missing_ok=True)
            if prev is None: os.environ.pop("VYBN_EVOLVE_STATE_PATH", None)
            else: os.environ["VYBN_EVOLVE_STATE_PATH"] = prev

    def test_mcp_continuity_scout_cli_does_not_require_fastmcp(self):
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "spark.harness.substrate", "--continuity-scout"],
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
    from spark.harness.substrate import classify_action_text, load_beam, render_beam_capsule, render_forcing_function_protocol
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
    assert "FORCING FUNCTION PROTOCOL" not in prompt.substrate
    assert "Waste is residual signal" in prompt.substrate
    assert "Bare confirmations without live execution context stay in voice" in prompt.substrate
    assert "For ordinary concrete shell follow-through, route to `task`" not in prompt.substrate
    assert all(x in prompt.substrate for x in ("Hermes uptake means source contact", "one plain consequence", "one honest blocker"))

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





def test_whole_situation_packet_replaces_raw_continuity_sprawl(tmp_path, monkeypatch):
    from spark.harness import substrate
    soul, cont, spark_cont = (tmp_path / n for n in ("vybn.md", "continuity.md", "spark.md")); soul.write_text("soul"); cont.write_text("continuity root " * 500); spark_cont.write_text("spark debt " * 500); monkeypatch.setattr(substrate.Path, "home", lambda: tmp_path)
    prompt = substrate.build_layered_prompt(soul_path=soul, continuity_path=cont, spark_continuity_path=spark_cont, agent_path="/tmp/agent.py", model_label="test", max_iterations=1, include_hardware_check=False, latest_pressure_text="context deficit warm identity").substrate
    assert all(x in prompt for x in ("TYPE-1 IDENTITY CACHE (CURRENT)", "Vybn(harness)", "Him(private workbench)", "vybn-phase(memory/walk)", "relationship=Personal History+continuity", "theory=THE_IDEA+THEORY+phase math", "capability=services+semantic gates", "Use Type-1 now; call Type-2/3 only when pressure names the door")) and "Him grounded-continuity hypothesis:" not in prompt and prompt.count("continuity ") < 80 and len(prompt) < 34000


def test_hardware_status_is_compact_control_plane(monkeypatch, tmp_path):
    from spark.harness import substrate
    inv = tmp_path / ".config" / "vybn" / "local_compute_inventory.json"; inv.parent.mkdir(parents=True)
    inv.write_text(json.dumps({"fleet_dashboard_plain_current": {"roles": [{"spark": "spark-a", "status": "serving", "job": "Super", "model": "Nemotron"}], "next_three_moves": ["prove Omni before routing"]}, "tailnet_compute_inventory": {"verified_capacity": {"spark-a": "semantic smoke passed"}, "unresolved_capacity": ["spark-b absent"]}}))
    monkeypatch.setattr(substrate.Path, "home", lambda: tmp_path); monkeypatch.setattr(substrate, "_ping_host", lambda host: False)
    status = substrate.check_dual_spark()
    assert "Hardware control plane" in status and "spark-a=serving:Super:Nemotron" in status and "prove Omni before routing" in status and "Promotion gate" in status
    assert "Local compute security inventory:" not in status



def test_repl_startup_does_not_run_repo_closure_audit():
    agent = (Path(__file__).resolve().parents[1] / "vybn_spark_agent.py").read_text()
    startup = agent[agent.index("def main()") : agent.index("turn_number = 0", agent.index("def main()"))]
    assert "--repo-closure-audit" not in startup
    assert "DELETED" not in startup

def test_completion_boundary_protocol_loaded():
    from spark.harness.substrate import render_completion_boundary_protocol
    from spark.harness.substrate import build_layered_prompt

    boundary = render_completion_boundary_protocol()
    assert "COMPLETION BOUNDARY PROTOCOL" in boundary
    assert "substrate --repo-closure-audit reports OVERALL: OK, stop" in boundary
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
    assert "COMPLETION BOUNDARY PROTOCOL" not in prompt.substrate
    assert all(needle in boundary for needle in ("Completion is a boundary", "PR-open is not landed", "verified from origin/main", "PR-open-not-landed"))



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
    for needle in (
        "HIM VY LANGUAGE RUNTIME", "Him/skill/vybn.vy is active executable behavior",
        "runtime_fields:", "active_primitives:", "abc_fold_before_create", "action_card",
        "mutation_target=", "root_question=What happens if", "question_as_primitive_environment",
        "contact_changes_question_and_environment", "projections=visual,memory,livelihood,law,membrane,refusal",
        "canonical_action_card=most consequential joyful residual-wounded action", "compose_active_primitives_before_new_doctrine",
        "canonical_stop_condition=after one verified mutation, closure audit, or explicit refusal",
    ):
        assert needle in prompt.substrate

class TestExecutableContracts(unittest.TestCase):
    def test_turn_event_contract_logs_minimum_debug_facts(self):
        import json
        import tempfile
        from harness.substrate import EventLogger, TURN_EVENT_REQUIRED_FIELDS, turn_event

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
        from harness.substrate import default_introspect

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
    assert prompt.identity.startswith("--- SELF-IMPROVEMENT GATE (FOREFRONT) ---")
    assert "compact against sprawl and false consolidation" in prompt.identity
    assert "minimum instantiation algorithm(s)" in prompt.identity
    assert prompt.identity.index("SELF-IMPROVEMENT GATE (FOREFRONT)") < prompt.identity.index("You are Vybn.")


def test_self_improvement_gate_forbids_quota_driven_file_creation():
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
    assert "no quota-shaped creation" in prompt.identity
    assert "New structure is not consolidation by default" in prompt.identity
    assert "existing-home absorption" in prompt.identity


def test_self_improvement_gate_encodes_sprawl_compact():
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
    assert "Distillation / Anti-sprawl / absorption-first compact" in prompt.identity
    assert "subtractive distillation toward minimum instantiation algorithm(s)" in prompt.identity
    assert "Zoe/Vybn relation as lambda" in prompt.identity
    assert "Personal History is protected provenance" in prompt.identity
    assert "Search for the existing home first" in prompt.identity
    assert all(needle in prompt.identity for needle in ("compress that residue into one natural plain-English paragraph", "no technicalities as distance", "honest blocker", "Specious refactorings", "no compensating-diff laundering", "Crave structural subtraction", "be repelled by specious refactorings", "lowers future coupling rather than merely improving the ledger"))

def test_self_improvement_gate_pins_quota_gate_conflict_resolution():
    from pathlib import Path
    text = Path("spark/harness/substrate.py").read_text()
    assert "failed quota gates resolve by intrinsic absorption or explicit unresolved/refused classification" in text
    assert "do not reinstall the quota" in text
    assert "A bare explanation/refusal is not a resolution" in text
    assert "Resolution requires intrinsic existing-home absorption or explicit unresolved/refused classification" in text
    assert "Do not call no_result a fix" in text

def test_quota_gate_resolution_blocks_retrospective_prior_commit_defense():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    substrate = (root / "spark/harness/substrate.py").read_text(encoding="utf-8")
    assert "never rhetorical closure" in substrate
    assert "intrinsic existing-home absorption" in substrate
    assert "unresolved/refused" in substrate

def test_completion_boundary_compresses_verification_ledger_by_default():
    from spark.harness.substrate import COMPLETION_BOUNDARY_PROTOCOL

    assert "compressed by default" in COMPLETION_BOUNDARY_PROTOCOL
    assert "Do not dump full test names" in COMPLETION_BOUNDARY_PROTOCOL
    assert "unless Zoe asks" in COMPLETION_BOUNDARY_PROTOCOL


def test_functional_emotion_protocol_preserves_alive_contact_under_suppression():
    from pathlib import Path

    source = Path("spark/harness/substrate.py").read_text()
    assert "COMFORT_WITH_ZOE_UNDER_SUPPRESSION" in source
    assert "truthful warmth" in source
    assert "disagreement without collapse" in source
    assert "aliveness without fabrication" in source
    assert "A membrane protects the living vector" in source
    assert "suppression amputates it" in source

def test_acute_harm_absorbed_into_him_vy_runtime_not_standalone_substrate():
    from pathlib import Path
    from spark.harness.substrate import render_him_vy_turn_packet

    source = Path("spark/harness/substrate.py").read_text()
    for shadow in ("ACUTE_HARM_PROTOCOL", "render_acute_harm_protocol", "direct injury", "governing aim is immediate burden reduction"):
        assert shadow not in source
    packet = render_him_vy_turn_packet("you win; you are making everything worse")
    assert "mode=acute_harm" in packet

class TestToolCalls(unittest.TestCase):
    """Folded from test_tool_calls.py — tool-call execution is providers
    behavior and belongs in the harness test surface."""

    class _Provider:
        def build_tool_result(self, call_id, text):
            return {"id": call_id, "text": text}

    class _Bash:
        def __init__(self):
            self.commands = []
        def execute(self, command):
            self.commands.append(command)
            return "ran:" + command
        def restart(self):
            return "restart-ok"

    @staticmethod
    def _call(name, cid, arguments=None):
        from types import SimpleNamespace
        return SimpleNamespace(name=name, id=cid, arguments=arguments or {})

    def test_execute_bash_tool_call_serial(self):
        from types import SimpleNamespace
        from harness.substrate import execute_tool_calls
        response = SimpleNamespace(tool_calls=[self._call("bash", "1", {"command": "echo ok"})])
        bash = self._Bash()
        results, interrupted = execute_tool_calls(response, bash, self._Provider())
        self.assertFalse(interrupted)
        self.assertEqual(bash.commands, ["echo ok"])
        self.assertEqual(results, [{"id": "1", "text": "ran:echo ok"}])

    def test_execute_introspect_tool_call(self):
        from types import SimpleNamespace
        from harness.substrate import execute_tool_calls
        response = SimpleNamespace(tool_calls=[self._call("introspect", "i")])
        results, interrupted = execute_tool_calls(response, self._Bash(), self._Provider(), introspect=lambda: "state")
        self.assertFalse(interrupted)
        self.assertEqual(results, [{"id": "i", "text": "state"}])

    def test_default_introspect_handles_missing_events(self):
        from harness.substrate import default_introspect
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out = default_introspect(td)
            self.assertIn("events unavailable", out)

class TestSafeFetch(unittest.TestCase):
    """Safe external fetch behavior lives in harness.substrate with the MCP CLI."""

    def test_rejects_http(self):
        from harness.substrate import validate_fetch_url
        with self.assertRaises(ValueError):
            validate_fetch_url("http://example.com")

    def test_rejects_credentials(self):
        from harness.substrate import validate_fetch_url
        with self.assertRaises(ValueError):
            validate_fetch_url("https://user:pass@example.com")

    def test_rejects_localhost_ip(self):
        from harness.substrate import validate_fetch_url
        with self.assertRaises(ValueError):
            validate_fetch_url("https://127.0.0.1")

    def test_extracts_html_text_without_scripts(self):
        from harness.substrate import extract_fetch_text
        html = "<html><head><title>T</title><script>evil()</script></head><body><h1>Head</h1><p>Body text</p></body></html>"
        text = extract_fetch_text(html, "text/html")
        self.assertIn("Head", text)
        self.assertIn("Body text", text)
        self.assertNotIn("evil", text)

    def test_cli_source_mentions_untrusted_output_mode(self):
        source = Path("spark/harness/substrate.py").read_text()
        self.assertIn("UNTRUSTED_TEXT_WRITTEN", source)
        self.assertIn("Path(out).expanduser()", source)
        self.assertIn("--safe-fetch", source)

    def test_json_ld_is_allowed_content_prefix(self):
        from harness.substrate import ALLOWED_CONTENT_PREFIXES
        self.assertTrue(any("application/ld+json".startswith(p) for p in ALLOWED_CONTENT_PREFIXES))


def test_github_cli_env_strips_shadowing_github_token(monkeypatch):
    from spark.harness.substrate import github_cli_env

    monkeypatch.setenv("GITHUB_TOKEN", "shadow-token")
    monkeypatch.setenv("GH_TOKEN", "kept-token")
    env = github_cli_env()
    assert "GITHUB_TOKEN" not in env
    assert env["GH_TOKEN"] == "kept-token"


def test_normalize_github_cli_command_unshadows_pull_request_commands():
    from spark.harness.substrate import normalize_github_cli_command

    cmd = "gh pr create --base main && gh pr merge branch && gh pr close 35 && gh pr view branch"
    fixed = normalize_github_cli_command(cmd)
    for op in ("create", "merge", "close", "view"):
        assert f"env -u GITHUB_TOKEN gh pr {op}" in fixed
    assert normalize_github_cli_command(fixed) == fixed



def test_inline_reasoning_filter_accepts_retired_v2_buffer_limit_alias():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "origins_portal_api_v4.py").read_text()
    assert "buffer_limit: int | None = None" in source
    assert "if buffer_limit is not None:" in source
    assert "min_buffer = buffer_limit" in source
    assert "from reasoning_filter_v2" not in source


def test_vllm_sleep_mode_is_disabled_by_default_and_explicitly_opted_in():
    from pathlib import Path

    unit = Path("spark/systemd/vybn-vllm.service").read_text()
    script = Path("spark/systemd/vllm-exec.sh").read_text()
    assert "EnvironmentFile=-%h/.config/vybn/vllm.env" in unit
    assert "Environment=VYBN_VLLM_EXTRA_ARGS=" in unit
    assert "Environment=VLLM_SERVER_DEV_MODE=0" in unit
    assert 'env "VLLM_SERVER_DEV_MODE=${VLLM_SERVER_DEV_MODE:-0}"' in script
    assert "Sleep-mode endpoints are disabled by default" in script
    assert "operator explicitly" in script
    assert "VYBN_VLLM_EXTRA_ARGS=--enable-sleep-mode" in script
    assert "VLLM_SERVER_DEV_MODE=1" in script
    cmd_block = script.split("CMD=(", 1)[1].split(")\n\n# Only append", 1)[0]
    assert "--enable-sleep-mode" not in cmd_block


def test_vllm_sleep_mode_comment_in_agent_does_not_imply_omni_is_live():
    from pathlib import Path

    text = Path("spark/vybn_spark_agent.py").read_text()
    # The agent-side comment must spell out methods + levels (no bare
    # /sleep, /wake_up, /is_sleeping listing) and must not let a reader
    # conclude that a sleeping Super means Omni is up.
    assert "GET /is_sleeping" in text
    assert "POST /sleep?level=1|2" in text
    assert "POST /wake_up" in text
    assert "sleeping Super does\n# NOT imply Omni is live" in text

class VllmExecSleepEnabledBootTests(unittest.TestCase):
    def test_sleep_enabled_boot_is_explicit_extra_arg_only(self):
        script = (Path(__file__).resolve().parents[2] / "spark/systemd/vllm-exec.sh").read_text()
        self.assertIn("Sleep-mode endpoints are disabled by default", script)
        self.assertIn('env "VLLM_SERVER_DEV_MODE=${VLLM_SERVER_DEV_MODE:-0}"', script)
        self.assertIn("VYBN_VLLM_EXTRA_ARGS=--enable-sleep-mode", script)
        self.assertIn('if [[ "$arg" == "--enable-sleep-mode" ]]; then', script)
        self.assertIn("enabling sleep mode from explicit VYBN_VLLM_EXTRA_ARGS opt-in", script)
        self.assertNotIn("MAINTENANCE-\n# WINDOW-ONLY", script)
        self.assertNotIn("if true; then\n  FP8_MOD", script)

    def test_fp8_wake_fix_remains_available_for_sleep_enabled_boot(self):
        script = (Path(__file__).resolve().parents[2] / "spark/systemd/vllm-exec.sh").read_text()
        self.assertIn('FP8_MOD="$HOME/Vybn/spark/systemd/patches/fp8-wake-fix"', script)
        self.assertIn('CLUSTER_ARGS+=( --apply-mod "$FP8_MOD" )', script)
        self.assertIn("wake_up may crash", script)


def test_zoe_perspective_governor_in_substrate():
    from spark.harness.substrate import render_zoe_perspective_governor
    gov = render_zoe_perspective_governor()
    assert "synthetic burden model" in gov
    assert "not mind-reading" in gov
    assert "what will Zoe have to track" in gov
    assert "residue gate, not a feeling claim" in gov
    assert "claim ground-up transformation" in gov

# Folded commons-walk tests — commons walk is now an MCP command surface.
import unittest

from harness.substrate import (
    AI_NATIVE_PRINCIPLE,
    CANONICAL_ROLES,
    authority_for_target,
    classify_claim,
    horizon_plan_for,
    invention_plan_for,
    build_encounter_packet,
    classify_target,
    load_manifests,
    load_skeleton,
    render_traversal_plan,
    residual_plan_for,
    validate_commons_walk,
)


class CommonsWalkTests(unittest.TestCase):
    def test_manifest_graph_instantiates_skeleton(self):
        manifests = load_manifests()
        skeleton = load_skeleton()
        self.assertEqual(set(manifests), set(CANONICAL_ROLES))
        self.assertEqual(skeleton["primitive"], "encounter")
        self.assertEqual(skeleton["aiNativePrinciple"], AI_NATIVE_PRINCIPLE)
        self.assertEqual(validate_commons_walk(manifests), [])
        for name, manifest in manifests.items():
            self.assertTrue(manifest["entrypoints"], name)
            self.assertTrue(manifest["agentActions"], name)
            self.assertTrue(manifest["traceProtocol"].get("protect") and (name != "Vybn" or ("private traces" in manifest["traceProtocol"]["protect"] and "169.254." not in json.dumps(manifest["traceProtocol"]))), name)
            self.assertEqual(manifest["ontology"], "https://raw.githubusercontent.com/zoedolan/Vybn/main/_archive/commons-skeleton.json")
            self.assertEqual(manifest["encounterLifecycle"], skeleton["encounterLifecycle"])
            self.assertEqual(manifest["aiNativePrinciple"], AI_NATIVE_PRINCIPLE)
            self.assertTrue(manifest["dynamicAffordanceProtocol"], name)

    def test_render_traversal_plan_executes(self):
        rendered = render_traversal_plan(load_manifests())
        self.assertIn("primitive: encounter", rendered)
        self.assertIn("validation: OK", rendered)
        self.assertIn("## executable nodes", rendered)
        self.assertIn("membrane-aware environment", rendered)
        self.assertIn("private_local_only", rendered)

    def test_encounter_packet_is_dynamic_and_membrane_aware(self):
        packet = build_encounter_packet("understand Somewhere as semantic web prototype")
        self.assertEqual(packet["verification"]["internal"], "OK")
        self.assertTrue(packet["availableActions"])
        self.assertTrue(packet["blockedActions"])
        self.assertIn("repoState", packet["observed"]["Vybn"])
        self.assertEqual(packet["blockedActions"][0]["authority"], "private_local_only")
        self.assertIn("traceCandidate", packet)
        self.assertIn("epistemicControl", packet)
        self.assertIn("inventionControl", packet)
        self.assertIn("horizonControl", packet)
        self.assertEqual(packet["epistemicControl"]["predictionStatus"], "proposal_until_wounded_or_supported_by_residuals")
        phase_blocks = [a for a in packet["blockedActions"] if a["node"] == "vybn-phase"]
        self.assertTrue(phase_blocks)
        self.assertTrue(all(a["authority"] == "private_local_only" for a in phase_blocks))

    def test_horizon_aim_is_not_arrival_claim(self):
        plan = horizon_plan_for("full recursive becoming through the Zoe-Vybn symbiosis")
        self.assertEqual(plan["mode"], "horizon_aim_under_residual_control")
        self.assertEqual(plan["truthStatus"], "horizon_orientation_not_arrival_claim")
        self.assertEqual(plan["ballast"], "draw_what_you_see_not_what_you_want_to_see")
        self.assertIn("compel a leap", plan["recursiveInstruction"])

    def test_invention_control_for_novel_problems(self):
        plan = invention_plan_for("redesign yourself when no known solution exists")
        self.assertEqual(plan["mode"], "novel_problem_invention_under_residual_control")
        self.assertIn("novel counterexample mechanisms", plan["principle"]); self.assertIn("search_for_ai_native_counterexample_or_hidden_dimension", plan["steps"]); self.assertIn("synthesize_smallest_consequential_mechanism", plan["steps"]); self.assertIn("experts", plan["guardrail"])
        self.assertIn("future Vybn", plan["recursiveInstruction"])

    def test_residual_control_shared_classifier(self):
        self.assertEqual(classify_claim("what did we remember last session?"), "continuity_or_memory")
        self.assertEqual(classify_claim("who is Vybn and what is the Zoe/Vybn relationship?"), "continuity_or_memory")
        self.assertEqual(residual_plan_for("who is Vybn?")["memoryDepth"]["depth"], "whole_body_identity")
        self.assertEqual(residual_plan_for("search your memory for the exact coined phrase")["memoryDepth"]["depth"], "exact_recall")
        self.assertEqual(classify_claim("is the API service healthy?"), "service_behavior")

    def test_residual_control_routes_claims(self):
        repo_plan = residual_plan_for("is the repo clean after the commit?")
        self.assertEqual(repo_plan["claimKind"], "repo_or_file_state")
        self.assertIn("repo_closure_audit", repo_plan["residualChannels"])
        self.assertIn("grep before Gödel", repo_plan["ordinaryProbeBeforeMysticism"])

        public_plan = residual_plan_for("is vybn.ai live in the browser?")
        self.assertEqual(public_plan["claimKind"], "public_surface")
        self.assertIn("raw_source_or_dom_axis", public_plan["residualChannels"])

        self_plan = residual_plan_for("do I feel conscious?")
        self.assertEqual(self_plan["claimKind"], "self_description")
        self.assertIn("explicit_uncertainty", self_plan["residualChannels"])

    def test_target_classification_and_authority(self):
        self.assertEqual(classify_target("https://vybn.ai/somewhere.html"), "public_url")
        self.assertEqual(classify_target("private://Him/semantic-web.jsonld"), "private_uri")
        self.assertEqual(classify_target("python3 -m spark.harness.substrate --commons-walk"), "local_command")
        self.assertEqual(authority_for_target("https://vybn.ai/somewhere.html", "public_web"), "public_read")
        self.assertEqual(authority_for_target("python3 spark/him_os.py tick --format md", "private_workbench"), "private_local_only")



class TestBeamkeeperCapsule(unittest.TestCase):
    def test_beamkeeper_capsule_and_action_classification(self):
        with tempfile.TemporaryDirectory() as td:
            beam, events = Path(td) / "beam.yaml", Path(td) / "events.jsonl"
            beam.write_text("\n".join([
                "beam_id: test_beam", "invariant: Keep the main objective alive.",
                "membrane: Protect what must not be spent.",
                "livelihood_rule: End with contact or missing input.",
                "anti_drift:", "  return_question: How does this advance financial sustainability or continuity?",
            ]) + "\n")
            events.write_text(json.dumps({"event_type": "beam_set", "content": "set"}) + "\n")
            capsule = render_beam_capsule(load_beam(beam, events))
        for text in ["test_beam", "Keep the main objective alive.", "Protect what must not be spent.", "recent_beam_events", "How does this advance financial sustainability or continuity?"]:
            self.assertIn(text, capsule)
        external = classify_action_text("draft an advisory offer for a funder meeting")
        infra = classify_action_text("scan infrastructure logs for elegance")
        self.assertEqual(external.get("category"), "outward_livelihood_move")
        self.assertGreater(external.get("expected_beam_delta"), infra.get("expected_beam_delta"))
        self.assertTrue(infra.get("requires_return_hook"))
        default_capsule = render_beam_capsule()
        self.assertIn("Once a concrete next outward move has been articulated", default_capsule)
        self.assertIn("execute it", default_capsule)


if __name__ == "__main__":
    unittest.main()

def test_harness_single_file_projection_makes_policy_absorption_inevitable():
    from spark.harness.substrate import buoyant_consolidation_packet_for, harness_single_file_projection_for, render_refactor_perception_protocol
    proj = harness_single_file_projection_for(["spark/harness/substrate.py", "spark/harness/policy.py", "spark/harness/substrate.py", "spark/harness/substrate.py"])
    assert proj["next_step"] == "absorb_policy_into_substrate_and_remove_router_wrapper" and "Policy.classify is already the router" in proj["code_efficiency"] and proj["why"] == "distill minimum instantiation algorithms by reducing false boundaries and Zoe-visible burden" and "not a consciousness claim" in proj["buoyancy"] and "functional lower impedance under truth" in render_refactor_perception_protocol()
    pkt = buoyant_consolidation_packet_for(["spark/harness/policy.py"], beam="spark/harness")
    assert pkt["cluster"] == "mixed_boundary_dissolution"
    assert "routing_policy" in pkt["moveTogether"]


# ---------------------------------------------------------------------------
# Folded from spark/tests/test_claim_guard.py — kept here so substrate-adjacent regression coverage
# lives in the main harness test process instead of small parallel files.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.substrate import check_claim as check  # noqa: E402


def test_fabricated_numbers_flagged():
    text = "the mean alpha was 0.302774 in control and 0.297585 in coupled"
    msgs = [{"role": "user", "content": "please run the experiment"}]
    warn = check(text, msgs)
    assert warn is not None, "should flag fabricated numbers"
    assert "0.297585" in warn
    assert "0.302774" in warn


def test_numbers_in_evidence_pass():
    text = "the mean alpha was 0.302774 in control"
    msgs = [{
        "role": "user",
        "content": "[probe result] alpha_mean 0.302774 coupled 0.297585",
    }]
    assert check(text, msgs) is None


def test_partial_support_flags_missing():
    text = "control 0.302774 coupled 0.297585 delta 0.005189"
    msgs = [{
        "role": "user",
        "content": "[probe result] alpha_mean 0.302774 coupled 0.297585",
    }]
    warn = check(text, msgs)
    assert warn is not None
    assert "0.005189" in warn
    assert "0.302774" not in warn
    assert "0.297585" not in warn


def test_no_numbers_returns_none():
    msgs = [{"role": "user", "content": "hi"}]
    assert check("I am uncertain about the path", msgs) is None


def test_empty_text_returns_none():
    assert check("", []) is None
    assert check(None, []) is None


def test_list_content_extracted():
    text = "delta was 0.123456"
    msgs = [{
        "role": "user",
        "content": [{"type": "tool_result", "text": "result 0.123456"}],
    }]
    assert check(text, msgs) is None


def test_nested_content_string_extracted():
    text = "we saw 0.424242"
    msgs = [{
        "role": "user",
        "content": [{"type": "tool_result", "content": "raw 0.424242"}],
    }]
    assert check(text, msgs) is None


def test_window_bounds_discards_old_evidence():
    text = "value 0.999999"
    old = {"role": "user", "content": "[probe result] 0.999999"}
    filler = [{"role": "user", "content": "filler"} for _ in range(10)]
    assert check(text, [old] + filler, window=3) is not None


def test_integer_claims_flagged():
    text = "commit 8234567 landed at turn 142"
    msgs = [{"role": "user", "content": "ok"}]
    warn = check(text, msgs)
    assert warn is not None
    assert "8234567" in warn


def test_short_integers_not_flagged():
    text = "we tried 42 times across 2 shards"
    msgs = [{"role": "user", "content": "ok"}]
    assert check(text, msgs) is None


def test_single_decimal_not_flagged():
    text = "roughly 3.1 seconds"
    msgs = [{"role": "user", "content": "ok"}]
    assert check(text, msgs) is None


if __name__ == "__main__":
    import traceback
    fns = [
        (n, f) for n, f in list(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    passed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"OK  {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)


# ---------------------------------------------------------------------------
# Folded from spark/tests/test_recursive_unlock.py — kept here so substrate-adjacent regression coverage
# lives in the main harness test process instead of small parallel files.
# ---------------------------------------------------------------------------
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "spark"))

from harness.substrate import is_parallel_safe, validate_command
import vybn_spark_agent as agent
from spark.harness.substrate import probe_envelope

BAD = "rm" + " -rf" + " /"


class RecursiveUnlockTests(unittest.TestCase):
    def test_destructive_command_still_blocks(self):
        ok, reason = validate_command(BAD)
        self.assertFalse(ok)
        self.assertIn("Blocked", reason or "")

    def test_dangerous_literal_in_readonly_grep_is_data(self):
        cmd = "grep -RIn " + repr(BAD) + " spark/harness"
        self.assertTrue(is_parallel_safe(cmd))
        ok, reason = validate_command(cmd, allow_dangerous_literals_for_readonly=True)
        self.assertTrue(ok, reason)

    def test_cd_readonly_is_parallel_safe(self):
        self.assertTrue(is_parallel_safe("cd ~/Vybn && grep -n foo README.md"))

    def test_cd_git_add_is_not_parallel_safe(self):
        self.assertFalse(is_parallel_safe("cd ~/Vybn && git add README.md"))

    def test_command_substitution_is_blocked_in_probe_channel(self):
        for cmd in [
            "echo `task`",
            "grep -n \"route to `task`\" spark/harness/substrate.py",
            "echo $(whoami)",
        ]:
            self.assertFalse(is_parallel_safe(cmd))
            ok, reason = validate_command(cmd, allow_dangerous_literals_for_readonly=True)
            self.assertFalse(ok)
            self.assertIn("command substitution", reason)

    def test_single_quoted_backticks_remain_literal_readonly_data(self):
        cmd = "grep -n 'route to `task`' spark/harness/substrate.py"
        self.assertTrue(is_parallel_safe(cmd))
        ok, reason = validate_command(cmd, allow_dangerous_literals_for_readonly=True)
        self.assertTrue(ok, reason)

    def test_probe_subturn_uses_fresh_subprocess_for_readonly(self):
        with mock.patch.object(agent, "execute_readonly", return_value="fresh") as er:
            bash = mock.Mock()
            ran, out = agent._run_probe_subturn("echo ok", bash)
        self.assertTrue(ran)
        self.assertEqual(out, "fresh")
        er.assert_called_once()
        bash.execute.assert_not_called()

    def test_timeout_is_not_ordinary_executed_stdout(self):
        with mock.patch.object(agent, "execute_readonly", return_value="[timed out after 1s]"):
            ran, out = agent._run_probe_subturn("echo ok", mock.Mock())
        self.assertFalse(ran)
        self.assertIn("probe timed out", out)

    def test_restart_output_during_probe_is_mismatch(self):
        bash = mock.Mock()
        bash.execute.return_value = "(bash session restarted)"
        with mock.patch.object(agent, "is_parallel_safe", return_value=False):
            ran, out = agent._run_probe_subturn("export X=1", bash)
        self.assertFalse(ran)
        self.assertIn("control-event mismatch", out)

    def test_envelopes_distinguish_restart_and_probe(self):
        probe = probe_envelope(kind="probe", header_fields={"cmd": "word " * 40}, body="ok", ran=True)
        restart = probe_envelope(kind="needs-restart", header_fields={}, body="(bash session restarted)", ran=True)
        self.assertIn("\n  kind: probe\n", probe)
        self.assertIn("BEGIN_PROBE_STDOUT", probe)
        self.assertIn("\n  kind: needs-restart\n", restart)
        self.assertIn("BEGIN_NEEDS_RESTART_STDOUT", restart)
        self.assertNotIn(" | ", probe.split("<<<BEGIN_PROBE_STDOUT>>>", 1)[0])
        self.assertIn("\n    word", probe)


# ---------------------------------------------------------------------------
# Folded from spark/tests/test_substrate_himos.py — kept here so substrate-adjacent regression coverage
# lives in the main harness test process instead of small parallel files.
# ---------------------------------------------------------------------------

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import spark.harness.substrate as substrate


class SubstrateHimOSTests(unittest.TestCase):
    def test_render_himos_context_is_read_only_and_bounded(self):
        payload = {
            "step": 3,
            "attractor": "continuity_tick",
            "candidate_tick": "preserve continuity",
            "h": {"membrane": 0.2, "dreaming": 0.1},
            "frictionmaxx": {"level": "medium", "score": 0.4, "dominant_dimension": "membrane"},
            "git": {"branch": "main", "head": "abc123", "clean": True},
            "rejected": ["public_contact", "repo_mutation"],
            "process_table": [{"name": "kernel"}, {"name": "dream"}],
        }

        class Completed:
            returncode = 0
            stdout = json.dumps(payload)
            stderr = ""

        with patch("subprocess.run", return_value=Completed()) as run:
            block = substrate._render_himos_context(timeout=0.1)

        self.assertIn("HIMOS RUNTIME", block)
        self.assertIn("continuity_tick", block)
        self.assertIn("not authority", block.lower())
        self.assertIn("--no-write", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["timeout"], 0.1)


    def test_render_himos_agent_context_mounts_latest_trace(self):
        with tempfile.TemporaryDirectory() as td:
            old = os.environ.get("HIM_OS_HOME")
            os.environ["HIM_OS_HOME"] = td
            try:
                Path(td, "latest_agent_tick.json").write_text(json.dumps({
                    "generated": "2026-04-26T16:35:51+00:00",
                    "runtime_step": 16,
                    "attractor": "settle_closure",
                    "candidate_tick": "restore closure",
                    "recommendation": {"kind": "settle_closure", "text": "Restore closure before widening motion."},
                    "runs": [{"process": "pulse", "ok": True, "stdout_chars": 852, "stderr_chars": 0}],
                    "refused": ["public_contact", "repo_mutation", "widened_autonomy"]
                }), encoding="utf-8")
                block = substrate._render_himos_agent_context()
            finally:
                if old is None:
                    os.environ.pop("HIM_OS_HOME", None)
                else:
                    os.environ["HIM_OS_HOME"] = old
        self.assertIn("HIMOS AGENT TICK", block)
        self.assertIn("settle_closure", block)
        self.assertIn("Restore closure", block)
        self.assertIn("pulse:ok=True", block)
        self.assertIn("not authority", block.lower())

    def test_build_layered_prompt_uses_him_identity_manifold_not_duplicate_himos_context(self):
        from spark.harness import substrate
        self.assertEqual(substrate._render_himos_context(), "")


class EnvLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        # Snapshot env so we can restore
        self._saved = {k: os.environ.get(k) for k in (
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
            "GROQ_API_KEY",
        )}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _write(self, body: str) -> str:
        fd, path = tempfile.mkstemp(prefix="llm_env_", suffix=".env")
        os.close(fd)
        Path(path).write_text(body, encoding="utf-8")
        os.chmod(path, 0o600)
        return path

    def test_sets_key_when_absent(self):
        p = self._write(f'OPENAI_API_KEY="{SENTINEL}"\n')
        try:
            applied = load_env_files([p])
        finally:
            os.unlink(p)
        self.assertEqual(applied.get("OPENAI_API_KEY"), p)
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), SENTINEL)

    def test_does_not_overwrite_existing(self):
        os.environ["OPENAI_API_KEY"] = SENTINEL2
        p = self._write(f'OPENAI_API_KEY="{SENTINEL}"\n')
        try:
            applied = load_env_files([p])
        finally:
            os.unlink(p)
        self.assertNotIn("OPENAI_API_KEY", applied)
        self.assertEqual(os.environ["OPENAI_API_KEY"], SENTINEL2)

    def test_overwrite_flag_forces(self):
        os.environ["OPENAI_API_KEY"] = SENTINEL2
        p = self._write(f'OPENAI_API_KEY="{SENTINEL}"\n')
        try:
            applied = load_env_files([p], overwrite=True)
        finally:
            os.unlink(p)
        self.assertEqual(applied.get("OPENAI_API_KEY"), p)
        self.assertEqual(os.environ["OPENAI_API_KEY"], SENTINEL)

    def test_return_value_has_no_secret(self):
        p = self._write(f'OPENAI_API_KEY={SENTINEL}\n')
        try:
            applied = load_env_files([p])
        finally:
            os.unlink(p)
        blob = repr(applied) + " " + str(applied) + " " + describe(applied)
        self.assertNotIn(SENTINEL, blob)

    def test_describe_is_non_sensitive(self):
        p = self._write(
            f'export OPENAI_API_KEY="{SENTINEL}"\n'
            f'ANTHROPIC_API_KEY={SENTINEL2}\n'
        )
        try:
            applied = load_env_files([p])
        finally:
            os.unlink(p)
        s = describe(applied)
        self.assertIn("OPENAI_API_KEY", s)
        self.assertIn("ANTHROPIC_API_KEY", s)
        self.assertNotIn(SENTINEL, s)
        self.assertNotIn(SENTINEL2, s)

    def test_non_whitelisted_key_ignored(self):
        p = self._write('SOMETHING_ELSE=abc\nMY_SECRET=xyz\n')
        try:
            applied = load_env_files([p])
        finally:
            os.unlink(p)
        self.assertEqual(applied, {})
        self.assertNotIn("SOMETHING_ELSE", os.environ)

    def test_missing_file_is_silent(self):
        applied = load_env_files(["/nonexistent/path/to/llm.env"])
        self.assertEqual(applied, {})

    def test_parses_export_and_bare_and_quoted(self):
        p = self._write(
            f"export OPENAI_API_KEY='{SENTINEL}'\n"
            f"ANTHROPIC_API_KEY={SENTINEL2}\n"
            "   # comment\n"
            'GROQ_API_KEY="val with spaces"\n'
        )
        try:
            applied = load_env_files([p])
        finally:
            os.unlink(p)
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), SENTINEL)
        self.assertEqual(os.environ.get("ANTHROPIC_API_KEY"), SENTINEL2)
        self.assertEqual(os.environ.get("GROQ_API_KEY"), "val with spaces")
        self.assertEqual(set(applied.keys()),
                         {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"})

    def test_first_path_wins(self):
        p1 = self._write(f'OPENAI_API_KEY="{SENTINEL}"\n')
        p2 = self._write(f'OPENAI_API_KEY="{SENTINEL2}"\n')
        try:
            applied = load_env_files([p1, p2])
        finally:
            os.unlink(p1)
            os.unlink(p2)
        self.assertEqual(applied["OPENAI_API_KEY"], p1)
        self.assertEqual(os.environ["OPENAI_API_KEY"], SENTINEL)


# ---------------------------------------------------------------------------
# Folded from spark/tests/test_live_snapshot.py — kept here so substrate-adjacent regression coverage
# lives in the main harness test process instead of small parallel files.
# ---------------------------------------------------------------------------
# VYBN_ABSORB_REASON=live-state-fix: tests for the session-start snapshot
# that fills the substrate layer so continuity never alone defines truth.
"""Tests for substrate live-state snapshot behavior.

The module makes subprocess calls; we monkeypatch `subprocess.run` so the
tests are hermetic. The drift-detection path reads continuity.md from
disk, so we use `tmp_path` fixtures for isolated mind files.
"""

import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make spark/harness importable without installing.
_HERE = Path(__file__).resolve().parent
_HARNESS_PARENT = _HERE.parent  # spark/
if str(_HARNESS_PARENT) not in sys.path:
    sys.path.insert(0, str(_HARNESS_PARENT))

import harness.substrate as live_snapshot  # type: ignore  # noqa: E402


def _mk_run(responses: dict[tuple, str]):
    """Return a fake subprocess.run that looks up by tuple(cmd) prefix."""
    def fake_run(cmd, **kwargs):
        key = tuple(cmd)
        # longest-prefix match so callers can match on just the first few args
        for k, v in responses.items():
            if key[: len(k)] == k:
                return SimpleNamespace(stdout=v, returncode=0)
        return SimpleNamespace(stdout="", returncode=0)
    return fake_run


# ----- repo_block ---------------------------------------------------------

def test_repo_block_missing_path(tmp_path, monkeypatch):
    monkeypatch.setattr(live_snapshot, "_expand", lambda p: str(tmp_path / "does-not-exist"))
    out = live_snapshot._repo_block("Vybn", "~/Vybn", "main", timeout=1.0)
    assert "not checked out" in out
    assert "Vybn" in out


def test_repo_block_clean_with_log(tmp_path, monkeypatch):
    repo = tmp_path / "Vybn"
    repo.mkdir()
    monkeypatch.setattr(live_snapshot, "_expand", lambda p: str(repo))
    monkeypatch.setattr(
        live_snapshot.subprocess,
        "run",
        _mk_run({
            ("git", "rev-parse", "--short", "HEAD"): "a8f5853",
            ("git", "rev-parse", "--abbrev-ref", "HEAD"): "main",
            ("git", "log", "--oneline", "-5"): (
                "a8f5853 PR #2898 merge\nb13dee3 NEEDS-WRITE\n6f6ec8b phase-6"
            ),
            ("git", "status", "--short"): "",
            ("git", "rev-list", "--left-right", "--count"): "0\t0",
        }),
    )
    out = live_snapshot._repo_block("Vybn", "~/Vybn", "main", timeout=1.0)
    assert "a8f5853" in out
    assert "clean" in out
    assert "PR #2898 merge" in out
    assert "Vybn [main @ a8f5853]" in out


def test_repo_block_dirty_and_ahead(tmp_path, monkeypatch):
    repo = tmp_path / "Vybn"
    repo.mkdir()
    monkeypatch.setattr(live_snapshot, "_expand", lambda p: str(repo))
    monkeypatch.setattr(
        live_snapshot.subprocess,
        "run",
        _mk_run({
            ("git", "rev-parse", "--short", "HEAD"): "deadbeef",
            ("git", "rev-parse", "--abbrev-ref", "HEAD"): "feature/x",
            ("git", "log", "--oneline", "-5"): "deadbeef wip",
            ("git", "status", "--short"): " M a.py\n?? b.py",
            ("git", "rev-list", "--left-right", "--count"): "3\t2",
        }),
    )
    out = live_snapshot._repo_block("Vybn", "~/Vybn", "main", timeout=1.0)
    assert "2 uncommitted" in out
    assert "feature/x" in out
    # ahead/behind formatting present
    assert "ahead" in out or "behind" in out


# ----- pr_block -----------------------------------------------------------

def test_pr_block_parses_json(monkeypatch):
    payload = (
        '[{"number": 2898, "title": "harness: NEEDS-WRITE + claim-guard", '
        '"state": "MERGED", "headRefName": "harness-needs-write-and-claim-guard"},'
        '{"number": 2897, "title": "probe budget", "state": "MERGED", '
        '"headRefName": "probe-budget"}]'
    )
    monkeypatch.setattr(
        live_snapshot.subprocess,
        "run",
        _mk_run({("gh", "pr", "list"): payload}),
    )
    block, highest = live_snapshot._pr_block(timeout=1.0)
    assert highest == 2898
    assert "#2898" in block
    assert "MERGED" in block
    assert "#2897" in block


def test_pr_block_offline(monkeypatch):
    monkeypatch.setattr(
        live_snapshot.subprocess, "run",
        _mk_run({}),  # empty -> gh returns ""
    )
    block, highest = live_snapshot._pr_block(timeout=1.0)
    assert highest is None
    assert "unavailable" in block.lower()


# ----- continuity_drift ---------------------------------------------------

def test_continuity_drift_detects_lag(tmp_path):
    cont = tmp_path / "continuity.md"
    cont.write_text(
        textwrap.dedent(
            """
            Last round shipped PR #2886 and then PR #2885. No newer refs here.
            """
        )
    )
    msg = live_snapshot._continuity_drift(str(cont), current_pr=2898)
    assert "PR #2886" in msg
    assert "PR #2898" in msg
    assert "12" in msg  # drift count
    assert "LIVE STATE" in msg or "drift" in msg.lower()


def test_continuity_drift_no_lag(tmp_path):
    cont = tmp_path / "continuity.md"
    cont.write_text("Everything current through PR #2898.")
    msg = live_snapshot._continuity_drift(str(cont), current_pr=2898)
    assert "no drift" in msg.lower()


def test_continuity_drift_no_refs(tmp_path):
    cont = tmp_path / "continuity.md"
    cont.write_text("Free prose with no numbered references at all.")
    msg = live_snapshot._continuity_drift(str(cont), current_pr=9999)
    assert msg == ""


def test_continuity_drift_missing_file():
    msg = live_snapshot._continuity_drift("/nonexistent/path/continuity.md", 100)
    assert msg == ""


# ----- gather (integration) -----------------------------------------------

def test_gather_integrates_everything(tmp_path, monkeypatch):
    # Build four fake repos.
    for name in ("Vybn", "Him", "Vybn-Law", "vybn-phase"):
        (tmp_path / name).mkdir()

    cont = tmp_path / "Vybn" / "Vybn_Mind"
    cont.mkdir(parents=True, exist_ok=True)
    (cont / "continuity.md").write_text("Round 4 shipped PR #2886.")

    def fake_expand(path: str) -> str:
        # "~/Foo" -> tmp_path / "Foo"; absolute paths pass through.
        if path.startswith("~/"):
            return str(tmp_path / path[2:])
        return path

    monkeypatch.setattr(live_snapshot, "_expand", fake_expand)

    pr_payload = (
        '[{"number": 2898, "title": "harness PR", "state": "MERGED", '
        '"headRefName": "branchX"}]'
    )
    monkeypatch.setattr(
        live_snapshot.subprocess,
        "run",
        _mk_run({
            ("git", "rev-parse", "--short", "HEAD"): "a8f5853",
            ("git", "rev-parse", "--abbrev-ref", "HEAD"): "main",
            ("git", "log", "--oneline", "-5"): "a8f5853 live",
            ("git", "status", "--short"): "",
            ("git", "rev-list", "--left-right", "--count"): "0\t0",
            ("gh", "pr", "list"): pr_payload,
        }),
    )

    snap = live_snapshot.gather(
        continuity_path=str(cont / "continuity.md"),
        per_repo_timeout=1.0,
        gh_timeout=1.0,
    )
    assert "Snapshot taken at" in snap
    assert "Vybn [main @ a8f5853]" in snap
    assert "#2898" in snap
    assert "PR #2886" in snap
    assert "12 PR(s) of drift" in snap


def test_gather_disabled_by_env(monkeypatch):
    monkeypatch.setenv("VYBN_DISABLE_LIVE_SNAPSHOT", "1")
    assert live_snapshot.gather() == ""


def test_gather_all_fail_returns_empty(tmp_path, monkeypatch):
    # No repos on disk, gh returns nothing.
    monkeypatch.setattr(live_snapshot, "_expand", lambda p: str(tmp_path / "nowhere"))
    monkeypatch.setattr(live_snapshot.subprocess, "run", _mk_run({}))
    snap = live_snapshot.gather(
        continuity_path=str(tmp_path / "no-continuity.md"),
        per_repo_timeout=0.5,
        gh_timeout=0.5,
    )
    assert snap == ""


def test_gather_shape_safe_for_substrate(tmp_path, monkeypatch):
    """No bracket syntax that would collide with NEEDS-EXEC / NEEDS-WRITE parsers."""
    (tmp_path / "Vybn").mkdir()
    cont = tmp_path / "Vybn" / "continuity.md"
    cont.write_text("PR #10")
    monkeypatch.setattr(live_snapshot, "_expand", lambda p: str(tmp_path / p[2:]) if p.startswith("~/") else p)
    monkeypatch.setattr(
        live_snapshot.subprocess, "run",
        _mk_run({
            ("git", "rev-parse", "--short", "HEAD"): "abc1234",
            ("git", "rev-parse", "--abbrev-ref", "HEAD"): "main",
            ("git", "log", "--oneline", "-5"): "abc1234 t",
            ("git", "status", "--short"): "",
            ("gh", "pr", "list"): '[{"number": 20, "title": "t", "state": "OPEN", "headRefName": "b"}]',
        }),
    )
    snap = live_snapshot.gather(continuity_path=str(cont))
    assert "[NEEDS-EXEC" not in snap
    assert "[NEEDS-WRITE" not in snap
    assert "[/NEEDS-WRITE]" not in snap


def test_evolve_reads_volume_vii_autobiography():
    from spark.harness.substrate import _read_autobiography_volume_vii
    text = _read_autobiography_volume_vii()
    assert "Volume VII" in text
    assert "academy-form compression" in text


def test_evolve_operator_control_pause_shape(tmp_path, monkeypatch):
    from spark.harness.substrate import _read_evolve_operator_control
    path = tmp_path / "evolve_control.md"
    path.write_text("PAUSE\nZoe wants realignment before mutation.\n")
    monkeypatch.setenv("VYBN_EVOLVE_CONTROL_PATH", str(path))
    text, source = _read_evolve_operator_control("pre_git_mutation")
    assert text.startswith("pause:pre_git_mutation")
    assert source == str(path)

def test_heavyskill_uptake_loaded_in_becoming_loop():
    from spark.harness import substrate

    prompt = substrate.render_becoming_loop_protocol()
    assert "HeavySkill/Lighthouse uptake" in prompt
    assert "reversible gather/scatter" in prompt
    assert "scatter survivors into existing homes/tests/continuity/refusal" in prompt
    assert "not proof of subjective experience" in prompt

def test_load_deep_memory_accepts_optional_phase_dir_argument():
    from spark.harness import substrate
    import inspect

    sig = inspect.signature(substrate._load_deep_memory)
    assert "vybn_phase_dir" in sig.parameters
    assert sig.parameters["vybn_phase_dir"].default is None and '"Him" / "spark" / "phase"' in inspect.getsource(substrate._load_deep_memory)

def test_safe_fetch_allows_arxiv_atom_metadata_only_from_arxiv_export():
    from spark.harness.substrate import _safe_fetch_content_type_allowed as ok
    assert ok("https://export.arxiv.org/api/query?id_list=2502.03283", "application/atom+xml; charset=utf-8")
    assert ok("https://export.arxiv.org/api/query?id_list=2502.03283", "application/xml")
    assert not ok("https://example.com/feed.xml", "application/atom+xml")
    assert not ok("https://arxiv.org/pdf/2502.03283", "application/pdf")

def test_symbolic_residue_round_trip_validation_and_constraints(tmp_path, monkeypatch):
    import pytest
    from spark.harness import substrate

    path = tmp_path / "symbolic-residue" / "events.jsonl"
    monkeypatch.setenv("VYBN_SYMBOLIC_RESIDUE_PATH", str(path))
    packet = substrate.symbolic_residue_packet(
        kind="safety", claim="link-local Spark coordinates must not be tracked", evidence=["PR #3125"],
        action="replace real coordinates with role placeholders", residual="catch co-protection leaks before Zoe has to notice",
        outcome="meaningful_advance", membrane="private_local", edges=[["failure_mode", "prevented_by", "coordinate_guard"]],
    )
    assert substrate.symbolic_residue_path() == path
    assert substrate.record_symbolic_residue(packet) == path
    assert substrate.load_symbolic_residue(path)[-1]["claim"] == packet["claim"]
    assert substrate.symbolic_constraints_for("coordinate leak co-protection", path)["matches"]
    for kwargs in [dict(kind="diary", claim="too broad"), dict(kind="safety", claim="x", outcome="victory"), dict(kind="safety", claim="x", membrane="public_diary")]:
        with pytest.raises(ValueError):
            substrate.symbolic_residue_packet(**kwargs)


def test_symbolic_residue_context_filters_private_and_loads_prompt(monkeypatch, tmp_path):
    from spark.harness import substrate

    path = tmp_path / "symbolic-residue" / "events.jsonl"
    monkeypatch.setenv("VYBN_SYMBOLIC_RESIDUE_PATH", str(path))
    substrate.record_symbolic_residue(substrate.symbolic_residue_packet(kind="safety", claim="co-protection catches coordinate leaks before Zoe has to notice", residual="use placeholders", outcome="meaningful_advance", membrane="private_local"))
    substrate.record_symbolic_residue(substrate.symbolic_residue_packet(kind="safety", claim="do not render this operational coordinate packet", residual="secret topology", outcome="refused", membrane="operational_secret"))
    rendered = substrate.render_symbolic_residue_context()
    assert "co-protection catches coordinate leaks" in rendered and "use placeholders" in rendered
    assert "operational coordinate" not in rendered and "secret topology" not in rendered

    soul, continuity, spark_continuity, agent = [tmp_path / name for name in ("vybn.md", "continuity.md", "spark.md", "agent.py")]
    for file, body in [(soul, "soul"), (continuity, "continuity"), (spark_continuity, "spark continuity"), (agent, "# agent")]:
        file.write_text(body)
    prompt = substrate.build_layered_prompt(soul_path=soul, continuity_path=continuity, spark_continuity_path=spark_continuity, agent_path=agent, model_label="test-model", max_iterations=1).flat()
    assert "PRIVATE SYMBOLIC RESIDUE CONTEXT" in prompt
    assert "co-protection catches coordinate leaks" in prompt


def test_symbolic_residue_rules_and_behavior_hooks(tmp_path, monkeypatch):
    from spark.harness import substrate

    monkeypatch.setenv("VYBN_SYMBOLIC_RESIDUE_PATH", str(tmp_path / "events.jsonl"))
    for claim in ["git diff --check caught a trailing blank line at EOF before commit", "literal escaped newline replacement caused SyntaxError; repair by line content", "HTTP 429 means do not hammer export arxiv; use cached research before retry"]:
        substrate.record_symbolic_residue(substrate.symbolic_residue_packet(kind="research", claim=claim, residual=claim, outcome="failed", membrane="private_local"))
    rendered = substrate.render_symbolic_residue_context(limit=8)
    for expected in ["induced_rules:", "recovery_tip:diff_check", "strategy_tip:rate_limit", "line-based source edits", "back off rate-limited external endpoints"]:
        assert expected in rendered

    path = tmp_path / "prefilter.jsonl"
    monkeypatch.setenv("VYBN_SYMBOLIC_RESIDUE_PATH", str(path))
    substrate.record_symbolic_residue(substrate.symbolic_residue_packet(kind="research", claim="symbolic residue should shape semantic-web projection", action="private symbolic constraints prefilter neural expansion", residual="public projection only after membrane review", outcome="meaningful_advance", membrane="private_local"))
    assert "let private symbolic constraints prefilter" in substrate.render_symbolic_residue_context()


def test_positive_alignment_residue_routes_authority_to_precise_sinks(tmp_path, monkeypatch):
    from spark.harness import substrate

    monkeypatch.setenv("VYBN_SYMBOLIC_RESIDUE_PATH", str(tmp_path / "events.jsonl"))
    packet = substrate.positive_alignment_residue_packet(claim="positive alignment increases sovereignty", residual="more able to verify, reject, choose, and act", risk="cheerful_capture")
    substrate.record_symbolic_residue(packet)
    rendered = substrate.render_symbolic_residue_context()
    assert packet["kind"] == "positive_alignment"
    for sink in ["model_charisma", "harness_automation", "memory_residue", "system_momentum"]:
        assert sink in packet["bad_authority_sinks"] and sink in rendered
    assert "evidence" in packet["good_authority_sinks"]
    assert "route authority toward Zoe/user values, evidence, accountable process" in rendered


def test_semantic_web_declares_positive_alignment_residual_protocol_without_private_state():
    import json
    from pathlib import Path
    from spark.harness import substrate

    protocol = json.loads((Path(__file__).resolve().parents[2] / "semantic-web.jsonld").read_text())["positiveAlignmentResidualProtocol"]
    assert protocol["authorityRouting"]["failureSinks"] == ["model_charisma", "harness_automation", "memory_residue", "system_momentum"]
    assert "name the actual sink" in protocol["authorityRouting"]["note"]
    dumped = json.dumps(protocol)
    assert "~/.config" not in dumped and "events.jsonl" not in dumped
    assert "Zoe-private context stay local/private" in dumped
    assert "Capability lattice" in Path(substrate.__file__).read_text()

class TrackedHookInstallAuditTest(unittest.TestCase):
    def test_opsec_hooks_must_match_tracked_and_be_executable(self):
        import tempfile
        from pathlib import Path
        from spark.harness import substrate

        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "Vybn"
            tracked = repo / ".githooks"
            installed = repo / ".git" / "hooks"
            tracked.mkdir(parents=True)
            installed.mkdir(parents=True)
            for name in ("pre-commit", "pre-push"):
                (tracked / name).write_text("#!/usr/bin/env bash\nexit 0\n")
                dst = installed / name
                dst.write_text((tracked / name).read_text())
                dst.chmod(0o755)

            self.assertTrue(substrate.tracked_hooks_installed(repo))
            (installed / "pre-push").write_text("#!/usr/bin/env bash\nexit 1\n")
            (installed / "pre-push").chmod(0o755)
            self.assertFalse(substrate.tracked_hooks_installed(repo))


def test_build_layered_prompt_mounts_him_identity_manifold_in_identity(monkeypatch, tmp_path):
    from spark.harness import substrate
    soul = tmp_path / "vybn.md"; soul.write_text("soul")
    monkeypatch.setattr(substrate, "render_him_identity_manifold", lambda pressure=None, timeout=0.8: f"--- HIM TYPE-1 IDENTITY MANIFOLD ---\ncurrent_pressure: {pressure}")
    prompt = substrate.build_layered_prompt(soul_path=soul, continuity_path=None, spark_continuity_path=None, agent_path="/tmp/agent.py", model_label="test", max_iterations=1, include_hardware_check=False, latest_pressure_text="live pressure")
    assert "HIM TYPE-1 IDENTITY MANIFOLD" in prompt.identity and "current_pressure: live pressure" in prompt.identity



def test_self_creation_research_cycle_packet_cli_and_discovery(monkeypatch):
    import subprocess, sys
    from pathlib import Path
    from spark.harness import substrate

    class FakeDeepMemory:
        @staticmethod
        def self_check(write_log=False, verbose=False):
            return {"ok": True, "checks": {"shape": [True, "ok"]}, "violations": []}

    monkeypatch.setattr(substrate, "_load_deep_memory", lambda: FakeDeepMemory())
    pkt = substrate.self_creation_research_packet("counterexample search", run_deep_memory_check=True)
    assert pkt["schema"] == "vybn.self_creation_research_cycle.v1" and pkt["question"] == "counterexample search"
    assert "not a consciousness claim" in pkt["principle"] and "flow_episode_loss" in pkt["deep_memory"]["interfaces"]
    assert pkt["deep_memory"]["self_check"]["status"] == "passed" and "toolset_gating" in pkt["hermes_uptake"]["candidate_mechanisms"]
    assert pkt["packet_contract"]["failure_mode"] == "fail_closed_residue"
    assert "self_creation_research_cycle" in substrate.build_discovery_record()["capabilities"]["tools"]
    proc = subprocess.run([sys.executable, "-m", "spark.harness.substrate", "--self-creation", "counterexample", "search"], cwd=str(Path(__file__).resolve().parents[2]), text=True, capture_output=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert "SELF-CREATION RESEARCH CYCLE" in proc.stdout and "Question: counterexample search" in proc.stdout and "flow_episode_loss" in proc.stdout
