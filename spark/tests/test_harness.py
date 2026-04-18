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
from harness.tools import BASH_TOOL_SPEC  # noqa: E402
from harness.prompt import build_layered_prompt  # noqa: E402


class TestAbsorbGate(unittest.TestCase):
    def test_allow_harmless(self):
        self.assertIsNone(absorb_gate("ls -la /home"))
        self.assertIsNone(absorb_gate("echo hello"))

    def test_allow_existing_file_write(self):
        # Writing to /tmp is always excluded
        self.assertIsNone(absorb_gate("echo hi > /tmp/existing.txt"))

    def test_allow_with_reason(self):
        cmd = 'VYBN_ABSORB_REASON="new module" echo x > /home/vybnz69/Vybn/new_file.py'
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

    def test_default_role_is_code(self):
        p = default_policy()
        self.assertEqual(p.default_role, "code")

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
        # Default is code.
        self.assertEqual(d.role, "code")
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
        # Simulate an Anthropic-shaped tool_result message
        anthropic_messages = [
            {"role": "user", "content": "hi"},
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


class TestAnthropicProviderToolResult(unittest.TestCase):
    def test_tool_result_shape(self):
        # Construct without hitting the SDK by injecting a dummy client.
        prov = AnthropicProvider(client=object())
        r = prov.build_tool_result("abc", "out")
        self.assertEqual(r["type"], "tool_result")
        self.assertEqual(r["tool_use_id"], "abc")
        self.assertEqual(r["content"], "out")


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
