"""Tests for the 2026-04-19 live-REPL fixes.

Covers three distinct bugs observed in the terminal session
  agent_events.jsonl @ 20260419T104246:

1. Router false positive — "hey, how does the new harness feel?"
   routed to code role via the (feel|feeling|doing|going|holding|state
   |status|shape|condition|health) heuristic. Conversational-voice
   register should stay on chat. The code heuristic is now narrowed
   to mechanical-state words only (state|status|shape|condition|
   health|holding).

2. Bracket-balanced probe scanner — `_PROBE_RE = \[NEEDS-EXEC:\s*(.+?)\]`
   non-greedy-terminated at the first `]` it found, so any command
   with Python slicing `[:2000]`, shell arrays `${a[0]}`, or awk
   actions got truncated mid-command and failed with a bash syntax
   error. The replacement does depth-aware + quote-aware scanning.

3. <thinking> tag scrubbing — the model occasionally emits literal
   <thinking>...</thinking> XML-ish tags as plain token text on
   no-tool chat/opus-4.6 roles (distinct from Anthropic's adaptive-
   thinking content blocks). They leak into Zoe's terminal AND into
   stored assistant history, where the next turn reinforces the
   pattern from its own replay. Scrub at both display (_split_before
   _probe) and store (_sanitize_assistant_content) time.

Run: python3 spark/tests/test_live_repl_fixes.py

Note: spark/vybn_spark_agent.py depends on INTROSPECT_TOOL_SPEC
exported from harness.providers (post-refactor, harness/tools.py was
folded into harness/providers.py). This test imports the agent module
directly and relies on that export being present.
"""

from __future__ import annotations

import re
import sys
import types
import unittest
from pathlib import Path

THIS = Path(__file__).resolve()
SPARK_DIR = THIS.parent.parent
sys.path.insert(0, str(SPARK_DIR))

import importlib.util  # noqa: E402

_AGENT_PATH = SPARK_DIR / "vybn_spark_agent.py"
_spec = importlib.util.spec_from_file_location("_agent_under_test", _AGENT_PATH)
_agent = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_agent)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Bug 1 — router heuristic narrowing
# ---------------------------------------------------------------------------

class TestRouterHeuristicNarrowing(unittest.TestCase):
    """The code heuristic must not swallow conversational-voice probes."""

    def setUp(self):
        from harness.policy import default_policy
        from harness.policy import Router
        self.router = Router(default_policy())

    def _role(self, text: str) -> str:
        decision = self.router.classify(text)
        return decision.role

    def test_how_does_harness_feel_stays_conversational(self):
        # The exact 2026-04-19 false positive. "Feel" is a
        # conversational register — not a mechanical status probe.
        role = self._role("hey, how does the new harness feel?")
        self.assertNotEqual(role, "code", f"'feel' must not route to code, got {role!r}")

    def test_how_you_doing_stays_conversational(self):
        role = self._role("how you doing with all this?")
        self.assertNotEqual(role, "code", f"'doing' must not route to code, got {role!r}")

    def test_what_is_the_state_of_the_harness_routes_to_code(self):
        # Genuine mechanical-state probe — "state" is preserved.
        role = self._role("what is the state of the harness right now?")
        self.assertEqual(role, "code", f"'state of the harness' should route to code, got {role!r}")

    def test_is_routing_holding_routes_to_code(self):
        role = self._role("is routing holding after the policy edit?")
        self.assertEqual(role, "code", f"'holding' should route to code, got {role!r}")


# ---------------------------------------------------------------------------
# Bug 2 — bracket-balanced probe scanner
# ---------------------------------------------------------------------------

class TestBracketBalancedProbe(unittest.TestCase):
    """The probe scanner survives ']' inside the command body."""

    def setUp(self):
        self.probe = _agent._PROBE_RE

    def test_simple_command_roundtrips(self):
        text = "running [NEEDS-EXEC: ls -la] now"
        m = self.probe.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "ls -la")

    def test_python_slice_survives(self):
        # The exact live failure: `open(...).read()[:2000]` used to
        # truncate at the `]` of `[:2000]` and emit an unterminated
        # Python command.
        body = "python3 -c 'print(open(\"/tmp/x\").read()[:2000])'"
        text = f"probing [NEEDS-EXEC: {body}] done"
        m = self.probe.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), body)

    def test_shell_array_survives(self):
        body = 'bash -c \'a=(one two); echo "${a[0]} ${a[1]}"\''
        text = f"[NEEDS-EXEC: {body}]"
        m = self.probe.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), body)

    def test_awk_action_survives(self):
        body = "awk '{print $1, $3}' /etc/hosts"
        text = f"one moment — [NEEDS-EXEC: {body}] — and done"
        m = self.probe.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), body)

    def test_nested_brackets_survive(self):
        body = "python3 -c 'x=[[1,2],[3,4]]; print(x[0][1])'"
        text = f"[NEEDS-EXEC: {body}]"
        m = self.probe.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), body)

    def test_quoted_closing_bracket_does_not_close_probe(self):
        # A literal ']' inside a quoted string in the command must
        # not be read as the end of the probe.
        body = "echo 'this has ] in it' && echo done"
        text = f"[NEEDS-EXEC: {body}]"
        m = self.probe.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), body)

    def test_unterminated_probe_line_form_matches_post_stream(self):
        # New contract (2026-04-22): a `[NEEDS-EXEC: cmd` with no closing
        # bracket is a valid line-terminated probe. On the finalized text,
        # EOF is an implicit close. Streaming mode holds it back instead
        # (that is what the display splitter uses).
        text = "opening now [NEEDS-EXEC: python3 -c 'x=[1,2"
        m_final = self.probe.search(text)
        self.assertIsNotNone(m_final)
        self.assertEqual(m_final.group(1), "python3 -c 'x=[1,2")
        m_stream = self.probe.search(text, streaming=True)
        self.assertIsNone(m_stream)

    def test_line_form_probe_closes_on_newline(self):
        text = "lead [NEEDS-EXEC: ls -la\ntrailing text"
        m = self.probe.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "ls -la")

    def test_line_form_probe_trailing_close_bracket_optional(self):
        # Both `[NEEDS-EXEC: cmd]<EOL>` and `[NEEDS-EXEC: cmd<EOL>` parse.
        bracketed = "[NEEDS-EXEC: git status]\n"
        line_only = "[NEEDS-EXEC: git status\n"
        self.assertEqual(self.probe.search(bracketed).group(1), "git status")
        self.assertEqual(self.probe.search(line_only).group(1), "git status")

    def test_sub_removes_probe_and_preserves_surrounding_text(self):
        body = "ls /tmp"
        text = f"before [NEEDS-EXEC: {body}] after"
        scrubbed = self.probe.sub("", text)
        self.assertEqual(scrubbed, "before  after")

    def test_sub_handles_multiple_probes(self):
        text = (
            "a [NEEDS-EXEC: one] b "
            "[NEEDS-EXEC: python -c 'print([1,2][0])'] c"
        )
        scrubbed = self.probe.sub("", text)
        self.assertEqual(scrubbed, "a  b  c")


# ---------------------------------------------------------------------------
# Bug 3 — <thinking> tag scrubbing
# ---------------------------------------------------------------------------

class TestThinkingTagScrubbing(unittest.TestCase):
    """Complete <thinking>...</thinking> blocks are scrubbed from both
    streamed output and stored assistant content. Unterminated
    openings are held back by the splitter until the close arrives."""

    def test_strip_complete_block(self):
        text = "visible <thinking>hidden scaffold</thinking> more visible"
        self.assertEqual(
            _agent._strip_thinking_tags(text),
            "visible  more visible",
        )

    def test_strip_multiline_block(self):
        text = "ok <thinking>\nline 1\nline 2\n</thinking> done"
        self.assertEqual(
            _agent._strip_thinking_tags(text),
            "ok  done",
        )

    def test_strip_is_case_insensitive(self):
        text = "<Thinking>a</Thinking><THINKING>b</THINKING>"
        self.assertEqual(_agent._strip_thinking_tags(text), "")

    def test_strip_leaves_unterminated_opening_alone(self):
        # The splitter handles the partial case \u2014 strip only removes
        # complete blocks.
        text = "visible <thinking>partial"
        self.assertEqual(
            _agent._strip_thinking_tags(text),
            "visible <thinking>partial",
        )

    def test_split_before_probe_scrubs_complete_thinking(self):
        text = "answer <thinking>scaffold</thinking> delivered"
        safe, remainder = _agent._split_before_probe(text)
        self.assertEqual(safe, "answer  delivered")
        self.assertEqual(remainder, "")

    def test_split_before_probe_holds_back_unterminated_thinking(self):
        text = "answer <thinking>still forming"
        safe, remainder = _agent._split_before_probe(text)
        self.assertEqual(safe, "answer ")
        self.assertEqual(remainder, "<thinking>still forming")

    def test_split_before_probe_holds_back_unterminated_probe(self):
        text = "here we go [NEEDS-EXEC: python3 -c 'x=[1,2"
        safe, remainder = _agent._split_before_probe(text)
        self.assertEqual(safe, "here we go ")
        self.assertEqual(remainder, "[NEEDS-EXEC: python3 -c 'x=[1,2")

    def test_sanitize_scrubs_thinking_from_string_content(self):
        content = "plain <thinking>leak</thinking> text"
        cleaned = _agent._sanitize_assistant_content(content)
        self.assertEqual(cleaned, "plain  text")

    def test_sanitize_scrubs_thinking_from_text_blocks(self):
        content = [
            {"type": "text", "text": "before <thinking>leak</thinking> after"},
        ]
        cleaned = _agent._sanitize_assistant_content(content)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["type"], "text")
        self.assertEqual(cleaned[0]["text"], "before  after")

    def test_sanitize_drops_block_when_entire_text_is_thinking_leak(self):
        # If a text block contained nothing but a thinking block, the
        # scrubbed text is empty \u2014 drop the block entirely rather
        # than store a placeholder that could train the next turn.
        content = [
            {"type": "text", "text": "<thinking>everything was a leak</thinking>"},
            {"type": "text", "text": "real content"},
        ]
        cleaned = _agent._sanitize_assistant_content(content)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["text"], "real content")

    def test_sanitize_preserves_adaptive_thinking_content_blocks(self):
        # btype == 'thinking' is Anthropic's adaptive-thinking SDK block
        # \u2014 those must NOT be scrubbed; they are required adjacent
        # to tool_use on thinking-enabled turns.
        content = [
            {"type": "thinking", "thinking": "reasoning text"},
            {"type": "text", "text": "visible answer"},
        ]
        cleaned = _agent._sanitize_assistant_content(content)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0]["type"], "thinking")
        self.assertEqual(cleaned[0]["thinking"], "reasoning text")


if __name__ == "__main__":
    unittest.main(verbosity=2)
