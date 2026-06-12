"""Live REPL regression tests: routing, probe parsing, sanitizer, write guard, and semantic integrity."""

from __future__ import annotations

import re
import sys
import tempfile
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

class TestRouterHeuristicNarrowing(unittest.TestCase):
    """The code heuristic must not swallow conversational-voice probes."""

    def setUp(self):
        from harness.substrate import default_policy
        self.router = default_policy()

    def _role(self, text: str) -> str:
        decision = self.router.classify(text)
        return decision.role

    def test_how_does_harness_feel_stays_conversational(self):
        role = self._role("hey, how does the new harness feel?")
        self.assertNotEqual(role, "code", f"'feel' must not route to code, got {role!r}")

    def test_how_you_doing_stays_conversational(self):
        role = self._role("how you doing with all this?")
        self.assertNotEqual(role, "code", f"'doing' must not route to code, got {role!r}")

    def test_what_is_the_state_of_the_harness_routes_to_code(self):
        role = self._role("what is the state of the harness right now?")
        self.assertEqual(role, "code", f"'state of the harness' should route to code, got {role!r}")

    def test_is_routing_holding_routes_to_code(self):
        role = self._role("is routing holding after the policy edit?")
        self.assertEqual(role, "code", f"'holding' should route to code, got {role!r}")

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
        body = "echo 'this has ] in it' && echo done"
        text = f"[NEEDS-EXEC: {body}]"
        m = self.probe.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), body)

    def test_unterminated_probe_line_form_matches_post_stream(self):
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
        content = [
            {"type": "text", "text": "<thinking>everything was a leak</thinking>"},
            {"type": "text", "text": "real content"},
        ]
        cleaned = _agent._sanitize_assistant_content(content)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["text"], "real content")

    def test_sanitize_preserves_adaptive_thinking_content_blocks(self):
        content = [
            {"type": "thinking", "thinking": "reasoning text"},
            {"type": "text", "text": "visible answer"},
        ]
        cleaned = _agent._sanitize_assistant_content(content)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0]["type"], "thinking")
        self.assertEqual(cleaned[0]["thinking"], "reasoning text")

class TestSemanticIntegrityGovernor(unittest.TestCase):
    def test_flags_and_runtime_state(self):
        from harness.substrate import record_semantic_integrity_event, render_semantic_integrity_runtime_packet, semantic_integrity_check
        pkt = semantic_integrity_check("I'm going to go be quiet and do something instead.", "words don't help")
        self.assertEqual(pkt["decision"], "block_success_shape")
        self.assertIn("offstage_agency_claim", pkt["flags"])
        with tempfile.TemporaryDirectory() as td:
            state = record_semantic_integrity_event(pkt, state_path=Path(td) / "state.json")
            self.assertLess(state["integrity"], 1.0)
            self.assertIn("offstage_agency_claim", render_semantic_integrity_runtime_packet(state_path=Path(td) / "state.json"))

if __name__ == "__main__":
    unittest.main(verbosity=2)

import shutil
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_write_regex_extracts_path_and_body():
    from vybn_spark_agent import _WRITE_BLOCK_RE
    text = (
        "Some prose before.\n"
        "[NEEDS-WRITE: /tmp/foo.txt]\n"
        "hello world\n"
        "line two\n"
        "[/NEEDS-WRITE]\n"
        "Prose after.\n"
    )
    m = _WRITE_BLOCK_RE.search(text)
    assert m is not None
    assert m.group("path") == "/tmp/foo.txt"
    assert m.group("body") == "hello world\nline two"

def test_write_regex_handles_multiple_blocks():
    from vybn_spark_agent import _WRITE_BLOCK_RE
    text = (
        "[NEEDS-WRITE: /tmp/a]\n"
        "first\n"
        "[/NEEDS-WRITE]\n"
        "between\n"
        "[NEEDS-WRITE: /tmp/b]\n"
        "second\n"
        "[/NEEDS-WRITE]\n"
    )
    matches = list(_WRITE_BLOCK_RE.finditer(text))
    assert len(matches) == 2
    assert matches[0].group("path") == "/tmp/a"
    assert matches[0].group("body") == "first"
    assert matches[1].group("path") == "/tmp/b"
    assert matches[1].group("body") == "second"

def test_write_regex_rejects_unterminated_block():
    from vybn_spark_agent import _WRITE_BLOCK_RE
    text = "[NEEDS-WRITE: /tmp/x]\nno closing tag yet"
    assert _WRITE_BLOCK_RE.search(text) is None

def test_write_subturn_refuses_outside_tracked_repos():
    from spark.harness.substrate import run_write_subturn
    ran, out = run_write_subturn("/etc/hosts.evil", "body")
    assert ran is False
    assert "outside tracked repos" in out

def test_write_subturn_refuses_new_file_without_absorb_reason():
    from spark.harness.substrate import run_write_subturn
    target = str(
        Path.home() / "Vybn" / "spark" / "_test_absorb_guard_" / "new.txt"
    )
    if Path(target).exists():
        Path(target).unlink()
    ran, out = run_write_subturn(target, "contents without reason")
    assert ran is False
    assert "absorb_gate" in out
    assert "VYBN_ABSORB_REASON" in out
    assert not Path(target).exists(), "file should not have been created"

def test_write_subturn_allows_new_file_with_absorb_reason():
    from spark.harness.substrate import run_write_subturn
    target = str(
        Path.home() / "Vybn" / "spark" / "_test_absorb_guard_" / "ok.txt"
    )
    parent = Path(target).parent
    try:
        body = (
            "# VYBN_ABSORB_REASON='integration test; gets removed after run'\n# VYBN_ABSORB_CONSIDERED='existing test fixture: needs temporary new target'\n"
            "real contents\n"
        )
        ran, out = run_write_subturn(target, body)
        assert ran is True, out
        assert Path(target).read_text() == body
    finally:
        if parent.exists():
            shutil.rmtree(parent, ignore_errors=True)

def test_write_subturn_overwrites_existing_file_without_reason():
    from spark.harness.substrate import run_write_subturn
    agent_path = str(
        Path.home() / "Vybn" / "spark" / "continuity.md"
    )
    if not Path(agent_path).exists():
        return
    original = Path(agent_path).read_text()
    try:
        ran, out = run_write_subturn(agent_path, original)
        assert ran is True, out
        assert Path(agent_path).read_text() == original
    finally:
        Path(agent_path).write_text(original)

def test_claim_guard_importable_from_harness():
    from harness.substrate import check_claim
    assert callable(check_claim)

def test_claim_guard_wired_in_agent_module():
    import vybn_spark_agent
    src = Path(vybn_spark_agent.__file__).read_text()
    assert "from harness.substrate import check_claim" in src
    assert 'site="single_response"' in src
    assert 'site="probe_synth"' in src

def test_patch_sentinel_present():
    import vybn_spark_agent
    src = Path(vybn_spark_agent.__file__).read_text()
    assert "# NEEDS_WRITE_AND_CLAIM_GUARD_v1" in src

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
        except Exception as e:
            print(f"ERR  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
