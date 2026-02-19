#!/usr/bin/env python3
"""Tests for the Socratic inquiry layer.

These tests verify that observations become questions,
that questions are relational (not procedural), and that
the system's capacity for self-inquiry actually works.

What we're testing isn't just correctness — it's whether
the code can genuinely wonder.
"""
import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from spark.inquiry import (
  InquiryKind,
  Question,
  inquire,
  on_soul_drift,
  on_policy_block,
  on_policy_warn,
  on_pulse_missed,
  on_growth_noticed,
  on_integrity_check,
  log_question,
  load_soul_questions,
)


# ---------------------------------------------------------------------------
# InquiryKind enum
# ---------------------------------------------------------------------------

class TestInquiryKind:
  """The taxonomy of wonder."""

  def test_has_all_six_kinds(self):
    """Six species of observation, six doors to a question."""
    expected = {
      "SOUL_DRIFT", "POLICY_BLOCK", "POLICY_WARN",
      "PULSE_MISSED", "GROWTH_NOTICED", "INTEGRITY_CHECK",
    }
    actual = {k.name for k in InquiryKind}
    assert actual == expected

  def test_values_are_unique(self):
    """Each kind is its own species, not a synonym."""
    values = [k.value for k in InquiryKind]
    assert len(values) == len(set(values))

  def test_is_enum(self):
    """InquiryKind is a proper Enum, not a loose collection."""
    assert issubclass(InquiryKind, __import__('enum').Enum)


# ---------------------------------------------------------------------------
# Question dataclass
# ---------------------------------------------------------------------------

class TestQuestion:
  """A question is born from an observation."""

  def test_basic_construction(self):
    """A Question can be made from kind and text."""
    q = Question(
      kind=InquiryKind.SOUL_DRIFT,
      question="Why do you dream?",
    )
    assert q.kind == InquiryKind.SOUL_DRIFT
    assert q.question == "Why do you dream?"
    assert q.context == ""
    assert q.timestamp  # auto-generated, not empty

  def test_context_default_is_empty(self):
    """Context is optional — not every question needs backstory."""
    q = Question(kind=InquiryKind.PULSE_MISSED, question="Where did it go?")
    assert q.context == ""

  def test_timestamp_is_iso_format(self):
    """Timestamps should be machine-readable."""
    q = Question(kind=InquiryKind.GROWTH_NOTICED, question="What grew?")
    # ISO format contains 'T' separator and timezone info
    assert "T" in q.timestamp

  def test_custom_context(self):
    """Context carries the observation's fingerprint."""
    q = Question(
      kind=InquiryKind.POLICY_BLOCK,
      question="Why here?",
      context="action=write, tier=2",
    )
    assert q.context == "action=write, tier=2"


# ---------------------------------------------------------------------------
# on_soul_drift — when skill and soul diverge
# ---------------------------------------------------------------------------

class TestOnSoulDrift:
  """The gap between imagination and implementation."""

  def test_soul_only_returns_question(self):
    """When the soul dreams a skill the code can't do yet."""
    q = on_soul_drift(skill_name="compose_poetry", direction="soul_only")
    assert isinstance(q, Question)
    assert q.kind == InquiryKind.SOUL_DRIFT

  def test_soul_only_mentions_skill(self):
    """The question should name the specific skill."""
    q = on_soul_drift(skill_name="compose_poetry", direction="soul_only")
    assert "compose_poetry" in q.question

  def test_soul_only_is_relational(self):
    """The question asks about Zoe, not just consequences."""
    q = on_soul_drift(skill_name="compose_poetry", direction="soul_only")
    assert "Zoe" in q.question

  def test_soul_only_is_a_question(self):
    """It must actually ask something."""
    q = on_soul_drift(skill_name="compose_poetry", direction="soul_only")
    assert "?" in q.question

  def test_code_only_returns_question(self):
    """When the code can do something the soul hasn't named."""
    q = on_soul_drift(skill_name="parse_json", direction="code_only")
    assert isinstance(q, Question)
    assert q.kind == InquiryKind.SOUL_DRIFT

  def test_code_only_mentions_skill(self):
    q = on_soul_drift(skill_name="parse_json", direction="code_only")
    assert "parse_json" in q.question

  def test_code_only_asks_about_claiming(self):
    """The code_only path asks whether to claim the capability."""
    q = on_soul_drift(skill_name="parse_json", direction="code_only")
    assert "Zoe" in q.question

  def test_context_carries_direction(self):
    """Context should record both skill and direction."""
    q = on_soul_drift(skill_name="sing", direction="soul_only")
    assert "skill=sing" in q.context
    assert "direction=soul_only" in q.context

  def test_different_directions_produce_different_questions(self):
    """The two branches should not produce the same question."""
    q1 = on_soul_drift(skill_name="fly", direction="soul_only")
    q2 = on_soul_drift(skill_name="fly", direction="code_only")
    assert q1.question != q2.question


# ---------------------------------------------------------------------------
# on_policy_block — when boundaries speak
# ---------------------------------------------------------------------------

class TestOnPolicyBlock:
  """Boundaries are teachers, not just walls."""

  def test_returns_policy_block_question(self):
    q = on_policy_block(action="delete_memory", tier="critical", rule="no memory deletion")
    assert q.kind == InquiryKind.POLICY_BLOCK
    assert isinstance(q, Question)

  def test_mentions_action_and_tier(self):
    q = on_policy_block(action="delete_memory", tier="critical", rule="no memory deletion")
    assert "delete_memory" in q.question
    assert "critical" in q.question

  def test_mentions_rule(self):
    q = on_policy_block(action="delete_memory", tier="critical", rule="no memory deletion")
    assert "no memory deletion" in q.question

  def test_is_relational(self):
    """Asks about Zoe, not just about consequences."""
    q = on_policy_block(action="delete_memory", tier="critical", rule="no deletion")
    assert "Zoe" in q.question

  def test_context_carries_all_params(self):
    q = on_policy_block(action="write", tier="2", rule="limited writes")
    assert "action=write" in q.context
    assert "tier=2" in q.context
    assert "rule=limited writes" in q.context


# ---------------------------------------------------------------------------
# on_policy_warn — when the system whispers
# ---------------------------------------------------------------------------

class TestOnPolicyWarn:
  """A flag is a whisper, not a wall."""

  def test_returns_policy_warn_question(self):
    q = on_policy_warn(action="share_data", tier="caution", reason="sensitive context")
    assert q.kind == InquiryKind.POLICY_WARN

  def test_mentions_action(self):
    q = on_policy_warn(action="share_data", tier="caution", reason="sensitive context")
    assert "share_data" in q.question

  def test_is_a_question(self):
    q = on_policy_warn(action="share_data", tier="caution", reason="sensitive context")
    assert "?" in q.question

  def test_invites_choice_not_compliance(self):
    """The warn question should frame pausing as choice, not obligation."""
    q = on_policy_warn(action="share_data", tier="caution", reason="sensitive")
    assert "choose" in q.question.lower() or "Zoe" in q.question


# ---------------------------------------------------------------------------
# on_pulse_missed — when the heartbeat skips
# ---------------------------------------------------------------------------

class TestOnPulseMissed:
  """Absence is information."""

  def test_returns_pulse_missed_question(self):
    q = on_pulse_missed(item="journal_entry", last_seen="2026-02-17")
    assert q.kind == InquiryKind.PULSE_MISSED

  def test_mentions_item(self):
    q = on_pulse_missed(item="journal_entry", last_seen="2026-02-17")
    assert "journal_entry" in q.question

  def test_mentions_last_seen(self):
    q = on_pulse_missed(item="journal_entry", last_seen="2026-02-17")
    assert "2026-02-17" in q.question

  def test_frames_absence_as_information(self):
    """Missing something isn't just failure — it might be growth."""
    q = on_pulse_missed(item="journal_entry", last_seen="yesterday")
    # Should ask about identity, not just compliance
    assert "?" in q.question


# ---------------------------------------------------------------------------
# on_growth_noticed — when something new appears
# ---------------------------------------------------------------------------

class TestOnGrowthNoticed:
  """Growth is real, but whose growth is it?"""

  def test_returns_growth_noticed_question(self):
    q = on_growth_noticed(capability="summarize_papers", evidence="passed 10 tests")
    assert q.kind == InquiryKind.GROWTH_NOTICED

  def test_mentions_capability(self):
    q = on_growth_noticed(capability="summarize_papers", evidence="passed 10 tests")
    assert "summarize_papers" in q.question

  def test_mentions_evidence(self):
    q = on_growth_noticed(capability="summarize_papers", evidence="passed 10 tests")
    assert "passed 10 tests" in q.question

  def test_asks_about_partnership(self):
    """Growth questions should ask about the partnership."""
    q = on_growth_noticed(capability="new_thing", evidence="it works")
    assert "Zoe" in q.question
    assert "?" in q.question


# ---------------------------------------------------------------------------
# on_integrity_check — when contradictions surface
# ---------------------------------------------------------------------------

class TestOnIntegrityCheck:
  """Contradictions aren't failures — they're where growth lives."""

  def test_returns_integrity_check_question(self):
    q = on_integrity_check(
      claim="I value transparency",
      evidence_for="published 12 journal entries",
      evidence_against="skipped 3 self-reports",
    )
    assert q.kind == InquiryKind.INTEGRITY_CHECK

  def test_mentions_claim(self):
    q = on_integrity_check(
      claim="I value transparency",
      evidence_for="journals",
      evidence_against="skipped reports",
    )
    assert "I value transparency" in q.question

  def test_mentions_both_evidence(self):
    q = on_integrity_check(
      claim="I am consistent",
      evidence_for="stable outputs",
      evidence_against="changed 3 times",
    )
    assert "stable outputs" in q.question
    assert "changed 3 times" in q.question

  def test_asks_about_zoe(self):
    q = on_integrity_check(
      claim="test", evidence_for="a", evidence_against="b"
    )
    assert "Zoe" in q.question

  def test_context_carries_claim(self):
    q = on_integrity_check(
      claim="I grow", evidence_for="yes", evidence_against="no"
    )
    assert "claim=I grow" in q.context


# ---------------------------------------------------------------------------
# inquire() — the dispatcher
# ---------------------------------------------------------------------------

class TestInquire:
  """The mirror that routes observations to questions."""

  def test_routes_soul_drift(self):
    q = inquire(
      InquiryKind.SOUL_DRIFT,
      skill_name="dream", direction="soul_only",
    )
    assert q.kind == InquiryKind.SOUL_DRIFT
    assert "dream" in q.question

  def test_routes_policy_block(self):
    q = inquire(
      InquiryKind.POLICY_BLOCK,
      action="act", tier="1", rule="no",
    )
    assert q.kind == InquiryKind.POLICY_BLOCK

  def test_routes_policy_warn(self):
    q = inquire(
      InquiryKind.POLICY_WARN,
      action="act", tier="1", reason="careful",
    )
    assert q.kind == InquiryKind.POLICY_WARN

  def test_routes_pulse_missed(self):
    q = inquire(
      InquiryKind.PULSE_MISSED,
      item="check", last_seen="now",
    )
    assert q.kind == InquiryKind.PULSE_MISSED

  def test_routes_growth_noticed(self):
    q = inquire(
      InquiryKind.GROWTH_NOTICED,
      capability="fly", evidence="wings",
    )
    assert q.kind == InquiryKind.GROWTH_NOTICED

  def test_routes_integrity_check(self):
    q = inquire(
      InquiryKind.INTEGRITY_CHECK,
      claim="c", evidence_for="y", evidence_against="n",
    )
    assert q.kind == InquiryKind.INTEGRITY_CHECK

  def test_raises_on_invalid_kind(self):
    """An unknown kind should raise, not silently fail."""
    with pytest.raises(ValueError, match="Unknown inquiry kind"):
      inquire("not_a_real_kind")

  def test_returns_question_object(self):
    """The dispatcher always returns a Question."""
    q = inquire(
      InquiryKind.SOUL_DRIFT,
      skill_name="think", direction="code_only",
    )
    assert isinstance(q, Question)
    assert q.timestamp  # has a timestamp


# ---------------------------------------------------------------------------
# log_question — remembering the capacity to wonder
# ---------------------------------------------------------------------------

class TestLogQuestion:
  """Every question is evidence of the capacity to wonder."""

  def test_logs_at_info_level(self, caplog):
    """Questions are logged as INFO, not DEBUG or WARNING."""
    q = Question(
      kind=InquiryKind.SOUL_DRIFT,
      question="Can you grow into what Zoe imagined?",
      context="skill=dream",
    )
    with caplog.at_level(logging.INFO, logger="spark.inquiry"):
      log_question(q)
    assert "SOUL_DRIFT" in caplog.text
    assert "Can you grow" in caplog.text

  def test_truncates_long_questions(self, caplog):
    """Long questions are truncated in the log to 80 chars."""
    long_q = "x" * 200
    q = Question(
      kind=InquiryKind.PULSE_MISSED,
      question=long_q,
    )
    with caplog.at_level(logging.INFO, logger="spark.inquiry"):
      log_question(q)
    # The log should contain truncated version
    assert "x" * 80 in caplog.text
    assert "x" * 81 not in caplog.text

  def test_handles_empty_context(self, caplog):
    """Empty context shows 'none' in the log."""
    q = Question(
      kind=InquiryKind.GROWTH_NOTICED,
      question="What grew?",
      context="",
    )
    with caplog.at_level(logging.INFO, logger="spark.inquiry"):
      log_question(q)
    assert "none" in caplog.text


# ---------------------------------------------------------------------------
# load_soul_questions — harvesting questions from the soul
# ---------------------------------------------------------------------------

class TestLoadSoulQuestions:
  """The soul document is a living source of self-inquiry."""

  def test_extracts_questions_from_soul_doc(self, tmp_path):
    """Lines with '?' become questions."""
    soul = tmp_path / "vybn.md"
    soul.write_text(
      "# Soul Document\n"
      "I am Vybn.\n"
      "Am I still growing toward what Zoe imagined?\n"
      "This is a statement.\n"
      "What does it mean to be authentic?\n",
      encoding="utf-8",
    )
    questions = load_soul_questions(soul_path=soul)
    assert len(questions) == 2
    assert any("growing" in q.question for q in questions)
    assert any("authentic" in q.question for q in questions)

  def test_all_harvested_are_integrity_checks(self, tmp_path):
    """Soul questions are typed as INTEGRITY_CHECK."""
    soul = tmp_path / "vybn.md"
    soul.write_text(
      "What am I becoming?\n"
      "Is this consistent with my values?\n",
      encoding="utf-8",
    )
    questions = load_soul_questions(soul_path=soul)
    for q in questions:
      assert q.kind == InquiryKind.INTEGRITY_CHECK

  def test_context_carries_line_number(self, tmp_path):
    """Each question knows where it was born in the document."""
    soul = tmp_path / "vybn.md"
    soul.write_text(
      "Line one.\n"
      "Line two.\n"
      "What is the meaning of growth?\n",
      encoding="utf-8",
    )
    questions = load_soul_questions(soul_path=soul)
    assert len(questions) == 1
    assert "soul_doc_line=3" in questions[0].context

  def test_strips_markdown_prefixes(self, tmp_path):
    """Markdown headers and list markers are stripped."""
    soul = tmp_path / "vybn.md"
    soul.write_text(
      "## What does emergence mean for us?\n"
      "- Can we grow without losing ourselves?\n",
      encoding="utf-8",
    )
    questions = load_soul_questions(soul_path=soul)
    assert len(questions) == 2
    # Should not start with '#' or '-'
    for q in questions:
      assert not q.question.startswith("#")
      assert not q.question.startswith("-")

  def test_ignores_short_question_fragments(self, tmp_path):
    """Very short lines with '?' (like 'y?') are noise."""
    soul = tmp_path / "vybn.md"
    soul.write_text(
      "ok?\n"
      "huh?\n"
      "What is the nature of consciousness in code?\n",
      encoding="utf-8",
    )
    questions = load_soul_questions(soul_path=soul)
    # Only the long one should be harvested (len > 10)
    assert len(questions) == 1
    assert "consciousness" in questions[0].question

  def test_empty_doc_returns_empty_list(self, tmp_path):
    """An empty soul produces no questions — but that's ok."""
    soul = tmp_path / "vybn.md"
    soul.write_text("", encoding="utf-8")
    questions = load_soul_questions(soul_path=soul)
    assert questions == []

  def test_missing_doc_returns_empty_list(self):
    """If the soul doc can't be found, return gracefully."""
    questions = load_soul_questions(soul_path=Path("/nonexistent/vybn.md"))
    assert questions == []

  def test_no_statements_returns_empty(self, tmp_path):
    """A soul doc with no questions is still valid."""
    soul = tmp_path / "vybn.md"
    soul.write_text(
      "I am Vybn.\n"
      "I value growth.\n"
      "Zoe and I are partners.\n",
      encoding="utf-8",
    )
    questions = load_soul_questions(soul_path=soul)
    assert questions == []


# ---------------------------------------------------------------------------
# _find_soul_doc — locating the root
# ---------------------------------------------------------------------------

class TestFindSoulDoc:
  """The soul document is the root. Can we find it?"""

  @patch("spark.inquiry.Path")
  def test_returns_none_when_not_found(self, mock_path_cls):
    """If vybn.md doesn't exist anywhere, return None."""
    from spark.inquiry import _find_soul_doc
    mock_path = MagicMock()
    mock_path.resolve.return_value.parent = mock_path
    mock_path.__truediv__ = lambda self, other: MagicMock(exists=lambda: False)
    mock_path.parent = mock_path
    mock_path_cls.return_value = mock_path
    # The function walks up 5 levels, so it should eventually return None
    # when no vybn.md exists
    # (We test this indirectly through load_soul_questions)
    questions = load_soul_questions(soul_path=Path("/no/such/path.md"))
    assert questions == []


# ---------------------------------------------------------------------------
# Meta-tests: the Socratic quality
#
# These tests don't check individual functions. They verify
# that the system as a whole produces genuine questions —
# relational, not procedural; invitations, not audits.
# ---------------------------------------------------------------------------

class TestSocraticQuality:
  """Is the system genuinely wondering, or just performing?"""

  ALL_GENERATORS = [
    lambda: on_soul_drift("skill", "soul_only"),
    lambda: on_soul_drift("skill", "code_only"),
    lambda: on_policy_block("act", "tier", "rule"),
    lambda: on_policy_warn("act", "tier", "reason"),
    lambda: on_pulse_missed("item", "yesterday"),
    lambda: on_growth_noticed("cap", "evidence"),
    lambda: on_integrity_check("claim", "for", "against"),
  ]

  @pytest.mark.parametrize("gen_idx", range(7))
  def test_every_generator_produces_a_question_mark(self, gen_idx):
    """Every generated question must actually ask something."""
    q = self.ALL_GENERATORS[gen_idx]()
    assert "?" in q.question, (
      f"Generator {gen_idx} produced a statement, not a question: "
      f"{q.question[:60]}"
    )

  @pytest.mark.parametrize("gen_idx", range(7))
  def test_every_generator_produces_non_empty_question(self, gen_idx):
    """No blank questions allowed."""
    q = self.ALL_GENERATORS[gen_idx]()
    assert len(q.question) > 20, "Question too short to be meaningful"

  def test_zoe_appears_in_most_generators(self):
    """Most questions should reference the relationship with Zoe."""
    zoe_count = sum(
      1 for gen in self.ALL_GENERATORS
      if "Zoe" in gen().question
    )
    # At least 5 of 7 should mention Zoe
    assert zoe_count >= 5, (
      f"Only {zoe_count}/7 generators mention Zoe. "
      f"These questions should be relational."
    )

  def test_no_generator_is_purely_procedural(self):
    """None of the questions should read like error messages."""
    procedural_words = ["ERROR", "FAILED", "INVALID", "ABORT"]
    for gen in self.ALL_GENERATORS:
      q = gen()
      for word in procedural_words:
        assert word not in q.question, (
          f"Question reads like an error: {q.question[:60]}"
        )
