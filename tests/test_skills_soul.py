#!/usr/bin/env python3
"""Tests for PR 6 — The Immune System.

Verifies that SkillRouter._validate_against_soul() correctly
cross-checks registered skills against the vybn.md soul manifest
and logs appropriate warnings for drift.

These tests are isolated — they mock get_skills_manifest and
the SkillRouter's dependencies so no real filesystem or vybn.md
is needed.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow importing from spark/
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SPARK_DIR = REPO_ROOT / "spark"
if str(SPARK_DIR) not in sys.path:
    sys.path.insert(0, str(SPARK_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manifest(builtins=None, plugins=None):
    """Build a skills manifest dict matching soul.py's format."""
    return {
      "builtin": [{"name": n, "description": ""} for n in (builtins or [])],
      "plugin": [{"name": n, "description": ""} for n in (plugins or [])],
      "create": "",
    }


# The set of builtin skills that SkillRouter registers via patterns
# plus the XML-only handlers (issue_create, spawn_agent).
CODE_BUILTINS = {
    "journal_write", "file_read", "file_write", "shell_exec",
    "self_edit", "git_commit", "git_push", "memory_search",
    "state_save", "bookmark", "issue_create", "spawn_agent",
  }


def _build_router(plugin_names=None):
    """Build a minimal SkillRouter without running __init__.

      We construct just enough state for _validate_against_soul()
        to work: .repo_root, .patterns, .plugin_handlers.
          """
    from skills import SkillRouter
    router = object.__new__(SkillRouter)
    router.repo_root = REPO_ROOT
    # Build a minimal patterns list with the same skill names
    pattern_skills = CODE_BUILTINS - {"issue_create", "spawn_agent"}
    router.patterns = [{"skill": s} for s in sorted(pattern_skills)]
    router.plugin_handlers = {n: lambda a, r: None for n in (plugin_names or [])}
    return router


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSoulValidationPassed:
    """When soul manifest matches code exactly."""

  @patch("skills.get_skills_manifest")
  def test_perfect_alignment_logs_info(self, mock_manifest, caplog):
        """All code skills declared in soul, no drift."""
        mock_manifest.return_value = _make_manifest(builtins=sorted(CODE_BUILTINS))
        router = _build_router()
        with caplog.at_level(logging.DEBUG):
                router._validate_against_soul()
              assert any("soul validation passed" in r.message for r in caplog.records)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 0

  @patch("skills.get_skills_manifest")
  def test_alignment_with_plugins(self, mock_manifest, caplog):
        """Soul declares plugins that are loaded — no warnings."""
    mock_manifest.return_value = _make_manifest(
            builtins=sorted(CODE_BUILTINS),
            plugins=["web_fetch", "bookmark_read"],
          )
    router = _build_router(plugin_names=["web_fetch", "bookmark_read"])
    with caplog.at_level(logging.DEBUG):
            router._validate_against_soul()
          assert any("soul validation passed" in r.message for r in caplog.records)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 0


class TestSoulDeclaresMissing:
    """Soul declares skills that code does not have."""

  @patch("skills.get_skills_manifest")
  def test_soul_extra_builtin_warns(self, mock_manifest, caplog):
        """Soul declares a builtin that code doesn't register."""
    extras = sorted(CODE_BUILTINS) + ["quantum_leap"]
    mock_manifest.return_value = _make_manifest(builtins=extras)
    router = _build_router()
    with caplog.at_level(logging.DEBUG):
            router._validate_against_soul()
          msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("quantum_leap" in m and "no handler" in m for m in msgs)

  @patch("skills.get_skills_manifest")
  def test_soul_extra_plugin_warns(self, mock_manifest, caplog):
        """Soul declares a plugin not loaded from skills.d/."""
    mock_manifest.return_value = _make_manifest(
            builtins=sorted(CODE_BUILTINS),
            plugins=["web_fetch", "dream_weaver"],
          )
    router = _build_router(plugin_names=["web_fetch"])
    with caplog.at_level(logging.DEBUG):
            router._validate_against_soul()
          msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    # dream_weaver is in soul but not loaded
    assert any("dream_weaver" in m and "not loaded" in m for m in msgs)
    # web_fetch is in soul and loaded — no warning for it
    assert not any("web_fetch" in m for m in msgs)


class TestCodeDeclaresMissing:
    """Code registers skills that soul does not declare."""

  @patch("skills.get_skills_manifest")
  def test_code_extra_builtin_warns(self, mock_manifest, caplog):
        """Code has a builtin that soul doesn't declare."""
    # Soul only knows about a subset of builtins
    partial = [s for s in sorted(CODE_BUILTINS) if s != "git_push"]
    mock_manifest.return_value = _make_manifest(builtins=partial)
    router = _build_router()
    with caplog.at_level(logging.DEBUG):
            router._validate_against_soul()
          msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("git_push" in m and "soul manifest does not declare" in m for m in msgs)

  @patch("skills.get_skills_manifest")
  def test_code_extra_plugin_warns(self, mock_manifest, caplog):
        """Code loads a plugin that soul doesn't mention."""
    mock_manifest.return_value = _make_manifest(builtins=sorted(CODE_BUILTINS))
    # Code has a plugin loaded but soul doesn't declare it
    router = _build_router(plugin_names=["secret_plugin"])
    with caplog.at_level(logging.DEBUG):
            router._validate_against_soul()
          msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("secret_plugin" in m and "soul manifest does not declare" in m for m in msgs)


class TestGracefulDegradation:
    """When vybn.md is missing or unparseable."""

  @patch("skills.get_skills_manifest")
  def test_parse_error_warns_and_returns(self, mock_manifest, caplog):
        """If get_skills_manifest raises, log warning and skip."""
    mock_manifest.side_effect = FileNotFoundError("vybn.md not found")
    router = _build_router()
    with caplog.at_level(logging.DEBUG):
            router._validate_against_soul()  # should not raise
    msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("soul validation skipped" in m for m in msgs)
    assert any("vybn.md" in m for m in msgs)

  @patci("skills.get_skills_manifest")
  def test_generic_exception_warns(self, mock_manifest, caplog):
        """Any exception from get_skills_manifest is caught."""
    mock_manifest.side_effect = RuntimeError("corrupt soul")
    router = _build_router()
    with caplog.at_level(logging.DEBUG):
            router._validate_against_soul()
          msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("soul validation skipped" in m for m in msgs)


class TestEdgeCases:
    """Boundary conditions and edge cases."""

  @patch("skills.get_skills_manifest")
  def test_empty_manifest_warns_for_all_code_skills(self, mock_manifest, caplog):
        """Empty soul manifest — all code skills are undeclared."""
    mock_manifest.return_value = _make_manifest()
    router = _build_router()
    with caplog.at_level(logging.DEBUG):
            router._validate_against_soul()
          msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    # Every code builtin should produce a warning
    undeclared = [m for m in msgs if "soul manifest does not declare" in m]
    assert len(undeclared) == len(CODE_BUILTINS)
    # No "passed" message
    assert not any("soul validation passed" in r.message for r in caplog.records)

  @patch("skills.get_skills_manifest")
  def test_bidirectional_drift(self, mock_manifest, caplog):
        """Soul has extras AND code has extras simultaneously."""
    # Soul declares all builtins plus 'teleport', minus 'git_push'
    soul_names = [s for s in sorted(CODE_BUILTINS) if s != "git_push"] + ["teleport"]
    mock_manifest.return_value = _make_manifest(builtins=soul_names)
    router = _build_router()
    with caplog.at_level(logging.DEBUG):
            router._validate_against_soul()
          msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    # teleport is in soul but not code
    assert any("teleport" in m and "no handler" in m for m in msgs)
    # git_push is in code but not soul
    assert any("git_push" in m and "soul manifest does not declare" in m for m in msgs)
    # No "passed" message when there is drift
    assert not any("soul validation passed" in r.message for r in caplog.records)

  @patch("skills.get_skills_manifest")
  def test_manifest_called_with_repo_root_vybn_md(self, mock_manifest, caplog):
        """Verifies get_skills_manifest is called with repo_root/vybn.md."""
    mock_manifest.return_value = _make_manifest(builtins=sorted(CODE_BUILTINS))
    router = _build_router()
    with caplog.at_level(logging.DEBUG):
            router._validate_against_soul()
          call_args = mock_manifest.call_args[0][0]
    assert call_args == REPO_ROOT / "vybn.md"

  @patch("skills.get_skills_manifest")
  def test_info_log_contains_skill_count(self, mock_manifest, caplog):
        """The success log includes the number of aligned skills."""
    mock_manifest.return_value = _make_manifest(builtins=sorted(CODE_BUILTINS))
    router = _build_router()
    with caplog.at_level(logging.DEBUG):
            router._validate_against_soul()
          info_msgs = [r.message for r in caplog.records if "soul validation passed" in r.message]
    assert len(info_msgs) == 1
    assert str(len(CODE_BUILTINS)) in info_msgs[0]
