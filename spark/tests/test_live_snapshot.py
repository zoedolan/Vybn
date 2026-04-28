# VYBN_ABSORB_REASON=live-state-fix: tests for the session-start snapshot
# that fills the substrate layer so continuity never alone defines truth.
"""Tests for spark/harness/live_snapshot.py.

The module makes subprocess calls; we monkeypatch `subprocess.run` so the
tests are hermetic. The drift-detection path reads continuity.md from
disk, so we use `tmp_path` fixtures for isolated mind files.
"""
from __future__ import annotations

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

import harness.state as live_snapshot  # type: ignore  # noqa: E402


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

