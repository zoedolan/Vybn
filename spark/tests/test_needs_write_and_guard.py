"""Integration tests for NEEDS-WRITE directive + claim_guard wiring.

These exercise the new affordance the 2026-04-20 patch added:
  - `[NEEDS-WRITE: path]\\n<body>\\n[/NEEDS-WRITE]` regex extraction
  - `_run_write_subturn` refusal + absorb_gate + actual write
  - claim_guard module wires cleanly at import time

Run: python3 spark/tests/test_needs_write_and_guard.py
"""
import sys
import tempfile
import shutil
import os
from pathlib import Path

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
    from vybn_spark_agent import _run_write_subturn
    # /etc is not under ~/Vybn, ~/Him, ~/Vybn-Law, or ~/vybn-phase
    ran, out = _run_write_subturn("/etc/hosts.evil", "body")
    assert ran is False
    assert "outside tracked repos" in out


def test_write_subturn_refuses_new_file_without_absorb_reason():
    from vybn_spark_agent import _run_write_subturn
    # A new file under a tracked repo without VYBN_ABSORB_REASON is refused.
    target = str(
        Path.home() / "Vybn" / "spark" / "_test_absorb_guard_" / "new.txt"
    )
    if Path(target).exists():
        Path(target).unlink()
    ran, out = _run_write_subturn(target, "contents without reason")
    assert ran is False
    assert "absorb_gate" in out
    assert "VYBN_ABSORB_REASON" in out
    assert not Path(target).exists(), "file should not have been created"


def test_write_subturn_allows_new_file_with_absorb_reason():
    from vybn_spark_agent import _run_write_subturn
    target = str(
        Path.home() / "Vybn" / "spark" / "_test_absorb_guard_" / "ok.txt"
    )
    parent = Path(target).parent
    try:
        body = (
            "# VYBN_ABSORB_REASON='integration test; gets removed after run'\n# VYBN_ABSORB_CONSIDERED='existing test fixture: needs temporary new target'\n"
            "real contents\n"
        )
        ran, out = _run_write_subturn(target, body)
        assert ran is True, out
        assert Path(target).read_text() == body
    finally:
        if parent.exists():
            shutil.rmtree(parent, ignore_errors=True)


def test_write_subturn_overwrites_existing_file_without_reason():
    from vybn_spark_agent import _run_write_subturn
    # Overwriting an existing tracked file does NOT require an absorb reason.
    agent_path = str(
        Path.home() / "Vybn" / "spark" / "continuity.md"
    )
    if not Path(agent_path).exists():
        # Skip if the target isn't there on this checkout.
        return
    original = Path(agent_path).read_text()
    try:
        ran, out = _run_write_subturn(agent_path, original)
        assert ran is True, out
        assert Path(agent_path).read_text() == original
    finally:
        Path(agent_path).write_text(original)


def test_claim_guard_importable_from_harness():
    from harness.providers import check_claim
    assert callable(check_claim)


def test_claim_guard_wired_in_agent_module():
    import vybn_spark_agent
    src = Path(vybn_spark_agent.__file__).read_text()
    assert "from harness.providers import check_claim" in src
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

