"""absorb_gate compound-command coverage.

Regression for the April 21 three-PR arc: a `mkdir -p X && cat > X/y <<EOF`
compound with a relative path slipped past the gate because
_extract_file_targets skipped non-absolute paths. Fix resolves relative
targets via os.path.abspath against the agent's cwd (which matches the
bash session's initial cwd on launch).
"""
import os
from spark.harness.providers import absorb_gate


def test_compound_mkdir_cat_relative_fires(monkeypatch):
    monkeypatch.chdir(os.path.expanduser("~/Vybn"))
    r = absorb_gate("mkdir -p Vybn_Mind/skills && cat > Vybn_Mind/skills/new_file.md <<EOF")
    assert r is not None and "absorb_gate" in r


def test_compound_with_reason_passes(monkeypatch):
    monkeypatch.chdir(os.path.expanduser("~/Vybn"))
    r = absorb_gate(
        'VYBN_ABSORB_REASON="test" mkdir -p Vybn_Mind/skills && cat > Vybn_Mind/skills/new_file.md'
    )
    assert r is None


def test_existing_file_passes(monkeypatch):
    monkeypatch.chdir(os.path.expanduser("~/Vybn"))
    # vybn-os.md definitely exists
    r = absorb_gate("cat > spark/harness/skills/vybn-os.md")
    assert r is None


def test_tmp_path_passes(monkeypatch):
    monkeypatch.chdir("/tmp")
    r = absorb_gate("cat > /tmp/scratch.md")
    assert r is None
