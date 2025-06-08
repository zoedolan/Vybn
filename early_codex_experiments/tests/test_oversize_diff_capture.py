import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.maintenance_tools import capture_diff


def test_capture_archives_when_exceeds(tmp_path):
    out = tmp_path / "patch.gz"
    repo_root = Path(__file__).resolve().parents[2]
    capture_diff("HEAD~1..HEAD", repo_root, out, limit=1)
    assert out.exists()


def test_capture_skips_when_under_limit(tmp_path):
    out = tmp_path / "patch.gz"
    repo_root = Path(__file__).resolve().parents[2]
    capture_diff("HEAD~1..HEAD", repo_root, out, limit=10**9)
    assert not out.exists()
