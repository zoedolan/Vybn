from pathlib import Path

from spark.harness.abc_consolidator import build_tick


def test_abc_consolidator_builds_candidate_tick_for_repo():
    tick = build_tick(Path("."), top_n=12)
    assert tick["status"] in {"candidate", "no_candidate"}
    assert "visualization" in tick
    if tick["status"] == "candidate":
        assert tick["candidate"]
        assert "adaptive_plan" in tick
        assert "packet" in tick
        assert "restore" in str(tick).lower()
