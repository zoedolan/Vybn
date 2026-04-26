from spark.harness.evolve import _local_continuity_scout


def test_local_continuity_scout_surfaces_horizon_and_self_assembly():
    report = _local_continuity_scout(
        delta_md="horizon horizoning cyberception",
        recent_log="refactor autonomous ensubstrate",
        letter="local Sparks deep_memory dreaming continuity",
    )
    assert "## Local continuity scout" in report
    assert "horizon_sense" in report
    assert "self_assembly" in report
    assert "local_compute" in report
    assert "Strongest local signal" in report
    assert "beam, or has it started pretending to be the horizon" in report
