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


def test_build_continuity_scout_report_is_non_mutating_report():
    from spark.harness.evolve import build_continuity_scout_report
    report = build_continuity_scout_report()
    assert "## Local continuity scout" in report
    assert "Signal counts" in report
    assert "Horizoning questions" in report

def test_mcp_continuity_scout_cli_does_not_require_fastmcp():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "spark.harness.mcp", "--continuity-scout"],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
    assert "## Local continuity scout" in proc.stdout
    assert "Horizoning questions" in proc.stdout
    assert "requires FastMCP" not in proc.stderr

