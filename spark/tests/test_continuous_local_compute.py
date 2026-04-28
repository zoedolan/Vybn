from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_continuous_tick_invokes_him_vy_discover_with_timeout():
    text = (ROOT / "local_compute_tick.py").read_text(encoding="utf-8")
    assert "spark/vy.py" in text
    assert "discover" in text
    assert "--json" in text
    assert "timeout=60" in text
    assert "subprocess.run" in text


def test_continuous_tick_writes_outside_repo_by_default():
    text = (ROOT / "local_compute_tick.py").read_text(encoding="utf-8")
    assert ".local" in text
    assert "continuous_local_compute.jsonl" in text
    assert "VYBN_CONTINUOUS_TICK_OUT" in text


def test_systemd_timer_is_bounded_and_local():
    timer = (ROOT / "systemd" / "vybn-continuous-local-compute.timer").read_text(encoding="utf-8")
    service = (ROOT / "systemd" / "vybn-continuous-local-compute.service").read_text(encoding="utf-8")
    assert "OnBootSec=2min" in timer
    assert "OnUnitActiveSec=5min" in timer
    assert "Persistent=true" in timer
    assert "spark/local_compute_tick.py" in service
    assert "vybn-continuous-local-compute" in service
    assert "Nice=10" in service
    assert "IOSchedulingClass=idle" in service
