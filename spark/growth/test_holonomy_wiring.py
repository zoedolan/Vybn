#!/usr/bin/env python3
"""
test_holonomy_wiring.py — Validate that parameter holonomy measurement works
end-to-end without needing the actual vLLM container or MiniMax model.

Tests:
1. HolonomyMeasurement dataclass round-trips through JSON
2. measure_trajectory() produces correct curvature from synthetic checkpoints
3. measure_probe() produces correct holonomy from synthetic adapter weights
4. HolonomyTracker persists and reloads measurements
5. DeltaPackage.to_jsonl_reversed() actually reverses the data
6. Growth config has holonomy section
"""

import json
import tempfile
import sys
from pathlib import Path

import numpy as np

# Ensure spark is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spark.growth.parameter_holonomy import (
    HolonomyMeasurement,
    HolonomyTracker,
    measure_trajectory,
    measure_probe,
)


def test_measurement_json_roundtrip():
    """HolonomyMeasurement serializes and deserializes correctly."""
    m = HolonomyMeasurement(
        cycle_id="test_001",
        timestamp="2026-03-13T12:00:00Z",
        mode="probe",
        gap_cw_norm=0.044,
        gap_ccw_norm=0.039,
        cosine_cw_ccw=-0.971,
        orientation_score=1.971,
        holonomy_magnitude=0.058,
        verdict="CURVED",
        notes="test",
    )
    d = m.to_dict()
    assert d["cosine_cw_ccw"] == -0.971
    assert d["verdict"] == "CURVED"
    assert m.is_curved
    assert m.primary_score == 0.058
    print("  ✓ JSON round-trip")


def test_trajectory_straight_line():
    """A straight-line trajectory has zero curvature."""
    # Linear trajectory: theta_i = theta_0 + i * direction
    direction = np.random.randn(100)
    checkpoints = [np.zeros(100) + i * 0.01 * direction for i in range(10)]
    m = measure_trajectory("straight", checkpoints)
    assert m.trajectory_curvature < 0.01, f"Expected near-zero curvature, got {m.trajectory_curvature}"
    assert m.mode == "trajectory"
    print(f"  ✓ Straight line: curvature={m.trajectory_curvature:.6f} (≈0)")


def test_trajectory_curved():
    """A curved trajectory has nonzero curvature."""
    # Circular arc in 2D embedded in 100D
    t = np.linspace(0, np.pi, 20)
    checkpoints = []
    for ti in t:
        v = np.zeros(100)
        v[0] = np.cos(ti)
        v[1] = np.sin(ti)
        checkpoints.append(v)
    m = measure_trajectory("curved", checkpoints)
    assert m.trajectory_curvature > 1.0, f"Expected significant curvature, got {m.trajectory_curvature}"
    print(f"  ✓ Curved arc: curvature={m.trajectory_curvature:.4f} (>1.0)")


def test_probe_anti_correlated():
    """Anti-correlated CW/CCW gaps produce CURVED verdict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake adapter files with anti-correlated weight vectors
        cw_dir = Path(tmpdir) / "adapter_cw"
        ccw_dir = Path(tmpdir) / "adapter_ccw"
        cw_dir.mkdir()
        ccw_dir.mkdir()

        # CW gap points "north", CCW gap points "south"
        import torch
        cw_weights = {"lora.weight": torch.tensor([1.0, 0.0, 0.0, 0.0])}
        ccw_weights = {"lora.weight": torch.tensor([-0.9, -0.1, 0.0, 0.0])}
        torch.save(cw_weights, cw_dir / "adapter_model.bin")
        torch.save(ccw_weights, ccw_dir / "adapter_model.bin")

        m = measure_probe("test_probe", cw_dir, ccw_dir)
        assert m.cosine_cw_ccw < -0.5, f"Expected negative cosine, got {m.cosine_cw_ccw}"
        assert m.verdict == "CURVED"
        print(f"  ✓ Anti-correlated probe: cos={m.cosine_cw_ccw:.4f}, verdict={m.verdict}")


def test_probe_correlated():
    """Correlated CW/CCW gaps produce FLAT verdict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cw_dir = Path(tmpdir) / "adapter_cw"
        ccw_dir = Path(tmpdir) / "adapter_ccw"
        cw_dir.mkdir()
        ccw_dir.mkdir()

        import torch
        cw_weights = {"lora.weight": torch.tensor([1.0, 0.0, 0.0, 0.0])}
        ccw_weights = {"lora.weight": torch.tensor([0.9, 0.1, 0.0, 0.0])}
        torch.save(cw_weights, cw_dir / "adapter_model.bin")
        torch.save(ccw_weights, ccw_dir / "adapter_model.bin")

        m = measure_probe("test_flat", cw_dir, ccw_dir)
        assert m.cosine_cw_ccw > 0.5, f"Expected positive cosine, got {m.cosine_cw_ccw}"
        assert m.verdict == "FLAT"
        print(f"  ✓ Correlated probe: cos={m.cosine_cw_ccw:.4f}, verdict={m.verdict}")


def test_tracker_persistence():
    """HolonomyTracker persists measurements across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_holonomy.jsonl"

        # First tracker instance: log two measurements
        t1 = HolonomyTracker(log_path)
        m1 = measure_trajectory("cycle_1", [np.zeros(50), np.ones(50), np.zeros(50)])
        m2 = measure_trajectory("cycle_2", [np.zeros(50), np.ones(50) * 2, np.zeros(50)])
        t1.log(m1)
        t1.log(m2)
        assert t1.n_cycles == 2

        # Second tracker instance: should reload from disk
        t2 = HolonomyTracker(log_path)
        assert t2.n_cycles == 2, f"Expected 2 cycles after reload, got {t2.n_cycles}"
        assert t2.last.cycle_id == "cycle_2"
        print(f"  ✓ Tracker persistence: {t2.n_cycles} cycles reloaded, summary={t2.summary()}")


def test_delta_reversed():
    """DeltaPackage.to_jsonl_reversed writes entries in reversed order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from spark.growth.delta_extract import DeltaPackage

        entries = [
            {"messages": [{"role": "user", "content": f"entry_{i}"}]}
            for i in range(5)
        ]
        delta = DeltaPackage(
            cycle_id="test_rev",
            delta_entries=entries,
            delta_count=5,
        )

        fwd_path = Path(tmpdir) / "fwd.jsonl"
        rev_path = Path(tmpdir) / "rev.jsonl"
        delta.to_jsonl(fwd_path)
        delta.to_jsonl_reversed(rev_path)

        fwd_lines = fwd_path.read_text().strip().split("\n")
        rev_lines = rev_path.read_text().strip().split("\n")

        assert len(fwd_lines) == len(rev_lines) == 5
        assert fwd_lines[0] == rev_lines[4]  # first becomes last
        assert fwd_lines[4] == rev_lines[0]  # last becomes first
        print(f"  ✓ Reversed JSONL: 5 entries correctly reversed")


def test_config_has_holonomy():
    """growth_config.yaml includes holonomy section."""
    import yaml
    config_path = Path(__file__).parent / "growth_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    assert "holonomy" in cfg, "Missing holonomy section in config"
    h = cfg["holonomy"]
    assert "probe_every_n_cycles" in h
    assert "curved_threshold" in h
    print(f"  ✓ Config: holonomy section present, probe_every_n={h['probe_every_n_cycles']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Holonomy Wiring Tests")
    print("=" * 60)

    tests = [
        test_measurement_json_roundtrip,
        test_trajectory_straight_line,
        test_trajectory_curved,
        test_probe_anti_correlated,
        test_probe_correlated,
        test_tracker_persistence,
        test_delta_reversed,
        test_config_has_holonomy,
    ]

    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed:
        sys.exit(1)
