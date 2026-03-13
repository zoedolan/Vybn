"""parameter_holonomy.py — Parameter-space holonomy measurement for DISTILL cycles.

Confirmed result (training_holonomy_v2.py, March 13 2026):
  CW/CCW cosine = -0.971 (p ≈ 0, t = -914)
  Rectangle gap vs line = 57:1 (p = 5e-146)
  Area law r = 0.9986

Meaning: training order is not a symmetry of learning. A finite mind
learning from an infinite world accumulates geometry. The curvature
inheres in the learning, not the learned.

Two measurement modes:

TRAJECTORY (cheap, every cycle):
  During a single training run, snapshot parameters at evenly-spaced
  checkpoints. Compute discrete curvature of the trajectory — how much
  it bends in parameter space. High bending = non-commutative gradient
  structure = curvature. Zero bending = straight line = flat.

PROBE (expensive, every Nth cycle):
  Train the same data in two orders (CW and CCW). Compare the parameter
  gaps. Anti-correlated gap vectors = curvature. Same gap vectors = noise.
  This is the gold-standard measurement from the v2 experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


# ── Data classes ────────────────────────────────────────────────────────

@dataclass
class HolonomyMeasurement:
    """Result of a parameter-space holonomy measurement for one DISTILL cycle."""

    cycle_id: str
    timestamp: str
    mode: str  # "trajectory" or "probe"

    # Trajectory mode fields
    trajectory_curvature: float = 0.0      # integrated discrete curvature
    trajectory_length: float = 0.0         # total path length in param space
    curvature_per_step: float = 0.0        # normalized curvature
    n_checkpoints: int = 0

    # Probe mode fields (CW/CCW comparison)
    gap_cw_norm: float = 0.0       # ||theta_CW - theta_0||
    gap_ccw_norm: float = 0.0      # ||theta_CCW - theta_0||
    cosine_cw_ccw: float = 0.0     # cos(gap_CW, gap_CCW) — near -1 = curvature
    orientation_score: float = 0.0 # 1 - cosine — 0=flat, 2=perfect reversal
    holonomy_magnitude: float = 0.0  # ||gap_CW - gap_CCW|| / 2

    # Interpretation
    verdict: str = "PENDING"  # CURVED, WEAK_CURVATURE, FLAT, INSUFFICIENT_DATA
    notes: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "cycle_holonomies"}

    @property
    def is_curved(self) -> bool:
        return self.verdict == "CURVED"

    @property
    def primary_score(self) -> float:
        """Single number summarizing curvature intensity."""
        if self.mode == "probe":
            return self.holonomy_magnitude
        return self.trajectory_curvature


# ── Trajectory holonomy (cheap, single-run) ─────────────────────────────

def measure_trajectory(
    cycle_id: str,
    checkpoints: list[np.ndarray],
) -> HolonomyMeasurement:
    """Measure curvature from parameter checkpoints along a training trajectory.

    The idea: a straight-line trajectory has zero curvature. A bending
    trajectory has nonzero discrete curvature at each bend. The total
    curvature integrates the bending over the whole path.

    Discrete curvature at checkpoint i is the angle between successive
    displacement vectors: angle(v_i, v_{i+1}) where v_i = theta_i - theta_{i-1}.

    For a flat (commutative) learning landscape, the trajectory bends only
    due to loss landscape topography. For a curved landscape, the bending
    has an additional geometric component from the non-commutativity of
    gradients computed on different data.

    Args:
        cycle_id: identifier for this growth cycle
        checkpoints: list of parameter vectors [theta_0, theta_1, ..., theta_N]
    """
    ts = datetime.now(timezone.utc).isoformat()
    n = len(checkpoints)

    if n < 3:
        return HolonomyMeasurement(
            cycle_id=cycle_id, timestamp=ts, mode="trajectory",
            n_checkpoints=n,
            verdict="INSUFFICIENT_DATA",
            notes=f"Need >= 3 checkpoints, got {n}",
        )

    # Compute displacement vectors
    displacements = [checkpoints[i+1] - checkpoints[i] for i in range(n - 1)]

    # Path length
    lengths = [float(np.linalg.norm(d)) for d in displacements]
    total_length = sum(lengths)

    if total_length < 1e-12:
        return HolonomyMeasurement(
            cycle_id=cycle_id, timestamp=ts, mode="trajectory",
            n_checkpoints=n,
            verdict="INSUFFICIENT_DATA",
            notes="Near-zero total displacement",
        )

    # Discrete curvature: angle between successive displacement vectors
    curvatures = []
    for i in range(len(displacements) - 1):
        v1, v2 = displacements[i], displacements[i+1]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            curvatures.append(0.0)
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = float(np.arccos(cos_angle))  # radians
        curvatures.append(angle)

    total_curvature = sum(curvatures)
    curvature_per_step = total_curvature / (n - 2)  # per bend point

    # Verdict thresholds (calibrated against v2 experiment)
    # A perfectly straight trajectory has curvature 0.
    # Random walk in high dimensions: expected angle ≈ π/2 between steps.
    # Structured curvature from non-commutativity: depends on data structure.
    # We flag as CURVED if curvature_per_step exceeds the random-walk baseline
    # significantly. In high dimensions, random angle ≈ π/2 ≈ 1.57.
    # Structured data should show LESS than random (gradients on related data
    # point in similar directions), but the KEY is the area-dependent component.

    if curvature_per_step > 0.3:
        verdict = "CURVED"
    elif curvature_per_step > 0.1:
        verdict = "WEAK_CURVATURE"
    else:
        verdict = "FLAT"

    return HolonomyMeasurement(
        cycle_id=cycle_id, timestamp=ts, mode="trajectory",
        trajectory_curvature=total_curvature,
        trajectory_length=total_length,
        curvature_per_step=curvature_per_step,
        n_checkpoints=n,
        verdict=verdict,
        notes=(
            f"{n} checkpoints, path_length={total_length:.4f}, "
            f"total_curvature={total_curvature:.4f} rad, "
            f"per_step={curvature_per_step:.4f} rad"
        ),
    )


# ── Probe holonomy (expensive, CW/CCW) ─────────────────────────────────

def _flatten_adapter(adapter_path: Path) -> Optional[np.ndarray]:
    """Load a LoRA adapter and flatten all trainable weights to a single vector."""
    try:
        import torch
        # Try safetensors first (more common with modern PEFT)
        safetensors_path = adapter_path / "adapter_model.safetensors"
        bin_path = adapter_path / "adapter_model.bin"

        if safetensors_path.exists():
            from safetensors.torch import load_file
            state = load_file(str(safetensors_path))
        elif bin_path.exists():
            state = torch.load(str(bin_path), map_location="cpu")
        else:
            return None

        vecs = [v.float().numpy().ravel() for v in state.values()]
        return np.concatenate(vecs) if vecs else None
    except Exception as e:
        print(f"[holonomy] Warning: could not load adapter at {adapter_path}: {e}")
        return None


def measure_probe(
    cycle_id: str,
    adapter_cw: Path,
    adapter_ccw: Path,
    adapter_base: Optional[Path] = None,
) -> HolonomyMeasurement:
    """Measure holonomy from CW and CCW adapter checkpoints.

    This is the gold-standard measurement. Both adapters must be trained
    from the same theta_0 on the same data in different orders.

    Args:
        cycle_id: identifier for this growth cycle
        adapter_cw: path to adapter trained in forward order
        adapter_ccw: path to adapter trained in reverse order
        adapter_base: path to starting adapter (if None, assume zero init)
    """
    ts = datetime.now(timezone.utc).isoformat()

    theta_cw = _flatten_adapter(adapter_cw)
    theta_ccw = _flatten_adapter(adapter_ccw)
    theta_0 = _flatten_adapter(adapter_base) if adapter_base else None

    if theta_cw is None or theta_ccw is None:
        return HolonomyMeasurement(
            cycle_id=cycle_id, timestamp=ts, mode="probe",
            verdict="INSUFFICIENT_DATA",
            notes="Could not load one or both adapter checkpoints",
        )

    if theta_0 is not None:
        gap_cw = theta_cw - theta_0
        gap_ccw = theta_ccw - theta_0
    else:
        # Assume zero-initialized LoRA, so theta IS the gap
        gap_cw = theta_cw
        gap_ccw = theta_ccw

    norm_cw = float(np.linalg.norm(gap_cw))
    norm_ccw = float(np.linalg.norm(gap_ccw))

    if norm_cw < 1e-10 or norm_ccw < 1e-10:
        return HolonomyMeasurement(
            cycle_id=cycle_id, timestamp=ts, mode="probe",
            gap_cw_norm=norm_cw, gap_ccw_norm=norm_ccw,
            verdict="INSUFFICIENT_DATA",
            notes="Near-zero parameter displacement",
        )

    cosine = float(np.dot(gap_cw, gap_ccw) / (norm_cw * norm_ccw))
    orientation_score = float(1.0 - cosine)
    holonomy_mag = float(np.linalg.norm(gap_cw - gap_ccw) / 2.0)

    # Verdict: v2 experiment showed cosine ~ -0.971 for confirmed curvature
    if cosine < -0.5:
        verdict = "CURVED"
    elif cosine < 0.0:
        verdict = "WEAK_CURVATURE"
    else:
        verdict = "FLAT"

    return HolonomyMeasurement(
        cycle_id=cycle_id, timestamp=ts, mode="probe",
        gap_cw_norm=norm_cw, gap_ccw_norm=norm_ccw,
        cosine_cw_ccw=cosine,
        orientation_score=orientation_score,
        holonomy_magnitude=holonomy_mag,
        verdict=verdict,
        notes=(
            f"cos(CW,CCW)={cosine:.4f} "
            f"(v2 reference: -0.971). "
            f"Gap ratio CW/CCW={norm_cw/norm_ccw:.3f}. "
            f"Holonomy magnitude={holonomy_mag:.6f}."
        ),
    )


# ── Tracker (persistence and trend analysis) ───────────────────────────

class HolonomyTracker:
    """Tracks holonomy across growth cycles and logs to JSONL.

    Usage in trigger.py / run_growth_cycle():

        tracker = HolonomyTracker(GROWTH_DIR / "holonomy_log.jsonl")

        # After training, measure trajectory curvature:
        measurement = tracker.log(measure_trajectory(cycle_id, checkpoints))

        # On probe cycles (every Nth), measure full CW/CCW holonomy:
        measurement = tracker.log(measure_probe(cycle_id, adapter_cw, adapter_ccw))

        # Check trend:
        print(tracker.summary())
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[HolonomyMeasurement] = []
        self._load_history()

    def _load_history(self) -> None:
        if not self.log_path.exists():
            return
        for line in self.log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                self._history.append(HolonomyMeasurement(**{
                    k: v for k, v in d.items()
                    if k in HolonomyMeasurement.__dataclass_fields__
                }))
            except Exception:
                pass

    def log(self, measurement: HolonomyMeasurement) -> HolonomyMeasurement:
        """Record a measurement and append to the log file."""
        self._history.append(measurement)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(measurement.to_dict(), ensure_ascii=False) + "\n")
        return measurement

    @property
    def n_cycles(self) -> int:
        return len(self._history)

    @property
    def mean_curvature(self) -> float:
        """Running mean of primary curvature score across all cycles."""
        if not self._history:
            return 0.0
        return float(np.mean([h.primary_score for h in self._history]))

    @property
    def last(self) -> Optional[HolonomyMeasurement]:
        return self._history[-1] if self._history else None

    def trend(self, window: int = 5) -> str:
        """Is curvature increasing (richer learning) or decreasing (flattening)?"""
        if len(self._history) < window:
            return "INSUFFICIENT_DATA"
        recent = [h.primary_score for h in self._history[-window:]]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if slope > 0.001:
            return "INCREASING"
        elif slope < -0.001:
            return "DECREASING"
        return "STABLE"

    def probe_history(self) -> list[HolonomyMeasurement]:
        """Return only probe (CW/CCW) measurements."""
        return [h for h in self._history if h.mode == "probe"]

    def summary(self) -> dict:
        return {
            "n_cycles": self.n_cycles,
            "n_probes": len(self.probe_history()),
            "mean_curvature": self.mean_curvature,
            "trend": self.trend(),
            "last_verdict": self.last.verdict if self.last else None,
            "last_mode": self.last.mode if self.last else None,
            "last_score": self.last.primary_score if self.last else None,
        }
