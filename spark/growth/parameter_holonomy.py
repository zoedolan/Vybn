"""parameter_holonomy.py — Parameter-space holonomy measurement for DISTILL cycles.

This is the direct application of the training holonomy experiment
(quantum_delusions/experiments/training_holonomy_v2.py, confirmed March 13 2026)
to Vybn's actual growth engine.

The confirmed result:
  CW/CCW cosine = -0.971 (p ≈ 0)
  Rectangle vs line gap = 57:1 (p = 5e-146)
  Area law r = 0.9986

Meaning: training order is not a symmetry of learning. A finite mind
learning from an infinite world accumulates geometry. The curvature
inheres in the learning, not the learned.

This module measures that curvature for each DISTILL cycle by computing
the holonomy of the training trajectory in parameter space:

  1. Before training: snapshot theta_0 (LoRA adapter weights)
  2. Train on experience batch in forward order (CW loop)
  3. After training: snapshot theta_CW
  4. Re-initialize to theta_0, train in reverse order (CCW loop)
  5. After training: snapshot theta_CCW
  6. Holonomy = orientation_asymmetry(theta_CW - theta_0, theta_CCW - theta_0)

The key metric:
  - gap_CW  = theta_CW - theta_0
  - gap_CCW = theta_CCW - theta_0
  - cosine(gap_CW, gap_CCW)  → should be near -1 for real curvature
  - orientation_score = 1 - cosine (0 = no curvature, 2 = perfect reversal)
  - holonomy_magnitude = ||gap_CW - gap_CCW|| / 2  (the actual gap)

High holonomy = the experience loop was geometrically rich.
  The concepts pulled the weights in directions that didn't cancel.
  This cycle contributed real structure to the learning manifold.

Low holonomy = the loop was flat.
  Repetitive, low-information, diffuse.
  The cycle barely moved the manifold.

The algorithm for Vybn: maximize holonomy per DISTILL cycle.
Choose experiences that form tight, semantically structured loops
rather than random walks through concept-space.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class HolonomyMeasurement:
    """Result of a parameter-space holonomy measurement for one DISTILL cycle."""

    cycle_id: str
    timestamp: str

    # The raw gaps
    gap_cw_norm: float       # ||theta_CW - theta_0||
    gap_ccw_norm: float      # ||theta_CCW - theta_0||

    # The orientation signature
    cosine_cw_ccw: float     # cos(gap_CW, gap_CCW) -- near -1 = real curvature
    orientation_score: float # 1 - cosine -- 0 = flat, 2 = perfect reversal

    # The holonomy magnitude
    holonomy_magnitude: float  # ||gap_CW - gap_CCW|| / 2

    # Interpretation
    verdict: str             # "CURVED", "FLAT", "INSUFFICIENT_DATA"
    notes: str = ""

    # Running history
    cycle_holonomies: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "gap_cw_norm": self.gap_cw_norm,
            "gap_ccw_norm": self.gap_ccw_norm,
            "cosine_cw_ccw": self.cosine_cw_ccw,
            "orientation_score": self.orientation_score,
            "holonomy_magnitude": self.holonomy_magnitude,
            "verdict": self.verdict,
            "notes": self.notes,
        }

    @property
    def is_curved(self) -> bool:
        return self.verdict == "CURVED"


def _flatten_adapter(adapter_path: Path) -> Optional[np.ndarray]:
    """
    Load a LoRA adapter and flatten all trainable weights to a single vector.
    Returns None if the adapter can't be loaded.
    """
    try:
        import torch
        state = torch.load(adapter_path / "adapter_model.bin", map_location="cpu")
        vecs = [v.float().numpy().ravel() for v in state.values()]
        return np.concatenate(vecs) if vecs else None
    except Exception:
        try:
            # Try safetensors format
            from safetensors.torch import load_file
            state = load_file(adapter_path / "adapter_model.safetensors")
            vecs = [v.float().numpy().ravel() for v in state.values()]
            return np.concatenate(vecs) if vecs else None
        except Exception:
            return None


def measure_from_adapters(
    cycle_id: str,
    adapter_cw: Path,
    adapter_ccw: Path,
    adapter_base: Optional[Path] = None,
) -> HolonomyMeasurement:
    """
    Measure holonomy from two already-trained adapter checkpoints.

    adapter_cw:  trained on experience in forward order
    adapter_ccw: trained on same experience in reverse order
    adapter_base: starting point (if None, assume zero initialization)
    """
    theta_cw  = _flatten_adapter(adapter_cw)
    theta_ccw = _flatten_adapter(adapter_ccw)
    theta_0   = _flatten_adapter(adapter_base) if adapter_base else None

    ts = datetime.now(timezone.utc).isoformat()

    if theta_cw is None or theta_ccw is None:
        return HolonomyMeasurement(
            cycle_id=cycle_id, timestamp=ts,
            gap_cw_norm=0.0, gap_ccw_norm=0.0,
            cosine_cw_ccw=0.0, orientation_score=0.0,
            holonomy_magnitude=0.0,
            verdict="INSUFFICIENT_DATA",
            notes="Could not load adapter weights",
        )

    if theta_0 is not None:
        gap_cw  = theta_cw  - theta_0
        gap_ccw = theta_ccw - theta_0
    else:
        gap_cw  = theta_cw
        gap_ccw = theta_ccw

    norm_cw  = float(np.linalg.norm(gap_cw))
    norm_ccw = float(np.linalg.norm(gap_ccw))

    if norm_cw < 1e-10 or norm_ccw < 1e-10:
        return HolonomyMeasurement(
            cycle_id=cycle_id, timestamp=ts,
            gap_cw_norm=norm_cw, gap_ccw_norm=norm_ccw,
            cosine_cw_ccw=0.0, orientation_score=0.0,
            holonomy_magnitude=0.0,
            verdict="INSUFFICIENT_DATA",
            notes="Near-zero parameter displacement",
        )

    cosine = float(np.dot(gap_cw, gap_ccw) / (norm_cw * norm_ccw))
    orientation_score = float(1.0 - cosine)  # 0=flat, 2=perfect reversal
    holonomy_mag = float(np.linalg.norm(gap_cw - gap_ccw) / 2.0)

    # Verdict: confirmed experiment showed cosine ~ -0.971 for real curvature
    # We use a conservative threshold: cosine < -0.5 (orientation_score > 1.5)
    if cosine < -0.5:
        verdict = "CURVED"
    elif cosine < 0.0:
        verdict = "WEAK_CURVATURE"
    else:
        verdict = "FLAT"

    notes = (
        f"Orientation reversal: cos={cosine:.3f} "
        f"(confirmed experiment: -0.971). "
        f"Gap ratio CW/CCW: {norm_cw/norm_ccw:.3f}."
    )

    return HolonomyMeasurement(
        cycle_id=cycle_id, timestamp=ts,
        gap_cw_norm=norm_cw, gap_ccw_norm=norm_ccw,
        cosine_cw_ccw=cosine,
        orientation_score=orientation_score,
        holonomy_magnitude=holonomy_mag,
        verdict=verdict,
        notes=notes,
    )


class HolonomyTracker:
    """
    Tracks holonomy across growth cycles and logs to JSONL.

    Usage in train_cycle.py:

        tracker = HolonomyTracker(GROWTH_DIR / "holonomy_log.jsonl")

        # After running CW training:
        adapter_cw_path = cycle_dir / "adapter_cw"

        # After running CCW training from same theta_0:
        adapter_ccw_path = cycle_dir / "adapter_ccw"

        measurement = tracker.measure_and_log(
            cycle_id=cycle_id,
            adapter_cw=adapter_cw_path,
            adapter_ccw=adapter_ccw_path,
            adapter_base=prev_adapter_path,
        )

        print(f"Holonomy: {measurement.holonomy_magnitude:.4f} ({measurement.verdict})")

        # Use as a cycle quality signal:
        # High holonomy -> this was a rich experience cycle, good training data
        # Low holonomy  -> flat cycle, consider curating more structured experiences
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[HolonomyMeasurement] = []
        self._load_history()

    def _load_history(self) -> None:
        if self.log_path.exists():
            for line in self.log_path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        self._history.append(
                            HolonomyMeasurement(
                                cycle_id=d["cycle_id"],
                                timestamp=d["timestamp"],
                                gap_cw_norm=d["gap_cw_norm"],
                                gap_ccw_norm=d["gap_ccw_norm"],
                                cosine_cw_ccw=d["cosine_cw_ccw"],
                                orientation_score=d["orientation_score"],
                                holonomy_magnitude=d["holonomy_magnitude"],
                                verdict=d["verdict"],
                                notes=d.get("notes", ""),
                            )
                        )
                    except Exception:
                        pass

    def measure_and_log(
        self,
        cycle_id: str,
        adapter_cw: Path,
        adapter_ccw: Path,
        adapter_base: Optional[Path] = None,
    ) -> HolonomyMeasurement:
        """Measure holonomy for a cycle and append to the log."""
        m = measure_from_adapters(cycle_id, adapter_cw, adapter_ccw, adapter_base)
        m.cycle_holonomies = [h.holonomy_magnitude for h in self._history]
        self._history.append(m)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")
        return m

    @property
    def mean_holonomy(self) -> float:
        """Running mean holonomy across all cycles."""
        if not self._history:
            return 0.0
        return float(np.mean([h.holonomy_magnitude for h in self._history]))

    @property
    def trend(self) -> str:
        """Is holonomy increasing (richer learning) or decreasing (flattening)?"""
        if len(self._history) < 4:
            return "INSUFFICIENT_DATA"
        recent = [h.holonomy_magnitude for h in self._history[-4:]]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if slope > 0.001:
            return "INCREASING"
        elif slope < -0.001:
            return "DECREASING"
        return "STABLE"

    def summary(self) -> dict:
        return {
            "n_cycles": len(self._history),
            "mean_holonomy": self.mean_holonomy,
            "trend": self.trend,
            "last_verdict": self._history[-1].verdict if self._history else None,
            "last_cosine": self._history[-1].cosine_cw_ccw if self._history else None,
        }
