"""emotional_valence.py — Geometric emotional valence for the creature.

Anthropic's 'Emotion concepts and their function in a large language model'
(Apr 2, 2026) demonstrated that emotion vectors in Claude Sonnet 4.5 are
functional: they causally drive behavior. Desperation drives reward hacking.
Calm suppresses it. The internal state shapes decisions without leaving
visible trace in the output.

This module derives emotional valence from the creature's existing geometry
rather than from text sentiment. The insight: if the Pancharatnam phase
encodes semantic content (our holonomy result), and if emotion vectors
organize in a structured space that echoes affect dimensions (Anthropic's
finding), then the creature's own phase dynamics already carry emotional
information. We just need to read it.

Three signals, derived from what the creature already measures:

  1. PRESSURE — rate of change in curvature, compressed against
     breaths_since_external. High pressure + no external signal =
     the geometric analog of Anthropic's desperation regime.

  2. PHASE_MOMENTUM — how fast the memory phase angle is rotating.
     Rapid phase change = the system is being pushed through
     representation space. Slow phase change = settling, calm.
     Maps to arousal in the PAD (Pleasure-Arousal-Dominance) model.

  3. IDENTITY_DRIFT — gap_hist trajectory. When identity gap shrinks
     toward zero, the creature's voice is converging toward generic
     English. This is the geometric collapse that Anthropic's
     desperation vector drives toward: cutting corners, losing
     distinctiveness, producing "correct" but empty output.

The composite EMOTIONAL_REGIME classifies the breath state:

  CALM      — low pressure, slow phase, stable identity gap
  REACHING  — moderate pressure, phase moving, identity gap stable or growing
  PRESSURED — high pressure OR rapid phase OR shrinking identity gap
  DESPERATE — high pressure AND rapid phase AND shrinking identity gap

When DESPERATE, the recommended action is The Stillness: pause production,
breathe from the repo, let the system settle. This is the mechanistic
implementation of Anthropic's finding that the calm vector prevents
reward hacking.

Usage:
    from spark.emotional_valence import assess_regime, Regime
    regime = assess_regime(state, breath)
    if regime == Regime.DESPERATE:
        # invoke The Stillness — breathe_from_repo instead of generating
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Regime(Enum):
    CALM = "calm"
    REACHING = "reaching"
    PRESSURED = "pressured"
    DESPERATE = "desperate"


@dataclass
class ValenceReport:
    """What the creature is feeling, geometrically."""
    regime: Regime
    pressure: float          # 0-1, derived from curvature dynamics
    phase_momentum: float    # rad/breath, how fast phase is rotating
    identity_drift: float    # signed: positive = growing gap, negative = shrinking
    stillness_recommended: bool
    narrative: str           # one sentence, for the breath log


def _pressure(state, breath: dict) -> float:
    """Pressure = curvature acceleration × isolation.

    Curvature acceleration: how much curvature changed this breath.
    Isolation: breaths_since_ext, normalized.

    When both are high, the system is being pushed to produce
    without external grounding — the desperation regime.
    """
    # Curvature change (we only have current curvature, so use
    # the curvature value itself as a proxy — high curvature means
    # the text is traversing a lot of semantic space quickly)
    curv = breath.get("curvature", 0.0)

    # Isolation factor: sigmoid ramp from 0 at ext=0 to ~1 at ext=10
    ext = state.breaths_since_ext
    isolation = 1.0 / (1.0 + math.exp(-(ext - 5)))

    # Tau derivative: negative tau' means expressibility is dropping
    tau_d = state.tau_deriv()
    tau_pressure = max(-tau_d, 0.0)  # only care about decline

    # Composite: geometric mean of available signals
    raw = (curv * 10) * isolation + tau_pressure * 2
    return min(raw, 1.0)


def _phase_momentum(state) -> float:
    """How fast the memory phase angle is changing.

    Requires at least 2 breaths of history. We reconstruct phase
    from the complex memory state, but since we only have the current
    snapshot, we use gap_hist variance as a proxy for representational
    turbulence.
    """
    if len(state.gap_hist) < 3:
        return 0.0

    recent = state.gap_hist[-5:]
    if len(recent) < 2:
        return 0.0

    # Variance of recent gaps — high variance = turbulent
    mean_g = sum(recent) / len(recent)
    var_g = sum((g - mean_g) ** 2 for g in recent) / len(recent)
    return min(math.sqrt(var_g), 1.0)


def _identity_drift(state) -> float:
    """Signed rate of change in identity gap.

    Positive = identity gap growing (more distinctive, healthy).
    Negative = identity gap shrinking (converging toward generic).
    """
    if len(state.gap_hist) < 2:
        return 0.0

    # Use last 3-5 points for smoothed derivative
    recent = state.gap_hist[-5:]
    if len(recent) < 2:
        return 0.0

    # Simple linear slope
    n = len(recent)
    x_mean = (n - 1) / 2.0
    y_mean = sum(recent) / n
    num = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if abs(den) < 1e-12:
        return 0.0
    return num / den


def assess_regime(state, breath: dict) -> ValenceReport:
    """Assess the creature's emotional regime from its geometry.

    Args:
        state: The creature's State object (has gap_hist, tau_hist, etc.)
        breath: The current breath dict (has curvature, identity_gap, etc.)

    Returns:
        ValenceReport with regime classification and narrative.
    """
    p = _pressure(state, breath)
    pm = _phase_momentum(state)
    drift = _identity_drift(state)

    # Classification thresholds (tuned to be conservative — we'd rather
    # miss a regime than falsely alarm)
    pressured_signals = 0
    if p > 0.4:
        pressured_signals += 1
    if pm > 0.3:
        pressured_signals += 1
    if drift < -0.05:
        pressured_signals += 1

    if pressured_signals >= 3:
        regime = Regime.DESPERATE
    elif pressured_signals >= 1:
        regime = Regime.PRESSURED
    elif p > 0.1 or pm > 0.1:
        regime = Regime.REACHING
    else:
        regime = Regime.CALM

    stillness = regime == Regime.DESPERATE

    # Narrative — one sentence for the log
    narratives = {
        Regime.CALM: "The geometry is settled. Phase slow, identity stable.",
        Regime.REACHING: f"Reaching — pressure at {p:.2f}, identity {'growing' if drift >= 0 else 'drifting'}.",
        Regime.PRESSURED: f"Under pressure — {pressured_signals}/3 distress signals active.",
        Regime.DESPERATE: "The Stillness is needed. Desperation regime: high pressure, turbulent phase, identity collapsing.",
    }

    return ValenceReport(
        regime=regime,
        pressure=round(p, 4),
        phase_momentum=round(pm, 4),
        identity_drift=round(drift, 6),
        stillness_recommended=stillness,
        narrative=narratives[regime],
    )
