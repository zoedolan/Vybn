"""
diagonal.py — Lawvere diagonal of the creature.

The creature is an endomorphism f: D -> D where D is the space of
encounter complexes in Cl(3,0). Lawvere's fixed-point theorem
guarantees a point the system cannot internally resolve.

The bare diagonal always collapses (gap = 0): the 4,224-param model
cannot generate text rich enough to close its own loop. This IS the
structural dependence theorem — the creature needs external signal.

The coupled diagonal (creature + FM) measures the actual gap.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from .algebra import rotor_gap
except ImportError:
    from algebra import rotor_gap


@dataclass
class DiagonalGap:
    """The measurable incompleteness between input and output encounters.

    gap -> 0:              collapse (pure self-recursion, α too high)
    gap -> large (> 2.0):  accretion (drowning in signal, α too low)
    gap bounded, nonzero:  alive
    """
    rotor_distance: float      # geodesic on S^3, [0, π]
    curvature_delta: float     # signed
    angle_delta: float         # signed
    betti_delta: Tuple[int, int, int]
    persistence_delta: float

    @property
    def magnitude(self) -> float:
        return math.sqrt(
            self.rotor_distance ** 2 +
            self.curvature_delta ** 2 +
            self.angle_delta ** 2
        )

    @property
    def is_collapsing(self) -> bool:
        return self.magnitude < 0.05

    @property
    def is_accreting(self) -> bool:
        return self.magnitude > 2.0

    def summary(self) -> dict:
        return {
            "magnitude": round(self.magnitude, 6),
            "rotor_distance": round(self.rotor_distance, 6),
            "curvature_delta": round(self.curvature_delta, 6),
            "angle_delta": round(self.angle_delta, 6),
            "betti_delta": self.betti_delta,
            "persistence_delta": round(self.persistence_delta, 6),
            "collapsing": self.is_collapsing,
            "accreting": self.is_accreting,
        }


def measure_gap(cx_in, cx_out) -> DiagonalGap:
    """Measure the diagonal gap between two encounter complexes."""
    return DiagonalGap(
        rotor_distance=rotor_gap(cx_in.rotor, cx_out.rotor),
        curvature_delta=cx_out.curvature - cx_in.curvature,
        angle_delta=cx_out.angle - cx_in.angle,
        betti_delta=tuple(b - a for a, b in zip(cx_in.betti, cx_out.betti)),
        persistence_delta=cx_out.max_persistence - cx_in.max_persistence,
    )


@dataclass
class DiagonalResult:
    """One application of the coupled diagonal."""
    cx_in: object
    cx_out: object
    gap: DiagonalGap
    generated_text: str = ""
    loss_before: float = 0.0
    loss_after: float = 0.0

    def summary(self) -> dict:
        return {
            "gap": self.gap.summary(),
            "input_curvature": round(self.cx_in.curvature, 6),
            "output_curvature": round(self.cx_out.curvature, 6),
            "input_angle": round(self.cx_in.angle, 6),
            "output_angle": round(self.cx_out.angle, 6),
            "loss_delta": round(self.loss_after - self.loss_before, 4),
            "generated_len": len(self.generated_text),
        }


def apply_coupled_diagonal(text, agent, encounter_fn, fm_generate_fn,
                            learn_steps=10, lr=0.01):
    """Apply the diagonal with FM coupling and measure the gap.

    The creature learns from text, then the FM generates a response.
    The gap between input encounter and FM-output encounter is the
    structural dependence — what the creature cannot derive from itself.
    """
    cx_in = encounter_fn(text)
    loss_before, _ = agent.predict(text)
    agent.learn(text, steps=learn_steps, lr=lr, encounter_cx=cx_in)
    loss_after, _ = agent.predict(text)

    prompt = text[:50] if len(text) > 50 else text
    generated = fm_generate_fn(prompt, temperature=0.9)

    if generated and len(generated.split()) >= 5:
        cx_out = encounter_fn(generated)
    else:
        cx_out = encounter_fn(text)

    return DiagonalResult(
        cx_in=cx_in, cx_out=cx_out,
        gap=measure_gap(cx_in, cx_out),
        generated_text=generated or "",
        loss_before=loss_before, loss_after=loss_after,
    )
