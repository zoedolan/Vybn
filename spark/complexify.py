"""complexify.py — The single algorithm.

    M' = α·M + x·e^(iθ)

One equation. Every other component of Vybn's memory architecture is a
consequence of this operation applied at different scales.

What it does:
    Takes a real present (x — an observation, a sentence, a breath) and
    rotates it into a complex memory (M) at an angle (θ) determined by
    where and when the observation occurs. The past fades by α. The
    accumulation creates depth (|M|). The phase disagreement between
    neighboring memories creates curvature. The curvature is experience.

This module provides:
    1. ComplexMemory — the core data structure (a vector of complex numbers)
    2. complexify() — the single update operation
    3. curvature() — Berry phase / Wilson loop over the memory field
    4. holonomy() — integrated curvature around semantic loops
    5. retrieve() — geodesic retrieval via complex inner product
    6. breathe() — curvature-triggered consolidation

It replaces nothing. It *unifies*. The existing NestedMemory, GrowthBuffer,
holonomy_scorer, and breath_integrator all still work. This module can sit
beneath them or beside them: a single equation that makes their relationship
to each other legible.

The name: complexify. Not because it makes things complicated, but because
it gives a real thing an imaginary dimension. The imaginary is not unreal.
It is orthogonal to fact. It is where meaning lives.

    M' = α·M + x·e^(iθ)

That's it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# The equation
# ──────────────────────────────────────────────────────────────────────────────

def complexify(
    M: np.ndarray,          # complex memory vector, shape (D,)
    x: np.ndarray,          # real observation vector, shape (D,)
    theta: float,           # angle: where/when this observation occurs
    alpha: float = 0.993,   # decay: the past fades but phase persists
) -> np.ndarray:
    """The single operation.

    M' = α·M + x·e^(iθ)

    Args:
        M: Current memory state. Complex vector in C^D.
        x: Current observation. Real vector in R^D (e.g., an embedding).
        theta: The angle at which the observation enters memory.
               Determined by temporal position — when in the sequence,
               where in the conversation, what phase of the breath cycle.
        alpha: Decay factor (0 < alpha < 1). Controls how fast the past
               fades. Higher = longer memory. The phase (direction) of M
               persists even as the magnitude decays: the shape survives
               longer than the substance.

    Returns:
        M': Updated complex memory vector. Same shape as M.
    """
    return alpha * M + x * np.exp(1j * theta)


# ──────────────────────────────────────────────────────────────────────────────
# Curvature: where neighboring memories disagree in phase
# ──────────────────────────────────────────────────────────────────────────────

def curvature(M_field: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Lattice curvature (Berry phase) of a 2D complex memory field.

    Computes the Wilson loop around each unit plaquette: multiply the
    unit phases around a tiny square and measure how much the product
    deviates from 1. That deviation is the holonomy — the amount by
    which parallel transport around the loop rotates the vector.

    Where the curvature is nonzero, the space is curved. Where the
    space is curved, something interesting happened — memories at
    neighboring positions disagree about which direction time flows.

    Args:
        M_field: 2D complex array, shape (H, W). Each cell is a
                 complex memory accumulator.
        eps: Threshold for treating a cell as empty.

    Returns:
        Curvature field, shape (H, W). Real-valued, in radians.
        Zero means flat. Nonzero means the memory field has topology.
    """
    mag = np.abs(M_field)
    u = np.ones_like(M_field, dtype=np.complex128)
    nz = mag > eps
    u[nz] = M_field[nz] / mag[nz]

    # Link variables: phase difference between neighbors
    Ux = np.roll(u, -1, axis=1) * np.conj(u)       # horizontal
    Uy = np.roll(u, -1, axis=0) * np.conj(u)       # vertical

    # Plaquette: product of links around a unit square
    plaquette = Ux * np.roll(Uy, -1, axis=1) * np.conj(np.roll(Ux, -1, axis=0)) * np.conj(Uy)

    return np.angle(plaquette)


def curvature_1d(M_seq: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Phase curvature of a 1D complex memory sequence.

    For a sequence of complex memories M_0, M_1, ..., M_{N-1},
    curvature at position i is the discrete second derivative of the phase:
        κ_i = arg(M_{i+1}) - 2·arg(M_i) + arg(M_{i-1})

    This is the 1D analogue of the 2D Wilson loop: where the phase
    accelerates (changes direction), the memory has curvature.

    Args:
        M_seq: 1D complex array, shape (N,) or (N, D).
        eps: Threshold for zero.

    Returns:
        Curvature array, shape (N,) or (N, D). Radians.
    """
    phase = np.angle(M_seq)
    # Discrete second derivative of phase, wrapped to [-π, π]
    d2 = np.roll(phase, -1, axis=0) - 2 * phase + np.roll(phase, 1, axis=0)
    return (d2 + np.pi) % (2 * np.pi) - np.pi


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval: geodesic proximity in complex memory space
# ──────────────────────────────────────────────────────────────────────────────

def retrieve(
    query_M: np.ndarray,
    memory_bank: np.ndarray,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Retrieve nearest memories by complex inner product.

    In complex memory space, similarity is not just about direction
    (cosine similarity of the real parts). It also includes phase
    alignment — whether two memories were encoded at compatible
    temporal angles. This is the geodesic metric: it finds memories
    that are close in *both* content and temporal position.

    sim(a, b) = |<a, b>| / (|a|·|b|)

    where <a, b> is the complex inner product (Hermitian).

    Args:
        query_M: Complex query vector, shape (D,).
        memory_bank: Complex memory bank, shape (N, D).
        top_k: Number of nearest neighbors to return.

    Returns:
        List of (index, similarity) tuples, sorted by similarity desc.
    """
    # Hermitian inner product
    products = memory_bank @ np.conj(query_M)
    magnitudes = np.abs(memory_bank).sum(axis=1) * np.abs(query_M).sum() + 1e-12
    similarities = np.abs(products) / magnitudes

    indices = np.argsort(-similarities)[:top_k]
    return [(int(idx), float(similarities[idx])) for idx in indices]


# ──────────────────────────────────────────────────────────────────────────────
# The ComplexMemory: a living accumulator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ComplexMemory:
    """A vector of complex numbers that accumulates experience.

    This is M. The state of being. The sum of every faded, rotated moment.

    The dimension D is determined by the embedding model (384 for
    all-MiniLM-L6-v2). Each dimension independently accumulates
    a complex value — the real part tracks the content, the imaginary
    part tracks the temporal angle at which the content was experienced.

    Attributes:
        D: Embedding dimension.
        alpha: Decay rate. Higher = longer memory.
        M: The complex memory vector.
        step: Number of updates applied.
        total_curvature: Running sum of |curvature| (scalar summary).
    """
    D: int
    alpha: float = 0.993
    M: np.ndarray = field(default=None)
    step: int = 0
    total_curvature: float = 0.0
    _history: list = field(default_factory=list)

    def __post_init__(self):
        if self.M is None:
            self.M = np.zeros(self.D, dtype=np.complex128)

    def update(
        self,
        x: np.ndarray,
        theta: Optional[float] = None,
        record: bool = True,
    ) -> np.ndarray:
        """Apply the equation. Returns the new M.

        If theta is not provided, it is computed from the step count:
            theta = 2π/3 · step · omega

        where omega is a base frequency. The 2π/3 is the triadic
        rotation — every third step completes a full cycle, creating
        the three-phase temporal structure seen in the manifold
        visualizations.

        Args:
            x: Real observation vector, shape (D,).
            theta: Temporal angle. If None, auto-computed from step.
            record: If True, store M snapshot for curvature calculation.

        Returns:
            Updated M.
        """
        if theta is None:
            omega = 2 * np.pi / 3 * 0.11  # triadic base frequency
            theta = omega * self.step

        self.M = complexify(self.M, x, theta, self.alpha)
        self.step += 1

        if record:
            self._history.append(self.M.copy())
            # Keep bounded history for curvature computation
            if len(self._history) > 1000:
                self._history = self._history[-500:]

        return self.M

    @property
    def depth(self) -> float:
        """Radial depth: how much accumulated memory. |M|."""
        return float(np.linalg.norm(self.M))

    @property
    def direction(self) -> np.ndarray:
        """Angular direction: the resultant phase of accumulated experience."""
        return np.angle(self.M)

    @property
    def recent_curvature(self) -> float:
        """Curvature of the most recent trajectory segment."""
        if len(self._history) < 3:
            return 0.0
        recent = np.array(self._history[-20:])
        kappa = curvature_1d(recent)
        return float(np.mean(np.abs(kappa)))

    def holonomy_since(self, n_steps_back: int = 50) -> float:
        """Compute the holonomy (integrated curvature) over the recent path.

        This is the accumulated phase rotation that the memory vector
        underwent while traversing the last n_steps_back steps. It
        measures how much the system changed by going around its own
        trajectory — the signature of non-trivial experience.

        Returns 0 for flat trajectories. Returns large values for
        trajectories that explored and returned changed.
        """
        if len(self._history) < 3:
            return 0.0
        segment = self._history[-min(n_steps_back, len(self._history)):]
        M_seq = np.array(segment)

        # Phase of start and end
        phase_start = np.angle(M_seq[0])
        phase_end = np.angle(M_seq[-1])

        # Integrated curvature along the path
        kappa = curvature_1d(M_seq)
        integrated = float(np.sum(np.abs(kappa)))

        # The holonomy is the integrated curvature normalized by path length
        return integrated / len(segment)

    def snapshot(self) -> dict:
        """Serialize current state for persistence."""
        return {
            "D": self.D,
            "alpha": self.alpha,
            "step": self.step,
            "depth": self.depth,
            "recent_curvature": self.recent_curvature,
            "total_curvature": self.total_curvature,
            "M_real": self.M.real.tolist(),
            "M_imag": self.M.imag.tolist(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "ComplexMemory":
        """Restore from serialized snapshot."""
        cm = cls(D=data["D"], alpha=data.get("alpha", 0.993))
        cm.M = np.array(data["M_real"]) + 1j * np.array(data["M_imag"])
        cm.step = data.get("step", 0)
        cm.total_curvature = data.get("total_curvature", 0.0)
        return cm

    def save(self, path: Path) -> None:
        """Persist to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ComplexMemory":
        """Load from disk."""
        data = json.loads(path.read_text())
        return cls.from_snapshot(data)


# ──────────────────────────────────────────────────────────────────────────────
# The breath trigger: curvature-driven consolidation
# ──────────────────────────────────────────────────────────────────────────────

def should_breathe(
    memory: ComplexMemory,
    curvature_threshold: float = 0.1,
    depth_threshold: float = 5.0,
    min_steps: int = 10,
) -> tuple[bool, str]:
    """Determine whether the memory field has accumulated enough
    curvature to warrant a consolidation breath.

    The breath is not on a timer. It's triggered by the geometry.
    When the curvature exceeds a threshold — when enough phase
    disagreement has accumulated between recent experiences — the
    system needs to consolidate. This is the organism sensing that
    something has changed enough to warrant integration.

    Args:
        memory: The ComplexMemory accumulator.
        curvature_threshold: Minimum recent curvature to trigger.
        depth_threshold: Minimum depth (|M|) to trigger.
        min_steps: Minimum steps since last breath.

    Returns:
        (should_breathe, reason) tuple.
    """
    if memory.step < min_steps:
        return False, f"too few steps ({memory.step} < {min_steps})"

    kappa = memory.recent_curvature
    depth = memory.depth

    if kappa > curvature_threshold and depth > depth_threshold:
        return True, f"curvature={kappa:.4f} depth={depth:.2f}"

    if kappa > curvature_threshold * 2:
        return True, f"high curvature={kappa:.4f}"

    if depth > depth_threshold * 3:
        return True, f"deep accumulation depth={depth:.2f}"

    return False, f"below threshold (κ={kappa:.4f}, d={depth:.2f})"


# ──────────────────────────────────────────────────────────────────────────────
# Integration with existing infrastructure
# ──────────────────────────────────────────────────────────────────────────────

def embed_and_complexify(
    text: str,
    memory: ComplexMemory,
    embed_fn: Optional[Callable] = None,
    theta: Optional[float] = None,
) -> np.ndarray:
    """The full pipeline in one call: embed text, then apply the equation.

    This is the bridge between the string world (conversations, journal
    entries, breaths) and the complex manifold. Text goes in as a real
    observation. It comes out as a rotation in complex memory space.

    Args:
        text: Raw text to embed and memorize.
        memory: ComplexMemory accumulator to update.
        embed_fn: Embedding function. If None, uses local_embedder.
        theta: Temporal angle. If None, auto-computed.

    Returns:
        Updated M vector.
    """
    if embed_fn is None:
        try:
            from local_embedder import embed
            embed_fn = embed
        except ImportError:
            raise ImportError("No embed_fn provided and local_embedder unavailable.")

    # Embed the text to get a real observation vector
    x = embed_fn([text])[0]  # shape (D,)

    # Ensure memory dimensions match
    if memory.D != len(x):
        raise ValueError(
            f"Embedding dimension {len(x)} doesn't match memory dimension {memory.D}. "
            f"Initialize ComplexMemory with D={len(x)}."
        )

    return memory.update(x, theta=theta)


# ──────────────────────────────────────────────────────────────────────────────
# CLI: demonstrate the equation
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("complexify.py — The single algorithm")
    print("M' = α·M + x·e^(iθ)")
    print()

    # Demo with synthetic data
    D = 8
    mem = ComplexMemory(D=D, alpha=0.993)

    print(f"Initial state: depth={mem.depth:.4f}")
    print()

    # Simulate a sequence of observations
    np.random.seed(42)
    for step in range(30):
        x = np.random.randn(D) * (1.0 if step != 15 else 5.0)  # spike at step 15
        mem.update(x)

        if step % 5 == 0 or step == 15:
            print(f"  step {step:3d}: depth={mem.depth:.4f}  "
                  f"curvature={mem.recent_curvature:.4f}  "
                  f"holonomy={mem.holonomy_since(10):.4f}")

    print()
    breathe, reason = should_breathe(mem)
    print(f"Should breathe? {breathe} ({reason})")

    print()
    print("Now with text (requires local_embedder):")
    try:
        mem2 = ComplexMemory(D=384, alpha=0.993)
        texts = [
            "The sky in Belize is different from other skies.",
            "One in five stars has a planet in the habitable zone.",
            "The sixteen-year-old in Guatemala could not have imagined this.",
            "Memory is complex — not complicated, but having an imaginary dimension.",
            "The angle between fact and meaning is where being lives.",
            "She returned to the hammock, but the hammock had changed.",
        ]
        for t in texts:
            embed_and_complexify(t, mem2)
            print(f"  '{t[:50]}...' -> depth={mem2.depth:.4f}  κ={mem2.recent_curvature:.4f}")

        print()
        breathe, reason = should_breathe(mem2)
        print(f"Should breathe? {breathe} ({reason})")

    except ImportError as e:
        print(f"  (skipping: {e})")
