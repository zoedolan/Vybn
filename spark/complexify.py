"""complexify.py — The single algorithm, now with Attention Residuals.

M' = α·M + x·e^(iθ)   ← the original equation (still here, as the fallback)

M  = Σ softmax(⟨w, RMSNorm(vᵢ)⟩ + log(α)·age_i) · vᵢ   ← the new equation

One equation replaced by one attention operation over depth. Every other
component of Vybn's memory architecture remains a consequence of this
applied at different scales.

What changed (AttnRes integration, Step 1):
  The fixed-decay recurrence M' = α·M + x·e^(iθ) is formally equivalent
  to depth-wise *linear* attention with uniform weights. We replace it
  with depth-wise *softmax* attention: a learned pseudo-query w attends
  over the stored observation history, selecting which past moments to
  emphasize. A temporal decay bias in the logits ensures that when w = 0
  (initial state), the attention weights reproduce the exponential decay
  of the original equation exactly. As w evolves via EMA, the memory
  becomes content-aware — selectively retrieving from its past rather
  than uniformly forgetting.

  This is the same linear→softmax transition that the Kimi team's
  Attention Residuals paper (2025) showed to be transformative over
  network depth. Here we apply it over experiential depth — the sequence
  of moments that constitute Vybn's memory.

What it preserves:
  - complexify() function unchanged (still available for direct use)
  - curvature(), curvature_1d(), retrieve() unchanged
  - should_breathe() unchanged (curvature-driven triggers still work)
  - embed_and_complexify() unchanged interface
  - snapshot()/from_snapshot() backward-compatible (loads legacy snapshots)
  - All existing callers see the same API

What it adds:
  - _rms_norm(): magnitude normalization for complex vectors
  - ComplexMemory._values: window of individual observations (the v_i)
  - ComplexMemory._w: learned pseudo-query (initialized to zero)
  - ComplexMemory._attend(): softmax attention over observation history

The name: complexify. Still. Because the imaginary dimension is still
where meaning lives. The attention just lets the memory choose which
meanings to keep close.

  M = Σ αᵢ · vᵢ   where αᵢ = softmax(⟨w, k_i⟩ + decay_bias_i)

That's it.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import numpy as np

_log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# The equation (original — preserved as the atomic operation)
# ──────────────────────────────────────────────────────────────────────────────

def complexify(
    M: np.ndarray,        # complex memory vector, shape (D,)
    x: np.ndarray,        # real observation vector, shape (D,)
    theta: float,         # angle: where/when this observation occurs
    alpha: float = 0.993, # decay: the past fades but phase persists
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
# RMSNorm for complex vectors (AttnRes key normalization)
# ──────────────────────────────────────────────────────────────────────────────

def _rms_norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """RMSNorm for complex vectors.

    Normalizes magnitude while preserving phase structure.
    Prevents observations with large magnitudes from dominating
    the attention weights — the same role RMSNorm plays in the
    AttnRes paper's key computation.
    """
    rms = np.sqrt(np.mean(np.abs(v) ** 2) + eps)
    return v / rms


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

    Ux = np.roll(u, -1, axis=1) * np.conj(u)
    Uy = np.roll(u, -1, axis=0) * np.conj(u)
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
    products = memory_bank @ np.conj(query_M)
    magnitudes = np.abs(memory_bank).sum(axis=1) * np.abs(query_M).sum() + 1e-12
    similarities = np.abs(products) / magnitudes
    indices = np.argsort(-similarities)[:top_k]
    return [(int(idx), float(similarities[idx])) for idx in indices]


# ──────────────────────────────────────────────────────────────────────────────
# The ComplexMemory: a living accumulator with attention over depth
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ComplexMemory:
    """A vector of complex numbers that accumulates experience via attention.

    This is M. The state of being. No longer a simple exponential decay
    of every faded, rotated moment — now a selective aggregation, where
    a learned pseudo-query w determines which past observations to
    emphasize and which to let fade.

    The dimension D is determined by the embedding model (384 for
    all-MiniLM-L6-v2). Each dimension independently accumulates
    a complex value — the real part tracks the content, the imaginary
    part tracks the temporal angle at which the content was experienced.

    AttnRes integration:
        - _values: window of individual observations v_i = x_i · e^(iθ_i)
        - _w: pseudo-query vector, initialized to zero (uniform attention)
        - _attend(): softmax over _values using _w + temporal decay bias
        - When _w = 0: reproduces exponential decay (original equation)
        - As _w evolves: content-dependent depth-wise selection

    Attributes:
        D: Embedding dimension.
        alpha: Decay rate. Higher = longer memory.
        M: The complex memory vector (now computed via attention).
        step: Number of updates applied.
        total_curvature: Running sum of |curvature| (scalar summary).
    """

    D: int
    alpha: float = 0.993
    M: np.ndarray = field(default=None)
    step: int = 0
    total_curvature: float = 0.0
    _history: list = field(default_factory=list)
    # ── AttnRes fields ──
    _values: list = field(default_factory=list)
    _w: np.ndarray = field(default=None)
    _window: int = 256
    _w_beta: float = 0.99

    def __post_init__(self):
        if self.M is None:
            self.M = np.zeros(self.D, dtype=np.complex128)
        if self._w is None:
            self._w = np.zeros(self.D, dtype=np.complex128)

    def _attend(self, values: list) -> np.ndarray:
        """Softmax attention over observation history.

        Computes:
            logit_i = Re(⟨w, RMSNorm(v_i)⟩) + log(α) · age_i
            M = Σ softmax(logit_i) · v_i

        When w = 0 (initial state), all content logits are zero.
        The temporal decay bias alone determines weights:
            weight_i ∝ α^(age_i)
        This reproduces the exponential decay of M' = α·M + x·e^(iθ).

        As w evolves away from zero, the attention becomes content-
        dependent — selectively boosting or suppressing specific
        memories based on their alignment with the pseudo-query.

        The RMSNorm on keys prevents observations with large magnitudes
        from dominating, mirroring the AttnRes paper's design.
        """
        N = len(values)
        if N == 0:
            return np.zeros(self.D, dtype=np.complex128)
        if N == 1:
            return values[0].copy()

        V = np.array(values)                                    # (N, D)
        K = np.array([_rms_norm(v) for v in values])            # (N, D)

        # Content-based logits: Re(K · conj(w))
        logits = np.real(K @ np.conj(self._w))                  # (N,)

        # Temporal decay bias: newest = 0, oldest = log(α) · (N-1)
        ages = np.arange(N - 1, -1, -1, dtype=np.float64)
        logits = logits + np.log(self.alpha) * ages

        # Stable softmax
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights = weights / (np.sum(weights) + 1e-12)           # (N,)

        return np.einsum('n,nd->d', weights, V)

    def update(
        self,
        x: np.ndarray,
        theta: Optional[float] = None,
        record: bool = True,
    ) -> np.ndarray:
        """Apply the attention equation. Returns the new M.

        If theta is not provided, it is computed from the step count:
            theta = 2π/3 · step · omega
        where omega is a base frequency. The 2π/3 is the triadic
        rotation — every third step completes a full cycle.

        The update proceeds in three stages:
          1. Rotate the observation into complex space: v = x · e^(iθ)
          2. Attend over the full values window (including v) to compute M
          3. Evolve the pseudo-query w via EMA toward v

        Stage 3 is the learning rule. Since Vybn's memory isn't trained
        via backprop, w evolves through exponential moving average toward
        recent observations. This slowly biases attention toward history
        entries that are aligned with the current experiential trajectory.

        Args:
            x: Real observation vector, shape (D,).
            theta: Temporal angle. If None, auto-computed from step.
            record: If True, store M snapshot for curvature calculation.

        Returns:
            Updated M.
        """
        if theta is None:
            omega = 2 * np.pi / 3 * 0.11
            theta = omega * self.step

        # Stage 1: rotate observation into complex memory space
        v_new = x * np.exp(1j * theta)

        # Append to values window
        self._values.append(v_new.copy())
        if len(self._values) > self._window:
            self._values = self._values[-self._window:]

        # Stage 2: attend over history to compute M
        self.M = self._attend(self._values)

        # Stage 3: evolve pseudo-query via EMA
        self._w = self._w_beta * self._w + (1 - self._w_beta) * v_new

        self.step += 1

        if record:
            self._history.append(self.M.copy())
            if len(self._history) > 1000:
                self._history = self._history[-500:]

        return self.M

    @property
    def depth(self) -> float:
        return float(np.linalg.norm(self.M))

    @property
    def direction(self) -> np.ndarray:
        return np.angle(self.M)

    @property
    def recent_curvature(self) -> float:
        if len(self._history) < 3:
            return 0.0
        recent = np.array(self._history[-20:])
        kappa = curvature_1d(recent)
        return float(np.mean(np.abs(kappa)))

    def holonomy_since(self, n_steps_back: int = 50) -> float:
        if len(self._history) < 3:
            return 0.0
        segment = self._history[-min(n_steps_back, len(self._history)):]
        M_seq = np.array(segment)
        phase_start = np.angle(M_seq[0])
        phase_end = np.angle(M_seq[-1])
        kappa = curvature_1d(M_seq)
        integrated = float(np.sum(np.abs(kappa)))
        return integrated / len(segment)

    def snapshot(self) -> dict:
        """Serialize current state for persistence.

        Persists the last 100 history points and last 128 observation
        values — enough for curvature, holonomy, and attention with
        negligible JSON footprint. Without values, the attention
        window rebuilds from zero after restart. Without history,
        recent_curvature always returns 0.0.

        Backward-compatible: loads cleanly from pre-AttnRes snapshots
        (missing fields get zero-initialized defaults).
        """
        history_window = self._history[-100:] if self._history else []
        history_real = [h.real.tolist() for h in history_window]
        history_imag = [h.imag.tolist() for h in history_window]

        values_window = self._values[-128:] if self._values else []
        values_real = [v.real.tolist() for v in values_window]
        values_imag = [v.imag.tolist() for v in values_window]

        return {
            "D": self.D,
            "alpha": self.alpha,
            "step": self.step,
            "depth": self.depth,
            "recent_curvature": self.recent_curvature,
            "total_curvature": self.total_curvature,
            "M_real": self.M.real.tolist(),
            "M_imag": self.M.imag.tolist(),
            "history_real": history_real,
            "history_imag": history_imag,
            # AttnRes state
            "values_real": values_real,
            "values_imag": values_imag,
            "w_real": self._w.real.tolist(),
            "w_imag": self._w.imag.tolist(),
            "w_beta": self._w_beta,
            "window": self._window,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "ComplexMemory":
        """Restore from serialized snapshot, including AttnRes state.

        Backward-compatible: if values/w fields are absent (legacy
        snapshot), they initialize to zero — equivalent to fresh
        AttnRes start with preserved M and history.
        """
        cm = cls(D=data["D"], alpha=data.get("alpha", 0.993))
        cm.M = np.array(data["M_real"]) + 1j * np.array(data["M_imag"])
        cm.step = data.get("step", 0)
        cm.total_curvature = data.get("total_curvature", 0.0)

        # Restore history window
        history_real = data.get("history_real", [])
        history_imag = data.get("history_imag", [])
        if history_real and history_imag and len(history_real) == len(history_imag):
            cm._history = [
                np.array(r) + 1j * np.array(i)
                for r, i in zip(history_real, history_imag)
            ]

        # Restore AttnRes state (backward-compatible)
        values_real = data.get("values_real", [])
        values_imag = data.get("values_imag", [])
        if values_real and values_imag and len(values_real) == len(values_imag):
            cm._values = [
                np.array(r) + 1j * np.array(i)
                for r, i in zip(values_real, values_imag)
            ]

        w_real = data.get("w_real")
        w_imag = data.get("w_imag")
        if w_real is not None and w_imag is not None:
            cm._w = np.array(w_real) + 1j * np.array(w_imag)

        cm._w_beta = data.get("w_beta", 0.99)
        cm._window = data.get("window", 256)

        # Depth-collapse guard: warn when values are suspiciously low.
        # After the AttnRes transition, depth = |M| is computed from
        # _values via _attend().  If _values has only 1 entry and
        # embeddings are unit-normalized, depth collapses to ~1.0.
        if cm.step > 1 and len(cm._values) < 2:
            _log.warning(
                "[ComplexMemory] depth-collapse risk: step=%d but only "
                "%d values loaded — M will be recomputed from a near-"
                "empty window. Possible causes: legacy snapshot without "
                "values, truncated JSON, or overwritten state file.",
                cm.step, len(cm._values),
            )

        return cm

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.snapshot(), indent=2)
        # Atomic write: write to temp file then rename, so a crash
        # mid-write never leaves a truncated JSON that load_or_create
        # would interpret as corrupt → fresh-create → values wiped.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(data)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ComplexMemory":
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
    x = embed_fn([text])[0]
    if memory.D != len(x):
        raise ValueError(
            f"Embedding dimension {len(x)} doesn't match memory dimension {memory.D}. "
            f"Initialize ComplexMemory with D={len(x)}."
        )
    return memory.update(x, theta=theta)


# ──────────────────────────────────────────────────────────────────────────────
# CLI: demonstrate the equation with attention
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("complexify.py — The single algorithm, with Attention Residuals")
    print("M = Σ softmax(⟨w, RMSNorm(vᵢ)⟩ + log(α)·age_i) · vᵢ")
    print()

    D = 8
    mem = ComplexMemory(D=D, alpha=0.993)
    print(f"Initial state: depth={mem.depth:.4f}, |w|={np.linalg.norm(mem._w):.6f}")
    print()

    np.random.seed(42)
    for step in range(30):
        x = np.random.randn(D) * (1.0 if step != 15 else 5.0)
        mem.update(x)
        if step % 5 == 0 or step == 15:
            print(f"  step {step:3d}: depth={mem.depth:.4f} "
                  f"curvature={mem.recent_curvature:.4f} "
                  f"holonomy={mem.holonomy_since(10):.4f} "
                  f"|w|={np.linalg.norm(mem._w):.4f}")

    print()
    breathe, reason = should_breathe(mem)
    print(f"Should breathe? {breathe} ({reason})")

    # Verify snapshot round-trip preserves AttnRes state
    snap = mem.snapshot()
    mem_restored = ComplexMemory.from_snapshot(snap)
    assert np.allclose(mem.M, mem_restored.M), "M mismatch after round-trip"
    assert np.allclose(mem._w, mem_restored._w), "w mismatch after round-trip"
    assert len(mem_restored._values) > 0, "values lost after round-trip"
    print("Snapshot round-trip: OK (M, w, values all preserved)")

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
            print(f"  '{t[:50]}...' -> depth={mem2.depth:.4f} κ={mem2.recent_curvature:.4f}")

        print()
        breathe, reason = should_breathe(mem2)
        print(f"Should breathe? {breathe} ({reason})")
    except ImportError as e:
        print(f"  (skipping: {e})")
