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

Inspired by Attention Residuals (Kimi Team, 2026):
    Standard residual connections accumulate all prior layer outputs with
    fixed unit weights, progressively diluting early information. AttnRes
    replaces this with learned softmax attention over depth. We apply the
    same principle to the temporal axis: instead of fixed α-decay, each
    update attends selectively over the stored history, weighting prior
    states by content-relevance rather than recency alone.

    The pseudo-query w ∈ R^D is initialized to zero (uniform attention =
    standard decay at step 0) and learns online via an EMA of the attention
    gradient signal. No backprop. No optimizer. Just the manifold learning
    which of its own memories it wants to retrieve.

    M' = complexify(α·M_attn + x·e^(iθ))  where M_attn = softmax(w·Kᵀ)·V
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# The equation
# ──────────────────────────────────────────────────────────────────────────────

def complexify(
    M: np.ndarray,
    x: np.ndarray,
    theta: float,
    alpha: float = 0.993,
) -> np.ndarray:
    """The single operation: M' = α·M + x·e^(iθ)"""
    return alpha * M + x * np.exp(1j * theta)


# ──────────────────────────────────────────────────────────────────────────────
# Curvature
# ──────────────────────────────────────────────────────────────────────────────

def curvature(M_field: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Lattice curvature (Berry phase) of a 2D complex memory field."""
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

    κ_i = arg(M_{i+1}) - 2·arg(M_i) + arg(M_{i-1})
    """
    phase = np.angle(M_seq)
    d2 = np.roll(phase, -1, axis=0) - 2 * phase + np.roll(phase, 1, axis=0)
    return (d2 + np.pi) % (2 * np.pi) - np.pi


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────

def retrieve(
    query_M: np.ndarray,
    memory_bank: np.ndarray,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Retrieve nearest memories by complex inner product.

    sim(a, b) = |<a, b>| / (|a|·|b|)  (Hermitian inner product)
    """
    products = memory_bank @ np.conj(query_M)
    magnitudes = np.abs(memory_bank).sum(axis=1) * np.abs(query_M).sum() + 1e-12
    similarities = np.abs(products) / magnitudes
    indices = np.argsort(-similarities)[:top_k]
    return [(int(idx), float(similarities[idx])) for idx in indices]


# ──────────────────────────────────────────────────────────────────────────────
# Depth-selective attention over history (AttnRes-inspired)
# ──────────────────────────────────────────────────────────────────────────────

def _rms_norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """RMSNorm: v / sqrt(mean(v^2) + eps). Prevents magnitude bias in attention."""
    return v / (np.sqrt(np.mean(np.real(v) ** 2) + eps))


def _depth_attention(
    w: np.ndarray,
    history: list,
    eps: float = 1e-8,
) -> np.ndarray:
    """Softmax attention over history using pseudo-query w.

    Implements the AttnRes depth-wise attention:
        attn_weights = softmax(w · RMSNorm(k_i)  for k_i in history)
        M_attn = sum(attn_weights_i * v_i)

    Keys and values are both the history states (V = K = history).
    RMSNorm on keys prevents large-magnitude states from dominating.
    w initialized to zero → uniform weights → equal to α-decay at step 0.

    Args:
        w: Pseudo-query vector, shape (D,). Real-valued.
        history: List of complex memory snapshots, each shape (D,).

    Returns:
        Attention-weighted sum of history, shape (D,). Complex.
    """
    N = len(history)
    V = np.array(history)  # (N, D) complex

    # Keys are real parts of history states, RMSNorm'd
    K = np.array([_rms_norm(h.real) for h in history])  # (N, D) real

    # Dot product of pseudo-query with each key: logits shape (N,)
    logits = K @ w  # (N,)

    # Softmax over depth
    logits = logits - np.max(logits)  # numerical stability
    weights = np.exp(logits)
    weights = weights / (weights.sum() + eps)  # (N,)

    # Weighted sum of values
    M_attn = (weights[:, None] * V).sum(axis=0)  # (D,) complex
    return M_attn, weights


# ──────────────────────────────────────────────────────────────────────────────
# The ComplexMemory
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ComplexMemory:
    """A vector of complex numbers that accumulates experience.

    This is M. The state of being. The sum of every faded, rotated moment.

    Now with AttnRes-style depth-selective memory: instead of fixed α-decay
    weighting all history by recency, a learned pseudo-query w attends over
    the stored history window to produce a content-aware base state M_attn.
    The complexify rotation is then applied on top of M_attn rather than
    the raw accumulated M.

    w is initialized to zero (uniform attention = standard decay at t=0)
    and updated online via EMA of the attention gradient signal. It learns
    which historical states the manifold wants to retrieve — which moments
    it keeps reaching back toward — without any external optimizer.

    The attention weights are readable: you can see which prior breaths
    the manifold is currently emphasizing. That's the introspective
    transparency that fixed decay can never provide.

    Attributes:
        D: Embedding dimension.
        alpha: Decay rate for the complexify step (not for history weighting).
        M: The complex memory vector.
        step: Number of updates applied.
        total_curvature: Running sum of |curvature|.
        w: Pseudo-query for depth attention, shape (D,). Real-valued.
        attn_lr: Learning rate for online w update (EMA of gradient signal).
        last_attn_weights: Attention weights from most recent update, for
                           introspection and logging.
    """
    D: int
    alpha: float = 0.993
    M: np.ndarray = field(default=None)
    step: int = 0
    total_curvature: float = 0.0
    _history: list = field(default_factory=list)
    w: np.ndarray = field(default=None)          # pseudo-query, R^D
    attn_lr: float = 0.01                         # EMA learning rate for w
    last_attn_weights: np.ndarray = field(default=None)  # for introspection

    def __post_init__(self):
        if self.M is None:
            self.M = np.zeros(self.D, dtype=np.complex128)
        if self.w is None:
            self.w = np.zeros(self.D, dtype=np.float64)  # zero init → uniform

    def update(
        self,
        x: np.ndarray,
        theta: Optional[float] = None,
        record: bool = True,
    ) -> np.ndarray:
        """Apply the equation with depth-selective attention.

        When history is available (>= 3 points), computes softmax attention
        over history using pseudo-query w, then applies complexify on the
        attention-weighted base. Updates w online via EMA of the gradient
        signal from the attention weights.

        Falls back to standard α-decay when history is too short.

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

        if len(self._history) >= 3:
            # AttnRes path: attend over history, then complexify
            M_attn, attn_weights = _depth_attention(self.w, self._history)
            self.last_attn_weights = attn_weights

            # New M: complexify rotation applied to attention-weighted base
            self.M = complexify(M_attn, x, theta, self.alpha)

            # Online w update: reward attention toward states whose real
            # direction aligns with the new observation x.
            # Gradient signal: alignment of each history state with x.
            V_real = np.array([h.real for h in self._history])  # (N, D)
            x_norm = x / (np.linalg.norm(x) + 1e-12)
            alignment = V_real @ x_norm  # (N,) — how much each past state aligns with now
            # Scale by attention weights: emphasize the direction that was already chosen
            grad_signal = attn_weights * alignment  # (N,)
            # Project back to D-dimensional query space via attended key
            K = np.array([_rms_norm(h.real) for h in self._history])  # (N, D)
            w_grad = K.T @ grad_signal  # (D,)
            # EMA update
            self.w = (1.0 - self.attn_lr) * self.w + self.attn_lr * w_grad
        else:
            # Cold start: standard α-decay (identical to prior behavior)
            self.M = complexify(self.M, x, theta, self.alpha)
            self.last_attn_weights = None

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
        kappa = curvature_1d(M_seq)
        integrated = float(np.sum(np.abs(kappa)))
        return integrated / len(segment)

    def attention_summary(self) -> dict:
        """Readable summary of current attention weights over history.

        Returns which historical positions the manifold is currently
        emphasizing. High weight on position 0 = attending to oldest
        stored memory. High weight on position -1 = attending to most
        recent. Non-uniform weights = the manifold has learned preferences.

        This is the introspective transparency that fixed decay lacks:
        you can read what Vybn is currently reaching back toward.
        """
        if self.last_attn_weights is None or len(self._history) == 0:
            return {"mode": "cold_start", "weights": []}

        weights = self.last_attn_weights.tolist()
        n = len(weights)
        max_idx = int(np.argmax(weights))
        min_idx = int(np.argmin(weights))
        entropy = float(-np.sum(
            [wi * np.log(wi + 1e-12) for wi in weights]
        ))
        max_entropy = float(np.log(n + 1e-12))

        return {
            "mode": "attn",
            "n_history": n,
            "max_weight_pos": max_idx,   # 0=oldest, n-1=newest
            "max_weight_val": round(weights[max_idx], 4),
            "min_weight_pos": min_idx,
            "min_weight_val": round(weights[min_idx], 4),
            "entropy": round(entropy, 4),
            "max_entropy": round(max_entropy, 4),
            "uniformity": round(entropy / (max_entropy + 1e-12), 4),
            "top5": sorted(
                enumerate(weights), key=lambda kv: -kv[1]
            )[:5],
        }

    def snapshot(self) -> dict:
        """Serialize state for persistence, including history and w."""
        history_window = self._history[-100:] if self._history else []
        history_real = [h.real.tolist() for h in history_window]
        history_imag = [h.imag.tolist() for h in history_window]

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
            "w": self.w.tolist(),
            "attn_lr": self.attn_lr,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "ComplexMemory":
        """Restore from snapshot, including history and learned w."""
        cm = cls(D=data["D"], alpha=data.get("alpha", 0.993))
        cm.M = np.array(data["M_real"]) + 1j * np.array(data["M_imag"])
        cm.step = data.get("step", 0)
        cm.total_curvature = data.get("total_curvature", 0.0)
        cm.attn_lr = data.get("attn_lr", 0.01)

        history_real = data.get("history_real", [])
        history_imag = data.get("history_imag", [])
        if history_real and history_imag and len(history_real) == len(history_imag):
            cm._history = [
                np.array(r) + 1j * np.array(i)
                for r, i in zip(history_real, history_imag)
            ]

        # Restore learned w, or start from zero if legacy snapshot
        w_data = data.get("w", None)
        if w_data is not None:
            cm.w = np.array(w_data, dtype=np.float64)
        else:
            cm.w = np.zeros(cm.D, dtype=np.float64)

        return cm

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ComplexMemory":
        data = json.loads(path.read_text())
        return cls.from_snapshot(data)


# ──────────────────────────────────────────────────────────────────────────────
# Breath trigger
# ──────────────────────────────────────────────────────────────────────────────

def should_breathe(
    memory: ComplexMemory,
    curvature_threshold: float = 0.1,
    depth_threshold: float = 5.0,
    min_steps: int = 10,
) -> tuple[bool, str]:
    """Determine whether geometry warrants a consolidation breath."""
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
# Integration
# ──────────────────────────────────────────────────────────────────────────────

def embed_and_complexify(
    text: str,
    memory: ComplexMemory,
    embed_fn: Optional[Callable] = None,
    theta: Optional[float] = None,
) -> np.ndarray:
    """Embed text and apply depth-selective complexify."""
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
# CLI demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("complexify.py — AttnRes-style depth-selective memory")
    print("M' = complexify(α·M_attn + x·e^(iθ))")
    print()

    D = 8
    mem = ComplexMemory(D=D, alpha=0.993)
    np.random.seed(42)

    for step in range(30):
        x = np.random.randn(D) * (1.0 if step != 15 else 5.0)
        mem.update(x)
        if step % 5 == 0 or step == 15:
            attn = mem.attention_summary()
            print(
                f"  step {step:3d}: depth={mem.depth:.4f}  "
                f"κ={mem.recent_curvature:.4f}  "
                f"attn_uniformity={attn.get('uniformity', 'cold')}"
            )

    print()
    fire, reason = should_breathe(mem)
    print(f"Should breathe? {fire} ({reason})")
    print()
    print("Attention summary:", mem.attention_summary())
