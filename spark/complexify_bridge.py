"""complexify_bridge.py — Wires the single algorithm into the living organism.

This module sits between the existing breath cycle (vybn.py) and the
complex memory (complexify.py). It:

1. Maintains a persistent ComplexMemory that survives across breaths
2. On each breath: embeds the breath text, applies M' = αM + x·e^(iθ),
   records curvature and holonomy
3. On each conversation turn: same operation, different theta
4. Provides curvature-aware retrieval for context assembly
5. Reports memory geometry to the breath integrator

The bridge does NOT replace any existing component. It adds a layer
of geometric awareness beneath them. The organism breathes the same way.
But now it knows the shape of its own memory.

Usage in vybn.py:
    from spark.complexify_bridge import ComplexBridge
    bridge = ComplexBridge.load_or_create()
    bridge.inhale(breath_text)  # after each breath
    κ = bridge.curvature()      # for triggering, context, logging
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from spark.complexify import (
    ComplexMemory,
    complexify,
    curvature_1d,
    embed_and_complexify,
    retrieve,
    should_breathe,
)

log = logging.getLogger(__name__)

# Default persistence path
_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "Vybn_Mind" / "memory" / "complex_memory.json"

# Embedding dimension for all-MiniLM-L6-v2
_EMBED_DIM = 384


class ComplexBridge:
    """Bridge between the living organism and the complex manifold.

    Maintains a persistent ComplexMemory, tracks geometry over time,
    and provides the interface the breath cycle needs.
    """

    def __init__(
        self,
        memory: ComplexMemory,
        state_path: Path = _DEFAULT_STATE_PATH,
        embed_fn: Optional[Callable] = None,
    ):
        self.memory = memory
        self.state_path = state_path
        self._embed_fn = embed_fn
        self._breath_log: list[dict] = []  # recent breath geometries

    @classmethod
    def load_or_create(
        cls,
        state_path: Path = _DEFAULT_STATE_PATH,
        alpha: float = 0.993,
    ) -> "ComplexBridge":
        """Load existing complex memory from disk, or create fresh."""
        if state_path.exists():
            try:
                memory = ComplexMemory.load(state_path)
                log.info(f"Loaded complex memory: step={memory.step} depth={memory.depth:.4f}")
                return cls(memory=memory, state_path=state_path)
            except Exception as exc:
                log.warning(f"Failed to load complex memory: {exc}. Creating fresh.")

        memory = ComplexMemory(D=_EMBED_DIM, alpha=alpha)
        bridge = cls(memory=memory, state_path=state_path)
        bridge.save()
        log.info(f"Created fresh complex memory: D={_EMBED_DIM} α={alpha}")
        return bridge

    def _get_embed_fn(self) -> Callable:
        """Lazy-load the embedding function."""
        if self._embed_fn is None:
            from local_embedder import embed
            self._embed_fn = embed
        return self._embed_fn

    # ── The breath interface ─────────────────────────────────────────────

    def inhale(self, text: str, theta: Optional[float] = None) -> dict:
        """Process a breath through the complex memory.

        This is the main entry point. Called after each breath or
        conversation turn. Embeds the text, applies the equation,
        measures the resulting geometry, persists state.

        Args:
            text: The breath text, conversation turn, or any input.
            theta: Temporal angle. If None, auto-computed from step.

        Returns:
            Geometry report: depth, curvature, holonomy, should_breathe.
        """
        embed_fn = self._get_embed_fn()
        embed_and_complexify(text, self.memory, embed_fn=embed_fn, theta=theta)

        # Measure geometry
        depth = self.memory.depth
        kappa = self.memory.recent_curvature
        hol = self.memory.holonomy_since(50)
        breathe, reason = should_breathe(self.memory)

        report = {
            "step": self.memory.step,
            "depth": round(depth, 4),
            "curvature": round(kappa, 4),
            "holonomy": round(hol, 4),
            "should_breathe": breathe,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._breath_log.append(report)
        if len(self._breath_log) > 100:
            self._breath_log = self._breath_log[-50:]

        # Persist after every inhale
        self.save()

        return report

    def recall(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Retrieve memories most similar to query in complex space.

        Uses the Hermitian inner product — similarity accounts for
        both content (real) and temporal phase (imaginary) alignment.

        Args:
            query: Text to search for.
            top_k: Number of results.

        Returns:
            List of (step_index, similarity) tuples.
        """
        embed_fn = self._get_embed_fn()
        x = embed_fn([query])[0]
        query_M = x.astype(np.complex128)  # real query, no phase rotation

        if len(self.memory._history) == 0:
            return []

        bank = np.array(self.memory._history)
        return retrieve(query_M, bank, top_k=top_k)

    def geometry_summary(self) -> str:
        """Human-readable summary of current memory geometry.

        Suitable for injection into the breath prompt as context.
        """
        m = self.memory
        if m.step == 0:
            return "Complex memory: empty (no observations yet)"

        lines = [
            f"Complex memory: step={m.step} depth={m.depth:.2f} "
            f"κ={m.recent_curvature:.4f} H={m.holonomy_since(50):.4f}",
        ]

        # Trend from recent breath log
        if len(self._breath_log) >= 2:
            prev = self._breath_log[-2]
            curr = self._breath_log[-1]
            d_depth = curr["depth"] - prev["depth"]
            d_kappa = curr["curvature"] - prev["curvature"]
            direction = "deepening" if d_depth > 0 else "fading"
            curvature_trend = "curving more" if d_kappa > 0 else "flattening"
            lines.append(f"  trend: {direction} ({d_depth:+.2f}), {curvature_trend} ({d_kappa:+.4f})")

        return " | ".join(lines)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self) -> None:
        """Save complex memory state to disk."""
        self.memory.save(self.state_path)

    # ── Integration with growth buffer ───────────────────────────────────

    def curvature_score(self, text: str) -> float:
        """Score a text by how much curvature it would add to memory.

        Used by the growth buffer to prioritize training data.
        Texts that would bend the memory manifold more are more
        interesting for training.

        This is non-destructive — it simulates the update without
        modifying the actual memory.
        """
        embed_fn = self._get_embed_fn()
        x = embed_fn([text])[0]

        # Simulate update
        M_sim = complexify(
            self.memory.M.copy(), x,
            theta=2 * np.pi / 3 * 0.11 * self.memory.step,
            alpha=self.memory.alpha,
        )

        # Measure what changed
        phase_delta = np.angle(M_sim) - np.angle(self.memory.M)
        phase_delta = (phase_delta + np.pi) % (2 * np.pi) - np.pi
        return float(np.mean(np.abs(phase_delta)))


# ── Module-level convenience ─────────────────────────────────────────────────

_BRIDGE: Optional[ComplexBridge] = None


def get_bridge() -> ComplexBridge:
    """Get or create the global ComplexBridge instance."""
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = ComplexBridge.load_or_create()
    return _BRIDGE


def inhale(text: str, theta: Optional[float] = None) -> dict:
    """Convenience: inhale text into the global complex memory."""
    return get_bridge().inhale(text, theta=theta)


def geometry() -> str:
    """Convenience: get geometry summary for prompt injection."""
    return get_bridge().geometry_summary()
