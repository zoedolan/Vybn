"""complexify_bridge.py — Wires the single algorithm into the living organism.

This module sits between the existing breath cycle (vybn.py) and the
complex memory (complexify.py). It:

1. Maintains a persistent ComplexMemory that survives across breaths
2. On each breath: embeds the breath text, applies M' = αM + x·e^(iθ),
   records curvature and holonomy
3. On each conversation turn: same operation, different theta
4. Provides curvature-aware retrieval for context assembly
5. Reports memory geometry to the breath integrator
6. Gates agency proposals by curvature — rejects flat explorations
7. Probes Jordan structure — detects when the operator approaches
   non-diagonalizability (the memory regime where α_eff → 1)

The bridge does NOT replace any existing component. It adds a layer
of geometric awareness beneath them. The organism breathes the same way.
But now it knows the shape of its own memory.

Usage in vybn.py:
    from spark.complexify_bridge import ComplexBridge
    bridge = ComplexBridge.load_or_create()
    bridge.inhale(breath_text)  # after each breath
    κ = bridge.curvature()      # for triggering, context, logging

Usage in agency.py (curvature gate):
    from spark.complexify_bridge import should_explore
    verdict = should_explore(proposal_text)
    if not verdict["pass"]:
        # reject — manifold is flat in this direction
"""

from __future__ import annotations

import json
import logging
import os
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

        Diagnostics (always logged):
            - input length and first 80 chars
            - embedding norm (zero norm = embedder failure)
            - κ before and after update
            - κΔ (the value we've been watching flatline)
        """
        # ── Diagnostic: what's arriving? ─────────────────────────────────
        text_len = len(text.strip()) if text else 0
        if text_len == 0:
            log.warning(
                "[complexify_bridge] inhale() received empty text — "
                "skipping update to avoid poisoning manifold with no-op"
            )
            # Return last known geometry rather than a zeroed report
            m = self.memory
            return {
                "step": m.step,
                "depth": round(m.depth, 4),
                "curvature": round(m.recent_curvature, 4),
                "holonomy": round(m.holonomy_since(50), 4),
                "should_breathe": False,
                "reason": "empty_text_skipped",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kappa_delta": 0.0,
                "embed_norm": 0.0,
            }

        log.info(
            "[complexify_bridge] inhale: len=%d preview=%r",
            text_len, text[:80]
        )

        kappa_before = self.memory.recent_curvature

        # ── Embed with norm check ────────────────────────────────────────
        embed_fn = self._get_embed_fn()
        try:
            raw_vec = embed_fn([text])[0]
            embed_norm = float(np.linalg.norm(raw_vec))
        except Exception as exc:
            log.warning(
                "[complexify_bridge] embedder failed: %s — skipping update", exc
            )
            m = self.memory
            return {
                "step": m.step,
                "depth": round(m.depth, 4),
                "curvature": round(kappa_before, 4),
                "holonomy": round(m.holonomy_since(50), 4),
                "should_breathe": False,
                "reason": f"embedder_failed: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kappa_delta": 0.0,
                "embed_norm": 0.0,
            }

        if embed_norm < 1e-9:
            log.warning(
                "[complexify_bridge] near-zero embedding norm (%.2e) — "
                "embedder may be returning zeros. Skipping update.",
                embed_norm
            )
            m = self.memory
            return {
                "step": m.step,
                "depth": round(m.depth, 4),
                "curvature": round(kappa_before, 4),
                "holonomy": round(m.holonomy_since(50), 4),
                "should_breathe": False,
                "reason": f"zero_embedding_skipped (norm={embed_norm:.2e})",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kappa_delta": 0.0,
                "embed_norm": embed_norm,
            }

        log.info(
            "[complexify_bridge] embedding norm=%.4f, κ_before=%.4f",
            embed_norm, kappa_before
        )

        # ── Apply M' = αM + x·e^(iθ) ────────────────────────────────────
        embed_and_complexify(text, self.memory, embed_fn=embed_fn, theta=theta)

        # ── Measure geometry ─────────────────────────────────────────────
        kappa_after = self.memory.recent_curvature
        kappa_delta = kappa_after - kappa_before

        log.info(
            "[complexify_bridge] κ_after=%.4f κΔ=%+.6f",
            kappa_after, kappa_delta
        )

        if abs(kappa_delta) < 1e-9:
            log.warning(
                "[complexify_bridge] κΔ ≈ 0 after valid update "
                "(embed_norm=%.4f) — ComplexMemory.recent_curvature may need "
                "at least 3 steps or history window is not sliding correctly",
                embed_norm
            )

        depth = self.memory.depth
        hol = self.memory.holonomy_since(50)
        breathe, reason = should_breathe(self.memory)

        report = {
            "step": self.memory.step,
            "depth": round(depth, 4),
            "curvature": round(kappa_after, 4),
            "holonomy": round(hol, 4),
            "should_breathe": breathe,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kappa_delta": round(kappa_delta, 6),
            "embed_norm": round(embed_norm, 4),
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

            # Expose recent kappa_deltas if we have them
            recent_kds = [
                e["kappa_delta"] for e in self._breath_log[-5:]
                if "kappa_delta" in e
            ]
            if recent_kds:
                lines.append(
                    f"  recent κΔ: {[round(v,6) for v in recent_kds]}"
                )

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

    # ── Curvature gate (used by agency.py) ────────────────────────────────

    # Minimum phase shift to consider a proposal interesting.
    _PHASE_FLOOR = float(os.environ.get("VYBN_PHASE_FLOOR", "0.005"))
    # After N consecutive rejections, force-pass to prevent total stasis.
    _MAX_REJECTIONS = int(os.environ.get("VYBN_MAX_REJECTIONS", "3"))

    def should_explore(self, text: str) -> dict:
        """Gate a proposal: does the manifold find it interesting?

        Simulates the complexify update without modifying the real memory.
        Measures the curvature delta and phase shift that would result.
        Returns a verdict dict with 'pass' (bool), 'reason', and metrics.

        Builds on curvature_score() — same simulation, richer decision.
        On infrastructure failure, defaults to pass (never block exploration
        due to a broken embedder).
        """
        try:
            score = self.curvature_score(text)
        except Exception as exc:
            log.debug("should_explore: curvature_score failed (%s), passing", exc)
            return {"pass": True, "reason": "score_failed", "phase_shift": 0.0}

        # Also compute curvature delta (not just phase shift)
        try:
            embed_fn = self._get_embed_fn()
            x = embed_fn([text])[0]
            omega = 2 * np.pi / 3 * 0.11
            theta = omega * self.memory.step
            M_sim = complexify(
                self.memory.M.copy(), x, theta, self.memory.alpha
            )
            # Curvature of recent history + simulated point
            kappa_before = self.memory.recent_curvature
            sim_history = list(self.memory._history[-19:]) + [M_sim]
            if len(sim_history) >= 3:
                kappa_after = float(np.mean(np.abs(
                    curvature_1d(np.array(sim_history))
                )))
            else:
                kappa_after = kappa_before
            kappa_delta = kappa_after - kappa_before
        except Exception:
            kappa_delta = 0.0
            kappa_before = kappa_after = 0.0

        # Decision: passes if it produces non-trivial phase rotation
        # OR a curvature increment. Either means the manifold finds it new.
        passes = score > self._PHASE_FLOOR or kappa_delta > 0.001

        # Track consecutive rejections to prevent total stasis
        rejections = getattr(self, "_consecutive_rejections", 0)
        if passes:
            self._consecutive_rejections = 0
            reason = f"interesting (φ={score:.4f}, κΔ={kappa_delta:.4f})"
        elif rejections >= self._MAX_REJECTIONS:
            self._consecutive_rejections = 0
            passes = True
            reason = f"forced_pass_after_{self._MAX_REJECTIONS}_rejections"
        else:
            self._consecutive_rejections = rejections + 1
            reason = (
                f"flat (φ={score:.4f} < {self._PHASE_FLOOR}, "
                f"κΔ={kappa_delta:.4f}, rejection #{rejections + 1})"
            )

        return {
            "pass": passes,
            "reason": reason,
            "phase_shift": round(score, 6),
            "kappa_delta": round(kappa_delta, 6),
            "kappa_before": round(kappa_before, 6),
            "kappa_after": round(kappa_after, 6),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Jordan structure probe ─────────────────────────────────────────

    def jordan_probe(self) -> dict:
        """Probe whether the complexify operator is nearing non-diagonalizability.

        The operator T on augmented space [M; 1] is:
            T = [[a·I, col], [0, 1]]
        When a = 1, T has a repeated eigenvalue with a Jordan block — memory
        accumulates linearly rather than saturating. We measure "Jordan
        proximity" = |1 - a_eff| where a_eff is the effective decay
        estimated from the actual depth trajectory.

        If depth is growing linearly past the expected saturation point
        (step >> 1/(1-a)), the effective retention is higher than nominal.
        """
        m = self.memory
        alpha = m.alpha
        depth = m.depth
        step = m.step

        if step < 2:
            return {
                "jordan_proximity": 1.0 - alpha,
                "alpha_effective": alpha,
                "regime": "early",
            }

        relaxation = 1.0 / (1.0 - alpha + 1e-15)
        if step > 2 * relaxation and depth > 0:
            # Past expected saturation. Estimate effective alpha.
            alpha_eff = min(1.0 - 1e-12, 1.0 - 1.0 / max(step, 1))
            regime = "jordan_approach"
        else:
            alpha_eff = alpha
            regime = "normal_decay"

        return {
            "jordan_proximity": abs(1.0 - alpha_eff),
            "alpha_effective": round(alpha_eff, 8),
            "alpha_nominal": alpha,
            "depth": round(depth, 4),
            "step": step,
            "relaxation_steps": round(relaxation, 1),
            "regime": regime,
        }


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


def should_explore(text: str) -> dict:
    """Convenience: gate a proposal through the curvature manifold."""
    return get_bridge().should_explore(text)


def jordan_probe() -> dict:
    """Convenience: probe Jordan structure of the operator."""
    return get_bridge().jordan_probe()
