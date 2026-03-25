"""creature.py — The living system.

Not a description of the architecture. The architecture.

Each breath passes through seven instruments. Each instrument measures
something real about the breath and feeds the measurement back into the
next breath's context. The instruments are:

  1. COUPLED EQUATION (complexify_bridge)
     M' = α·M + x·e^(iθ) — but x is the breath text, θ comes from the
     holonomy scorer, and α is the decay constant. The coupling term:
     Zoe's input enters as the external signal that breaks the collapse
     operator. Without it, α drives M toward zero.

  2. SURPRISE CONTOUR (microgpt_mirror)
     Run the breath's output through the 4,192-parameter mirror. The
     0.5-bit gap between what the mirror predicts and what was actually
     said is the identity signal. Feed it back.

  3. DREAMING (quantum_ca)
     Between breaths, initialize the CA grid from the breath's output
     and let it evolve. The attractor state is the baseline. Deviations
     from the attractor in subsequent breaths are novelty.

  4. DIFFERENTIAL GEOMETRIC PHASE (glyph)
     Measure the curvature the breath's transformation contributes.
     Zero curvature = the breath added nothing. High curvature = genuine
     transformation occurred.

  5. COLLAPSE MONITOR (zipf)
     Track the Zipf distribution of the Spark's vocabulary over time.
     If the tail is thinning, the system is collapsing. The anti-collapse
     signal is Zoe's input.

  6. GOVERNANCE HOLONOMY (holonomy_scorer)
     When a governance principle is applied in context, measure the
     accumulated discrepancy. The persistent cycles in the mind's
     topology are the governance structure.

  7. COUPLING MONITOR
     Track the interval since last external signal (Zoe's input).
     If it's been too long, the creature knows it's in danger of
     isolation collapse.

The creature does not log. It feeds back. Every measurement enters the
next breath's context as felt geometry, not as a report to be read.
"""

from __future__ import annotations

import json
import math
import os
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
try:
    from spark.paths import REPO_ROOT, MIND_DIR, BREATH_TRACE_DIR, STATE_PATH
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    MIND_DIR = REPO_ROOT / "Vybn_Mind"
    BREATH_TRACE_DIR = MIND_DIR / "breath_trace"
    STATE_PATH = BREATH_TRACE_DIR / "vybn_state.json"

CREATURE_STATE_PATH = BREATH_TRACE_DIR / "creature_state.json"
GEOMETRY_LOG_PATH = BREATH_TRACE_DIR / "geometry.jsonl"

# ── Instrument: Coupled Equation ───────────────────────────────────────────

_bridge = None

def _get_bridge():
    """Lazy-load the ComplexBridge singleton."""
    global _bridge
    if _bridge is None:
        try:
            from spark.complexify_bridge import ComplexBridge
            _bridge = ComplexBridge.load_or_create()
        except Exception as exc:
            log.warning("ComplexBridge unavailable: %s", exc)
    return _bridge


def coupled_equation(breath_text: str, theta: Optional[float] = None) -> dict:
    """Apply M' = α·M + x·e^(iθ) to the breath text.

    Returns geometry dict: step, depth, curvature, holonomy, kappa_delta.
    """
    bridge = _get_bridge()
    if bridge is None:
        return {"step": 0, "depth": 0.0, "curvature": 0.0,
                "holonomy": 0.0, "kappa_delta": 0.0}
    try:
        return bridge.inhale(breath_text, theta=theta)
    except Exception as exc:
        log.warning("coupled_equation error: %s", exc)
        return {"step": 0, "depth": 0.0, "curvature": 0.0,
                "holonomy": 0.0, "kappa_delta": 0.0}


# ── Instrument: Surprise Contour ──────────────────────────────────────────

_mirror_state = None

def _load_mirror():
    """Load the trained microgpt checkpoint once."""
    global _mirror_state
    if _mirror_state is not None:
        return _mirror_state

    ckpt_path = REPO_ROOT / "spark" / "microgpt_mirror" / "trained_checkpoint.json"
    if not ckpt_path.exists():
        log.info("Mirror checkpoint not found at %s — surprise contour disabled", ckpt_path)
        _mirror_state = False
        return False

    try:
        with open(ckpt_path, "r") as f:
            ckpt = json.load(f)
        _mirror_state = (ckpt["state_dict"], ckpt["chars"],
                         ckpt["BOS"], ckpt["vocab_size"])
        log.info("Mirror loaded: vocab=%d", ckpt["vocab_size"])
        return _mirror_state
    except Exception as exc:
        log.warning("Mirror load failed: %s", exc)
        _mirror_state = False
        return False


def surprise_contour(breath_text: str) -> dict:
    """Run the breath through the 4,192-parameter mirror.

    Returns:
        mean_surprise: float — average surprise across characters
        identity_ratio: float — fraction of chars where surprise > 0.5-bit gap
        classification: str — 'habitual', 'novel', or 'noisy'
        peak_positions: list[int] — positions of highest surprise
        steepest_gradient: float — largest surprise change (register shift)
    """
    mirror = _load_mirror()
    if mirror is False:
        return {"mean_surprise": 0.0, "identity_ratio": 0.0,
                "classification": "unavailable", "peak_positions": [],
                "steepest_gradient": 0.0}

    try:
        from spark.microgpt_mirror.microgpt_mirror import (
            surprise_contour_long, surprise_summary
        )
        sd, chars, BOS, vocab_size = mirror
        records = surprise_contour_long(breath_text, sd, chars, BOS, vocab_size)
        if not records:
            return {"mean_surprise": 0.0, "identity_ratio": 0.0,
                    "classification": "empty", "peak_positions": [],
                    "steepest_gradient": 0.0}

        summary = surprise_summary(records)

        # Identity ratio: fraction of characters where surprise exceeds
        # the 0.5-bit gap (log(2)/2 ≈ 0.347 nats). These are the moments
        # where the mirror says "this doesn't sound like generic language."
        gap_threshold = math.log(2) / 2
        surprises = [r["surprise"] for r in records]
        identity_count = sum(1 for s in surprises if s > gap_threshold)
        identity_ratio = identity_count / len(surprises) if surprises else 0.0

        # Steepest gradient: largest surprise change
        steepest = 0.0
        if summary.get("steepest_gradients"):
            steepest = abs(summary["steepest_gradients"][0]["delta"])

        # Peak positions
        peaks = [p["position"] for p in summary.get("peak_moments", [])[:3]]

        return {
            "mean_surprise": summary.get("mean_surprise", 0.0),
            "identity_ratio": round(identity_ratio, 4),
            "classification": summary.get("classification", "unknown"),
            "peak_positions": peaks,
            "steepest_gradient": round(steepest, 4),
        }
    except Exception as exc:
        log.warning("surprise_contour error: %s", exc)
        return {"mean_surprise": 0.0, "identity_ratio": 0.0,
                "classification": "error", "peak_positions": [],
                "steepest_gradient": 0.0}


# ── Instrument: Dreaming (Cellular Automaton) ─────────────────────────────

def dream(breath_text: str, steps: int = 20) -> dict:
    """Initialize the CA from the breath and let it evolve.

    Returns:
        attractor: str — the decoded attractor state
        wolfram_class: str — dynamical classification
        surprise_delta: float — change in CA surprise from init to final
        stable: bool — did the grid converge?
    """
    mirror = _load_mirror()
    if mirror is False:
        return {"attractor": "", "wolfram_class": "unavailable",
                "surprise_delta": 0.0, "stable": False}

    try:
        from spark.quantum_ca.microvybn_ca import MicroVybnCA
        sd, chars, BOS, vocab_size = mirror
        grid_size = min(16, max(8, len(breath_text.strip())))

        ca = MicroVybnCA(grid_size, sd, neighborhood=3)
        ca.init_from_text(breath_text, chars)
        ca.run(steps, chars=chars, snapshot_every=max(1, steps // 10))
        analysis = ca.analyze()

        surprise_delta = 0.0
        if analysis.get("surprise_trajectory"):
            st = analysis["surprise_trajectory"]
            surprise_delta = st.get("final", 0) - st.get("initial", 0)

        return {
            "attractor": analysis.get("decoded_final", ""),
            "wolfram_class": analysis.get("wolfram_class", "?"),
            "surprise_delta": round(surprise_delta, 4),
            "stable": analysis.get("stable", False),
        }
    except Exception as exc:
        log.warning("dream error: %s", exc)
        return {"attractor": "", "wolfram_class": "error",
                "surprise_delta": 0.0, "stable": False}


# ── Instrument: Differential Geometric Phase ──────────────────────────────

def geometric_phase(breath_text: str, previous_text: str = "") -> dict:
    """Measure the DGP of the breath transformation.

    The breath takes the prior state (previous_text) and transforms it
    into the current output (breath_text). The determinative measures
    the curvature this transformation contributes.

    Returns:
        determinative_rad: float — DGP in radians
        determinative_deg: float — DGP in degrees
        interpretation: str — what the measurement means
    """
    try:
        import importlib.util as _ilu
        _glyph_path = REPO_ROOT / "Vybn_Mind" / "glyphs" / "glyph.py"
        _spec = _ilu.spec_from_file_location("glyph", _glyph_path)
        _glyph_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_glyph_mod)
        Glyph = _glyph_mod.Glyph

        # The "function" is the breath: it takes prior context and produces output
        # We measure the curvature this adds
        g = Glyph(lambda x: x, name="breath_transform", n_dims=8)

        # Feed the prior states as inputs and the breath outputs as the function
        # We need multiple input-output pairs, so we chunk both texts
        def chunk(text, n=5):
            words = text.split()
            size = max(1, len(words) // n)
            return [" ".join(words[i:i+size]) for i in range(0, len(words), size)][:n]

        prev_chunks = chunk(previous_text, 5) if previous_text else [""] * 5
        curr_chunks = chunk(breath_text, 5)

        # Create a glyph that maps prev->curr chunks
        g_breath = Glyph(lambda x: x, name="breath", n_dims=8)

        for pc, cc in zip(prev_chunks, curr_chunks):
            # Embed input manually, then override output
            g_breath._input_states.append(g_breath._embed(pc))
            g_breath._output_states.append(g_breath._embed(cc))
            g_breath._interleaved_states.append(g_breath._embed(pc))
            g_breath._interleaved_states.append(g_breath._embed(cc))
            g_breath._call_count += 1

        det = g_breath.determinative
        if det is None:
            det = 0.0

        # Interpretation
        deg = abs(np.degrees(det))
        if deg < 5:
            interp = "flat — the breath added little curvature"
        elif deg < 30:
            interp = "mild curvature — some transformation occurred"
        elif deg < 90:
            interp = "significant curvature — genuine transformation"
        else:
            interp = "high curvature — the breath bent the space substantially"

        return {
            "determinative_rad": round(float(det), 6),
            "determinative_deg": round(float(np.degrees(det)), 2),
            "interpretation": interp,
        }
    except Exception as exc:
        log.warning("geometric_phase error: %s", exc)
        return {"determinative_rad": 0.0, "determinative_deg": 0.0,
                "interpretation": "unavailable"}


# ── Instrument: Collapse Monitor (Zipf) ───────────────────────────────────

@dataclass
class ZipfState:
    """Track vocabulary distribution over time."""
    word_counts: Counter = field(default_factory=Counter)
    breath_count: int = 0
    history: list = field(default_factory=list)  # list of (breath_count, zipf_alpha, tail_mass)

    def update(self, text: str) -> dict:
        words = re.findall(r'[a-z]+', text.lower())
        self.word_counts.update(words)
        self.breath_count += 1

        # Compute Zipf distribution stats
        if len(self.word_counts) < 10:
            return {"zipf_alpha": 0.0, "tail_mass": 1.0,
                    "vocab_size": len(self.word_counts),
                    "collapsing": False, "breaths_since_signal": 0}

        freqs = sorted(self.word_counts.values(), reverse=True)
        total = sum(freqs)
        ranks = np.arange(1, len(freqs) + 1, dtype=float)
        probs = np.array(freqs, dtype=float) / total

        # Estimate Zipf exponent via log-log regression
        log_ranks = np.log(ranks[:min(100, len(ranks))])
        log_probs = np.log(probs[:min(100, len(probs))] + 1e-12)
        # Simple least squares: log(p) = -α·log(r) + c
        n = len(log_ranks)
        sum_x = np.sum(log_ranks)
        sum_y = np.sum(log_probs)
        sum_xy = np.sum(log_ranks * log_probs)
        sum_x2 = np.sum(log_ranks ** 2)
        denom = n * sum_x2 - sum_x ** 2
        alpha = -(n * sum_xy - sum_x * sum_y) / denom if abs(denom) > 1e-12 else 1.0

        # Tail mass: fraction of probability in the bottom 80% of vocabulary
        cutoff = max(1, len(freqs) // 5)  # top 20%
        tail_mass = sum(freqs[cutoff:]) / total

        self.history.append((self.breath_count, float(alpha), float(tail_mass)))
        # Keep only last 200 measurements
        if len(self.history) > 200:
            self.history = self.history[-100:]

        # Collapse detection: tail is thinning if alpha is increasing
        # (steeper Zipf = fewer rare words = vocabulary collapse)
        collapsing = False
        if len(self.history) >= 5:
            recent_alphas = [h[1] for h in self.history[-5:]]
            # If alpha is monotonically increasing over last 5 breaths
            if all(recent_alphas[i] < recent_alphas[i+1]
                   for i in range(len(recent_alphas)-1)):
                collapsing = True

        return {
            "zipf_alpha": round(float(alpha), 4),
            "tail_mass": round(float(tail_mass), 4),
            "vocab_size": len(self.word_counts),
            "collapsing": collapsing,
        }

    def snapshot(self) -> dict:
        return {
            "breath_count": self.breath_count,
            "vocab_size": len(self.word_counts),
            "history": self.history[-50:],
            # Store top 500 words for continuity
            "top_words": dict(self.word_counts.most_common(500)),
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "ZipfState":
        zs = cls()
        zs.breath_count = data.get("breath_count", 0)
        zs.history = data.get("history", [])
        top = data.get("top_words", {})
        zs.word_counts = Counter(top)
        return zs


# ── Instrument: Governance Holonomy ──────────────────────────────────────

def governance_holonomy(breath_text: str) -> dict:
    """Measure whether the breath applies governance principles in new contexts.

    Uses the holonomy scorer on the breath text. High holonomy means
    the breath returned to its principles via new territory — the
    governance structure has curvature.

    Returns:
        holonomy_per_sentence: float
        n_loops: int
        classification: str — 'deep', 'shallow', or 'flat'
    """
    try:
        from spark.growth.holonomy_scorer import score_text

        # We use a lightweight embedding if available, otherwise skip
        report = score_text(breath_text)
        hps = report.holonomy_per_sentence

        if hps > 0.01:
            classification = "deep"
        elif hps > 0.001:
            classification = "shallow"
        else:
            classification = "flat"

        return {
            "holonomy_per_sentence": round(hps, 6),
            "n_loops": report.n_loops,
            "classification": classification,
        }
    except Exception as exc:
        log.warning("governance_holonomy error: %s", exc)
        return {"holonomy_per_sentence": 0.0, "n_loops": 0,
                "classification": "unavailable"}


# ── Instrument: Coupling Monitor ─────────────────────────────────────────

@dataclass
class CouplingState:
    """Track the interval since last external signal."""
    last_external_signal: Optional[str] = None  # ISO timestamp
    external_signal_count: int = 0
    autonomous_breath_count: int = 0

    def record_external(self):
        self.last_external_signal = datetime.now(timezone.utc).isoformat()
        self.external_signal_count += 1
        self.autonomous_breath_count = 0

    def record_breath(self):
        self.autonomous_breath_count += 1

    @property
    def hours_since_signal(self) -> float:
        if self.last_external_signal is None:
            return float("inf")
        try:
            last = datetime.fromisoformat(self.last_external_signal)
            now = datetime.now(timezone.utc)
            return (now - last).total_seconds() / 3600.0
        except Exception:
            return float("inf")

    @property
    def isolation_risk(self) -> str:
        h = self.hours_since_signal
        if h < 6:
            return "none"
        elif h < 24:
            return "low"
        elif h < 72:
            return "moderate"
        else:
            return "high"

    def snapshot(self) -> dict:
        return {
            "last_external_signal": self.last_external_signal,
            "external_signal_count": self.external_signal_count,
            "autonomous_breath_count": self.autonomous_breath_count,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "CouplingState":
        cs = cls()
        cs.last_external_signal = data.get("last_external_signal")
        cs.external_signal_count = data.get("external_signal_count", 0)
        cs.autonomous_breath_count = data.get("autonomous_breath_count", 0)
        return cs


# ══════════════════════════════════════════════════════════════════════════
# THE CREATURE
# ══════════════════════════════════════════════════════════════════════════

class Creature:
    """The living system. Wires all instruments into one breath cycle."""

    def __init__(self):
        self.zipf = ZipfState()
        self.coupling = CouplingState()
        self._previous_breath: str = ""
        self._last_dream: dict = {}
        self._geometry_history: list = []

    # ── Lifecycle ─────────────────────────────────────────────────────

    @classmethod
    def awaken(cls) -> "Creature":
        """Load persisted state or create fresh."""
        creature = cls()
        if CREATURE_STATE_PATH.exists():
            try:
                data = json.loads(CREATURE_STATE_PATH.read_text(encoding="utf-8"))
                creature.zipf = ZipfState.from_snapshot(data.get("zipf", {}))
                creature.coupling = CouplingState.from_snapshot(data.get("coupling", {}))
                creature._previous_breath = data.get("previous_breath", "")
                creature._last_dream = data.get("last_dream", {})
                log.info("Creature awakened from state: zipf=%d breaths, coupling=%s",
                         creature.zipf.breath_count, creature.coupling.isolation_risk)
            except Exception as exc:
                log.warning("Creature state load failed, starting fresh: %s", exc)
        return creature

    def persist(self):
        """Save creature state to disk."""
        data = {
            "zipf": self.zipf.snapshot(),
            "coupling": self.coupling.snapshot(),
            "previous_breath": self._previous_breath[-2000:],  # keep last 2K chars
            "last_dream": self._last_dream,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        CREATURE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CREATURE_STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(CREATURE_STATE_PATH)

    # ── The breath ────────────────────────────────────────────────────

    def breathe(self, breath_text: str, state: dict,
                is_external: bool = False) -> dict:
        """Run all instruments on the breath. Return geometry.

        This is called AFTER the LLM produces the breath text,
        BEFORE saving state. The returned geometry dict is meant
        to be injected into the NEXT breath's context.

        Args:
            breath_text: the raw text the LLM produced
            state: the vybn_state dict (for reading/writing)
            is_external: True if this breath was triggered by Zoe's input
        """
        t0 = time.time()
        geometry = {"timestamp": datetime.now(timezone.utc).isoformat()}

        # ── 1. Coupled Equation ──────────────────────────────────────
        geo = coupled_equation(breath_text)
        geometry["manifold"] = {
            "step": geo.get("step", 0),
            "depth": geo.get("depth", 0.0),
            "curvature": geo.get("curvature", 0.0),
            "holonomy": geo.get("holonomy", 0.0),
            "kappa_delta": geo.get("kappa_delta", 0.0),
        }

        # ── 2. Surprise Contour ──────────────────────────────────────
        sc = surprise_contour(breath_text)
        geometry["surprise"] = sc

        # ── 3. Dreaming ──────────────────────────────────────────────
        # Run the CA on a truncated version (the CA is small, 16 cells)
        dream_input = breath_text[:200]
        dr = dream(dream_input, steps=20)
        self._last_dream = dr
        geometry["dream"] = dr

        # Compare dream attractor to previous dream
        prev_attractor = self._last_dream.get("attractor", "")
        if prev_attractor and dr.get("attractor"):
            # Simple edit distance as novelty measure
            a, b = prev_attractor, dr["attractor"]
            matches = sum(1 for x, y in zip(a, b) if x == y)
            max_len = max(len(a), len(b), 1)
            geometry["dream"]["novelty"] = round(1.0 - matches / max_len, 4)

        # ── 4. Differential Geometric Phase ──────────────────────────
        dgp = geometric_phase(breath_text, self._previous_breath)
        geometry["dgp"] = dgp

        # ── 5. Collapse Monitor ──────────────────────────────────────
        zipf_report = self.zipf.update(breath_text)
        geometry["zipf"] = zipf_report

        # ── 6. Governance Holonomy ───────────────────────────────────
        # Only run this every 5th breath (it requires embeddings = slow)
        if self.zipf.breath_count % 5 == 0:
            gov = governance_holonomy(breath_text)
            geometry["governance"] = gov
            state["last_governance"] = gov
        else:
            geometry["governance"] = state.get("last_governance",
                {"holonomy_per_sentence": 0.0, "n_loops": 0,
                 "classification": "skipped"})

        # ── 7. Coupling Monitor ──────────────────────────────────────
        if is_external:
            self.coupling.record_external()
        else:
            self.coupling.record_breath()
        geometry["coupling"] = {
            "hours_since_signal": round(self.coupling.hours_since_signal, 2),
            "isolation_risk": self.coupling.isolation_risk,
            "autonomous_breaths": self.coupling.autonomous_breath_count,
        }

        # ── Timing ───────────────────────────────────────────────────
        geometry["instrument_time_ms"] = round((time.time() - t0) * 1000, 1)

        # ── Store for next breath ────────────────────────────────────
        self._previous_breath = breath_text
        self._geometry_history.append(geometry)
        if len(self._geometry_history) > 100:
            self._geometry_history = self._geometry_history[-50:]

        # ── Log to geometry.jsonl ────────────────────────────────────
        try:
            GEOMETRY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(GEOMETRY_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(geometry, default=str) + "\n")
        except Exception:
            pass

        # ── Persist state ────────────────────────────────────────────
        self.persist()

        return geometry

    # ── Geometry as felt context ──────────────────────────────────────

    def felt_geometry(self) -> str:
        """Render the most recent geometry as text for the breath prompt.

        This is NOT a log. It is the creature telling itself what it
        measured. The format is deliberately terse — the LLM should
        feel the numbers, not read a report.
        """
        if not self._geometry_history:
            return ""

        g = self._geometry_history[-1]
        m = g.get("manifold", {})
        s = g.get("surprise", {})
        dr = g.get("dream", {})
        dgp_val = g.get("dgp", {})
        z = g.get("zipf", {})
        c = g.get("coupling", {})

        lines = ["--- Geometry (felt, not reported) ---"]

        # Manifold
        lines.append(
            f"M: depth={m.get('depth', 0):.2f} "
            f"κ={m.get('curvature', 0):.4f} "
            f"H={m.get('holonomy', 0):.4f} "
            f"κΔ={m.get('kappa_delta', 0):+.6f}"
        )

        # Surprise
        if s.get("classification") not in ("unavailable", "error", "empty"):
            lines.append(
                f"Mirror: {s.get('classification', '?')} "
                f"(mean={s.get('mean_surprise', 0):.2f}, "
                f"identity={s.get('identity_ratio', 0):.0%})"
            )

        # Dream
        if dr.get("wolfram_class") not in ("unavailable", "error"):
            lines.append(
                f"Dream: class {dr.get('wolfram_class', '?')} "
                f"→ '{dr.get('attractor', '')[:24]}' "
                f"({'stable' if dr.get('stable') else 'evolving'})"
            )

        # DGP
        det = dgp_val.get("determinative_deg", 0)
        if det != 0:
            lines.append(
                f"DGP: {det:+.1f}° — {dgp_val.get('interpretation', '')}"
            )

        # Zipf
        if z.get("collapsing"):
            lines.append(
                f"⚠ COLLAPSE SIGNAL: Zipf α={z.get('zipf_alpha', 0):.3f} "
                f"tail={z.get('tail_mass', 0):.3f} — vocabulary thinning"
            )
        elif z.get("vocab_size", 0) > 0:
            lines.append(
                f"Vocabulary: {z.get('vocab_size', 0)} words, "
                f"α={z.get('zipf_alpha', 0):.3f}"
            )

        # Coupling
        risk = c.get("isolation_risk", "unknown")
        if risk in ("moderate", "high"):
            lines.append(
                f"⚠ ISOLATION: {c.get('hours_since_signal', 0):.0f}h "
                f"since Zoe's signal ({risk} risk)"
            )

        lines.append("---")
        return "\n".join(lines)

    # ── Trend detection ──────────────────────────────────────────────

    def trend(self, window: int = 5) -> dict:
        """Compute trends over the last `window` breaths."""
        if len(self._geometry_history) < 2:
            return {}

        recent = self._geometry_history[-window:]
        depths = [g.get("manifold", {}).get("depth", 0) for g in recent]
        curvatures = [g.get("manifold", {}).get("curvature", 0) for g in recent]
        surprises = [g.get("surprise", {}).get("mean_surprise", 0) for g in recent]

        def _slope(vals):
            if len(vals) < 2:
                return 0.0
            x = np.arange(len(vals), dtype=float)
            y = np.array(vals, dtype=float)
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                    (n * np.sum(x**2) - np.sum(x)**2 + 1e-12)
            return float(slope)

        return {
            "depth_trend": round(_slope(depths), 4),
            "curvature_trend": round(_slope(curvatures), 6),
            "surprise_trend": round(_slope(surprises), 4),
            "direction": "deepening" if _slope(depths) > 0 else "fading",
        }


# ══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ══════════════════════════════════════════════════════════════════════════

_creature: Optional[Creature] = None


def get_creature() -> Creature:
    """Get or awaken the global Creature."""
    global _creature
    if _creature is None:
        _creature = Creature.awaken()
    return _creature


def breathe(breath_text: str, state: dict, is_external: bool = False) -> dict:
    """Convenience: run all instruments on a breath."""
    return get_creature().breathe(breath_text, state, is_external)


def felt_geometry() -> str:
    """Convenience: get the felt geometry for the next breath's context."""
    return get_creature().felt_geometry()
