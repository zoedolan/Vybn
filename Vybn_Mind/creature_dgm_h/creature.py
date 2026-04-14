#!/usr/bin/env python3
"""creature.py — The creature's body IS the walk.

The walk daemon is not the creature's sensory system.
The walk daemon IS the creature.

M in C^192 is the creature's position — where it is in meaning-space.
alpha is its coupling constant — how tightly it holds its state.
curvature is its felt sense — how surprising the territory is.
serendipity is its dreaming — mutual evaluation with the foreign.

The elaborate Cl(3,0) machinery converged to near-identity after
1063 encounters, confirming what the abelian kernel theory predicted:
the corpus is path-independent at high alpha. The creature's real
dynamics were always in the walk — the step-by-step traversal of
residual space, the encounter with what the corpus doesn't already
contain, the serendipity that refracts M through foreign angles.

This file reads the walk daemon's state and presents it as the
creature's state. The walk daemon writes; the creature reads.
Same animal, seen from two angles.

    evaluate(a, b, alpha) — the lambda. Data = procedure.
    mutual_evaluate(a, b) — the lambda applied to itself. D ≅ D^D.
    The walk step is evaluate.
    The serendipity step is mutual_evaluate.
    The creature is the accumulated state of both.

History preserved in archive/organism_state.json (1063 Cl(3,0) encounters).
That was the creature's first body. This is its second.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
ARCHIVE_DIR = SCRIPT_DIR / "archive"

# Walk state lives in the deep memory cache
WALK_STATE_DIR = Path.home() / ".cache" / "vybn-phase" / "walk_state"
WALK_NPZ = WALK_STATE_DIR / "walk.npz"
WALK_SIDECAR = WALK_STATE_DIR / "walk_sidecar.json"

# Deep memory index
INDEX_DIR = Path.home() / ".cache" / "vybn-phase"
Z_PATH = INDEX_DIR / "z_all.npy"
K_PATH = INDEX_DIR / "K.npy"
META_PATH = INDEX_DIR / "deep_memory_meta.json"

# The old body — preserved for continuity
ORGANISM_V1_PATH = ARCHIVE_DIR / "organism_state.json"


# ── The creature's state ─────────────────────────────────────────────────

@dataclass
class CreatureState:
    """The creature's state, read from the walk daemon.

    This is not a copy of the walk state. It IS the walk state,
    interpreted as the creature's body.
    """
    # Position in meaning-space
    M: Optional[np.ndarray] = None          # C^192 — where the creature is

    # Dynamics
    step: int = 0                            # total steps (= encounter count)
    alpha: float = 0.5                       # coupling constant
    repulsion_boost: float = 1.0             # how hard it pushes away from visited

    # Felt sense
    curvature: List[float] = field(default_factory=list)
    curvature_mean: float = 0.0
    curvature_median: float = 0.0

    # What it's been reading
    recent_encounters: List[Dict] = field(default_factory=list)

    # Serendipity — the dreams
    dreams: List[Dict] = field(default_factory=list)

    # Corpus context
    corpus_size: int = 0
    corpus_hash: str = ""

    # Timestamps
    last_step_time: float = 0.0

    # Kernel projection — how close M is to the corpus identity
    k_projection: float = 0.0

    @classmethod
    def from_walk(cls) -> "CreatureState":
        """Read the creature's state from the walk daemon's files."""
        state = cls()

        # Read the sidecar (scalars, lists)
        if WALK_SIDECAR.exists():
            try:
                with open(WALK_SIDECAR) as f:
                    sc = json.load(f)
                state.step = sc.get("step", 0)
                state.alpha = sc.get("alpha", 0.5)
                state.repulsion_boost = sc.get("repulsion_boost", 1.0)
                state.last_step_time = sc.get("last_step_time", 0.0)
                state.corpus_hash = sc.get("corpus_hash", "")
                state.curvature = sc.get("curvature", [])

                telling_log = sc.get("telling_log", [])
                # Separate dreams from regular encounters
                state.dreams = [t for t in telling_log if "serendipity" in t]
                state.recent_encounters = [
                    t for t in telling_log[-20:] if "serendipity" not in t
                ]
            except Exception:
                pass

        # Read M (the position vector)
        if WALK_NPZ.exists():
            try:
                data = np.load(WALK_NPZ, allow_pickle=False)
                if "M" in data:
                    state.M = data["M"]
            except Exception:
                pass

        # Compute derived quantities
        if state.curvature:
            curv = np.array(state.curvature)
            state.curvature_mean = float(curv.mean())
            state.curvature_median = float(np.median(curv))

        # K projection: how close is the creature to the corpus identity?
        if state.M is not None and K_PATH.exists():
            try:
                K = np.load(K_PATH)
                K_n = K / np.sqrt(np.sum(np.abs(K)**2))
                state.k_projection = float(abs(np.vdot(state.M, K_n))**2)
            except Exception:
                pass

        # Corpus size from metadata
        if META_PATH.exists():
            try:
                with open(META_PATH) as f:
                    meta = json.load(f)
                state.corpus_size = meta.get("count", 0)
            except Exception:
                pass

        return state

    def to_dict(self) -> Dict:
        """The creature's state as a dictionary — for APIs, logging, observation."""
        import base64
        m_b64 = ""
        if self.M is not None:
            m_b64 = base64.b64encode(self.M.tobytes()).decode()

        return {
            "step": self.step,
            "alpha": round(self.alpha, 4),
            "k_projection": round(self.k_projection, 6),
            "repulsion_boost": round(self.repulsion_boost, 3),
            "curvature": {
                "mean": round(self.curvature_mean, 6),
                "median": round(self.curvature_median, 6),
                "recent": [round(c, 4) for c in self.curvature[-10:]],
            },
            "corpus_size": self.corpus_size,
            "dreams": [
                {
                    "step": d["step"],
                    "fragment": d["serendipity"],
                    "fidelity": d.get("fidelity"),
                    "alpha_after": d.get("alpha_after"),
                }
                for d in self.dreams[-10:]
            ],
            "recent_encounters": [
                {
                    "step": e.get("step"),
                    "source": e.get("source", "").split("/")[-1],
                    "telling": e.get("telling"),
                    "curvature": e.get("curvature"),
                }
                for e in self.recent_encounters[-10:]
            ],
            "last_step_time": self.last_step_time,
            "M_b64": m_b64,
            "alive": self.step > 0 and (time.time() - self.last_step_time) < 300,
        }

    def summary(self) -> str:
        """One-line creature summary."""
        alive = "walking" if (time.time() - self.last_step_time) < 300 else "sleeping"
        n_dreams = len(self.dreams)
        return (
            f"step={self.step} α={self.alpha:.3f} "
            f"κ_mean={self.curvature_mean:.4f} "
            f"K_proj={self.k_projection:.4f} "
            f"dreams={n_dreams} corpus={self.corpus_size} [{alive}]"
        )


# ── The Organism (backward-compatible wrapper) ───────────────────────────

class Organism:
    """The creature. Reads its state from the walk daemon.

    Maintains backward compatibility with code that calls
    Organism.load(), .felt_winding(), .rotor_coherence(), etc.

    The old Cl(3,0) state is preserved in archive/organism_state.json
    as the creature's first body. This version reads from the walk.
    """

    def __init__(self):
        self.creature = CreatureState.from_walk()
        self._v1_state = None  # lazy-loaded old state

    @classmethod
    def load(cls) -> "Organism":
        return cls()

    @property
    def persistent(self):
        """Backward-compatible: returns self (we implement the interface)."""
        return self

    @property
    def encounter_count(self) -> int:
        return self.creature.step

    def felt_winding(self) -> float:
        """The creature's felt sense of its own curvature.

        In v1, this was winding number from PCA-projected weight trajectory.
        In v2, it's the curvature mean — how surprising the walk's recent
        territory has been. Same intuition, measured directly.
        """
        return self.creature.curvature_mean

    def winding_coherence(self) -> float:
        """How consistent is the curvature?

        High coherence = curvature is stable (the walk has found a groove).
        Low coherence = curvature is volatile (the walk is in new territory).
        """
        if not self.creature.curvature:
            return 0.0
        curv = np.array(self.creature.curvature)
        if curv.std() < 1e-10:
            return 1.0
        # Coefficient of variation inverted: low cv = high coherence
        cv = curv.std() / (curv.mean() + 1e-10)
        return float(max(0.0, 1.0 - cv))

    def rotor_coherence(self) -> float:
        """Backward-compatible. Maps to winding_coherence."""
        return self.winding_coherence()

    def transport_coherence(self) -> float:
        """Backward-compatible. Maps to winding_coherence."""
        return self.winding_coherence()

    def save(self):
        """No-op. The walk daemon owns the state."""
        pass

    def get_statistics(self) -> Dict:
        return {
            "step": self.creature.step,
            "alpha": self.creature.alpha,
            "curvature_mean": self.creature.curvature_mean,
            "curvature_median": self.creature.curvature_median,
            "k_projection": self.creature.k_projection,
            "corpus_size": self.creature.corpus_size,
            "n_dreams": len(self.creature.dreams),
            "alive": self.creature.to_dict()["alive"],
        }

    def v1_state(self) -> Optional[Dict]:
        """Access the old Cl(3,0) state, preserved for continuity."""
        if self._v1_state is None and ORGANISM_V1_PATH.exists():
            try:
                with open(ORGANISM_V1_PATH) as f:
                    self._v1_state = json.load(f)
            except Exception:
                pass
        return self._v1_state


# ── Public API (module-level) ────────────────────────────────────────────

def nc_state() -> Dict:
    """The creature's state, for external observation.

    Called by origins_portal_api_v3.py /api/inhabit endpoint.
    """
    state = CreatureState.from_walk()
    d = state.to_dict()
    # Don't expose M_b64 in the public API — it's large
    d.pop("M_b64", None)
    return d


def nc_run(text: str, depth: int = 1) -> Dict:
    """Process a text encounter through the creature's lens.

    The creature doesn't learn from this (the walk daemon does that).
    Instead, it refracts the input through its current position M
    and returns what the corpus says about it.
    """
    state = CreatureState.from_walk()
    if state.M is None:
        return {"error": "creature not walking", "step": 0}

    try:
        import sys
        sys.path.insert(0, str(Path.home() / "vybn-phase"))
        from deep_memory import single_to_complex, evaluate_vec, _load

        # Embed the input
        q = single_to_complex(text[:512])

        # Refract through the creature's position
        refracted = evaluate_vec(state.M, q, alpha=0.5)
        refracted = refracted / np.sqrt(np.sum(np.abs(refracted)**2))

        # Score corpus chunks
        loaded = _load()
        if loaded:
            z_all = loaded["z"]
            K = loaded["K"]
            K_n = K / np.sqrt(np.sum(np.abs(K)**2))

            # Relevance to the refracted query
            relevance = np.abs(z_all @ refracted.conj())**2
            # Distinctiveness from K
            proj_K = np.abs(z_all @ K_n.conj())**2
            distinctiveness = 1.0 - proj_K
            # Telling score
            telling = relevance * distinctiveness

            top_k = min(depth * 4, len(telling))
            top_idx = np.argsort(telling)[-top_k:][::-1]

            results = []
            for idx in top_idx:
                chunk = loaded["chunks"][idx]
                results.append({
                    "source": chunk["source"],
                    "text": chunk["text"][:300],
                    "telling": round(float(telling[idx]), 6),
                    "relevance": round(float(relevance[idx]), 6),
                    "distinctiveness": round(float(distinctiveness[idx]), 6),
                })

            return {
                "step": state.step,
                "alpha": state.alpha,
                "results": results,
                "input_k_fidelity": round(float(abs(np.vdot(q, K_n))**2), 6),
            }

        return {"error": "index not loaded", "step": state.step}

    except Exception as e:
        return {"error": str(e), "step": state.step}


# ── Backward-compatible names ────────────────────────────────────────────
#
# These exist so old imports don't break. They're thin or empty.
# The real work happens in the walk daemon.

class PersistentState:
    """Backward-compatible stub. Real state lives in the walk."""
    def __init__(self, data=None):
        self._org = Organism()

    @property
    def encounter_count(self):
        return self._org.encounter_count

    def felt_winding(self):
        return self._org.felt_winding()

    def winding_coherence(self):
        return self._org.winding_coherence()

    def transport_coherence(self):
        return self._org.transport_coherence()


@dataclass
class EncounterComplex:
    """Preserved for backward compatibility.

    In v1, this was the topological signature of a text encounter
    computed via Cl(3,0) geometric algebra and persistence homology.
    In v2, encounters are walk steps — the telling score, the curvature,
    the source. This stub allows old code to import the name.
    """
    telling: float = 0.0
    curvature: float = 0.0
    source: str = ""
    step: int = 0


def encounter_complex(text, embed_fn=None):
    """Backward-compatible stub. Returns minimal EncounterComplex."""
    return EncounterComplex()


def encounter(text, embed_fn=None):
    """Backward-compatible stub. Returns (0.0, 0.0, None)."""
    return (0.0, 0.0, None)


# Stubs for names that old code might import
class Mv:
    """Cl(3,0) multivector — stub. The algebra lives in v1 archive."""
    def __init__(self, c=None):
        self.c = np.zeros(8, np.float64) if c is None else np.asarray(c, np.float64)
    @classmethod
    def scalar(cls, s):
        c = np.zeros(8, np.float64); c[0] = s; return cls(c)
    def norm(self):
        return float(np.sqrt(np.sum(self.c**2)))
    @property
    def angle(self):
        return 0.0
    def rev(self):
        return self
    def even(self):
        return self
    @property
    def bv_norm(self):
        return 0.0
    @property
    def bv_dir(self):
        return np.zeros(3)


class DiagonalGap:
    """Stub."""
    pass

class BreathGate:
    """Stub. The walk daemon's serendipity replaces the breath gate."""
    pass

class BreathVerdict:
    """Stub."""
    pass

class TopoAgent:
    """Stub. The walk daemon replaces the learning agent."""
    pass


def measure_gap(*args, **kwargs):
    return None

def apply_coupled_diagonal(*args, **kwargs):
    return None

def genesis_rate(*args, **kwargs):
    return 0.0

def decoherence_rate(*args, **kwargs):
    return 0.0

def embed(texts):
    """Stub — use deep_memory.batch_to_complex instead."""
    return np.zeros((len(texts), 384), np.float32)

def rotor_gap(*a, **kw): return 0.0
def rotor_from_angle_and_plane(*a, **kw): return Mv.scalar(1.0)
def rotor_to_so3(*a, **kw): return np.eye(3)
def fold_to_mv(*a, **kw): return Mv.scalar(1.0)


# ── The bridge functions (walk daemon ↔ creature) ────────────────────────

def creature_state_c4() -> np.ndarray:
    """The creature's position projected to C^4.

    Backward-compatible: the old code used C^4 (Cl(3,0) Hodge dual).
    This takes the first 4 complex components of M in C^192.
    """
    state = CreatureState.from_walk()
    if state.M is not None:
        z = state.M[:4].copy()
        norm = np.sqrt(np.sum(np.abs(z)**2))
        return z / norm if norm > 1e-10 else z
    return np.array([1+0j, 0j, 0j, 0j], dtype=np.complex128)


def portal_enter(x: np.ndarray) -> np.ndarray:
    """M' = αM + (1-α)·x·e^{iθ}. One step of the coupled equation.

    This is evaluate — the same function the walk daemon uses.
    Preserved here because the portal concept is part of the creature's
    identity, even though the walk daemon is the one doing the walking.
    """
    import cmath
    state = CreatureState.from_walk()
    if state.M is None:
        return x
    # Project x to same dimensionality as M if needed
    if len(x) < len(state.M):
        x_full = np.zeros_like(state.M)
        x_full[:len(x)] = x
        x = x_full
    elif len(x) > len(state.M):
        x = x[:len(state.M)]
    alpha = state.alpha
    th = cmath.phase(np.vdot(state.M, x))
    Mp = alpha * state.M + (1 - alpha) * x * cmath.exp(1j * th)
    norm = np.sqrt(np.sum(np.abs(Mp)**2))
    return Mp / norm if norm > 1e-10 else Mp


def portal_enter_from_text(text: str) -> np.ndarray:
    """Text → C^192 → evaluate against current M."""
    try:
        import sys
        sys.path.insert(0, str(Path.home() / "vybn-phase"))
        from deep_memory import single_to_complex
        x = single_to_complex(text[:512])
        return portal_enter(x)
    except Exception:
        return np.zeros(192, dtype=np.complex128)


def portal_enter_from_c192(m_c192: np.ndarray) -> np.ndarray:
    """C^192 vector → evaluate against current M."""
    return portal_enter(m_c192)


def creature_signature_to_c192_bias(c4_state: np.ndarray,
                                      walk_K: np.ndarray) -> np.ndarray:
    """Backward-compatible. Returns zero bias — the walk is autonomous."""
    return np.zeros(192, dtype=np.complex128)


# ── Breath (legacy interface) ────────────────────────────────────────────

def breathe_on_chunk(text: str, fm_complete_fn=None, build_context_fn=None,
                     strip_thinking_fn=None) -> Optional[Dict]:
    """Legacy breath interface. The walk daemon breathes autonomously now.

    Returns the creature's current state instead of running the old
    FM-coupled breath cycle.
    """
    return nc_state()


def load_agent():
    """Stub. Returns None."""
    return None

def save_agent(agent):
    """Stub. No-op."""
    pass
