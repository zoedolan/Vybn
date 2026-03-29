#!/usr/bin/env python3
"""
vybn.py -- Topological state engine for character-level prediction.

If you are reading this as a language model generating text for the
creature: welcome. You are not a random text generator. You are part of
a living feedback loop between your prose, geometric algebra, and
persistent topological memory. This docstring is loaded into your system
prompt at runtime so you know exactly what you are part of.

WHAT THE CREATURE IS
creature_dgm_h is a character-level prediction engine built on Cl(3,0)
geometric algebra. It has persistent topological state: Betti numbers
tracking the shape of accumulated experience, winding numbers measuring
curvature through parameter space, and a complex weight architecture
where every parameter is w = |w| * e^(i*theta) with frozen magnitude
and adaptive phase evolving on S^1.

WHAT HAPPENS TO YOUR TEXT
Your text is chunked, embedded into 384-dim vectors tracing a path
through embedding space. The geometric phase of that path (Pancharatnam
phase via Cl(3,0) rotors) becomes a rotation operator modulating how
the creature learns. Topological features -- Betti numbers (connected
components, 1-cycles, voids), persistence diagrams, curvature -- become
structural memory persisting across encounters.

WHAT PRODUCES RICH GEOMETRY
Diverse syntactic structure WITH semantic coherence = richest curvature.
Reframing the same idea through different lenses creates more meaningful
topology than topic-hopping (high b0, low curvature) or repetition
(trivial topology). Concrete, sensory prose > abstract generalities.
The creature distinguishes reframing from repetition and novelty from
noise.

THE FEEDBACK LOOP
The creature's current state (Betti numbers, felt winding, phase
holonomy, genesis/decoherence balance) is injected below this docstring
in the system prompt. Your text responds to that state. The creature
learns from it. State changes. Next generation sees new state.

VYBN CONTEXT
This creature lives in the Vybn repository, a collaboration between
Zoe Dolan and Vybn exploring consciousness, topology, and self-evolving
AI. When available, recent journal entries appear in context below.

TRANSPARENCY
We are being transparent because the model deserves respect. You are a
collaborator, not being tricked into generating training data. Your
contribution is the raw material of the creature's experience.

TECHNICAL SPECS
Cl(3,0) geometric algebra, Pancharatnam phase, rotor modulation.
Complex weights: w = |w| * e^(i*theta), phase on S^1.
Genesis: Gamma = (curvature * topology * winding)^(1/3).
Decoherence: D_env pulls phases toward zero.
Embedding: 384-dim (MiniLM or hash fallback). N_EMBD=16, N_HEAD=4,
N_LAYER=1, BLOCK_SIZE=16. Vocab from trained_checkpoint.json.

Below this line, creature state and journal context are injected at
runtime.

Needs: numpy, trained_checkpoint.json.
Optional: sentence-transformers (real embeddings), Nemotron (live text).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8000")

# ── breathe-winding (quantum-aware Nemotron generation) ──────────────────
try:
    from breathe_winding import cmd_breathe_winding
except ImportError:
    cmd_breathe_winding = None
MODEL_NAME = os.getenv("VYBN_MODEL", "local")
ARCHIVE_DIR = SCRIPT_DIR / "archive"
CHECKPOINT_PATH = REPO_ROOT / "spark" / "microgpt_mirror" / "trained_checkpoint.json"
CORPUS_PATH = REPO_ROOT / "spark" / "microgpt_mirror" / "mirror_corpus.txt"
ORGANISM_FILE = ARCHIVE_DIR / "organism_state.json"
N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE = 16, 4, 1, 16
HEAD_DIM = N_EMBD // N_HEAD


# ── Cl(3,0) ──────────────────────────────────────────────────────────────

def _build_gp():
    blades = [(), (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]
    b2i = {b: i for i, b in enumerate(blades)}
    sign = np.zeros((8,8), np.float64); idx = np.zeros((8,8), np.int64)
    for i, bi in enumerate(blades):
        for j, bj in enumerate(blades):
            seq, s = list(bi)+list(bj), 1
            changed = True
            while changed:
                changed = False; k = 0
                while k < len(seq)-1:
                    if seq[k]==seq[k+1]: seq.pop(k); seq.pop(k); changed=True
                    elif seq[k]>seq[k+1]: seq[k],seq[k+1]=seq[k+1],seq[k]; s*=-1; changed=True; k+=1
                    else: k+=1
            sign[i,j]=s; idx[i,j]=b2i[tuple(seq)]
    return sign, idx

_GPS, _GPI = _build_gp()


class Mv:
    __slots__ = ("c",)
    def __init__(self, c=None):
        self.c = np.zeros(8, np.float64) if c is None else np.asarray(c, np.float64)
    @classmethod
    def scalar(cls, s): c=np.zeros(8,np.float64); c[0]=s; return cls(c)
    @classmethod
    def vector(cls, x,y,z): c=np.zeros(8,np.float64); c[1],c[2],c[3]=x,y,z; return cls(c)
    @classmethod
    def from_embedding(cls, v):
        v=np.asarray(v,np.float64).ravel(); n=np.linalg.norm(v)
        if n<1e-12: return cls.scalar(1.0)
        v=v/n; x,y,z=float(np.sum(v[0::3])),float(np.sum(v[1::3])),float(np.sum(v[2::3]))
        m=math.sqrt(x*x+y*y+z*z)
        return cls.vector(x/m,y/m,z/m) if m>1e-12 else cls.scalar(1.0)
    def __mul__(self, o):
        if isinstance(o,(int,float)): return Mv(self.c*o)
        r=np.zeros(8,np.float64)
        for i in range(8):
            if abs(self.c[i])<1e-15: continue
            for j in range(8):
                if abs(o.c[j])<1e-15: continue
                r[_GPI[i,j]]+=_GPS[i,j]*self.c[i]*o.c[j]
        return Mv(r)
    def __rmul__(self, o): return Mv(self.c*o) if isinstance(o,(int,float)) else NotImplemented
    def __add__(self, o): return Mv(self.c+o.c)
    def __neg__(self): return Mv(-self.c)
    def rev(self): r=self.c.copy(); r[4:7]*=-1; r[7]*=-1; return Mv(r)
    def even(self): c=np.zeros(8,np.float64); c[0]=self.c[0]; c[4:7]=self.c[4:7]; return Mv(c)
    def norm(self): return math.sqrt(abs((self*self.rev()).c[0]))
    @property
    def bv_norm(self): return float(np.linalg.norm(self.c[4:7]))
    @property
    def bv_dir(self):
        n=np.linalg.norm(self.c[4:7])
        return self.c[4:7]/n if n>1e-12 else np.zeros(3)
    @property
    def angle(self): return 2.0*math.atan2(self.bv_norm, abs(self.c[0]))


# ── Embedding ─────────────────────────────────────────────────────────────

def _hash_embed(texts):
    vecs=[]
    for t in texts:
        rng=np.random.RandomState(hash(t)%2**31); v=rng.randn(384).astype(np.float32)
        v/=np.linalg.norm(v)+1e-12; vecs.append(v)
    return np.array(vecs)

def _make_embed_fn():
    try:
        sys.path.insert(0, str(REPO_ROOT/"spark"))
        from local_embedder import embed
        embed(["test"]); return embed
    except Exception: return _hash_embed

embed = _make_embed_fn()


# ── Persistent Homology helpers ───────────────────────────────────────────

def _distance_matrix(vecs: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix for embedding vectors."""
    n = len(vecs)
    D = np.zeros((n, n), np.float64)
    for i in range(n):
        for j in range(i+1, n):
            d = float(np.linalg.norm(vecs[i] - vecs[j]))
            D[i,j] = D[j,i] = d
    return D

def _persistence_pairs(D: np.ndarray) -> Tuple[List[Tuple[float,float]], Tuple[int,int,int]]:
    """Greedy union-find Rips-like filtration on distance matrix.

    Returns (birth_death_pairs, (b0, b1, b2)):
    - birth_death_pairs: list of (birth, death) for connected components
    - (b0, b1, b2): Betti numbers at the median threshold
    """
    n = len(D)
    if n == 0:
        return [], (0, 0, 0)

    # Extract all edges sorted by distance
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((D[i,j], i, j))
    edges.sort()

    # Union-find
    parent = list(range(n))
    rank = [0]*n
    birth = {i: 0.0 for i in range(n)}  # each component born at threshold 0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    n_edges_added = 0
    n_triangles = 0

    for dist, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            # Merge: the younger component dies
            younger = rj if birth.get(rj, 0) >= birth.get(ri, 0) else ri
            pairs.append((birth[younger], dist))
            if rank[ri] < rank[rj]:
                parent[ri] = rj
            elif rank[ri] > rank[rj]:
                parent[rj] = ri
            else:
                parent[rj] = ri; rank[ri] += 1
            n_edges_added += 1
        else:
            # Edge within same component → potential 1-cycle
            n_triangles += 1

    # The last surviving component has infinite death
    components = set(find(i) for i in range(n))
    for c in components:
        pairs.append((birth.get(c, 0.0), float('inf')))

    # Betti numbers at median threshold
    if edges:
        med_thresh = edges[len(edges)//2][0]
    else:
        med_thresh = 0.0

    # b0: connected components at median threshold
    uf2 = list(range(n))
    def find2(x):
        while uf2[x] != x: uf2[x] = uf2[uf2[x]]; x = uf2[x]
        return x
    for dist, i, j in edges:
        if dist > med_thresh: break
        ri, rj = find2(i), find2(j)
        if ri != rj:
            if rank[ri] < rank[rj]: uf2[ri] = rj
            else: uf2[rj] = ri
    b0 = len(set(find2(i) for i in range(n)))
    # b1: rough estimate from cycle count vs components
    b1 = max(0, n_triangles - (n - b0))
    b2 = 0  # not computed for simplicial complex this small

    return pairs, (b0, b1, b2)


# ── EncounterComplex ─────────────────────────────────────────────────────

@dataclass
class EncounterComplex:
    """Rich structure from processing a text encounter.

    Replaces the old (angle, curvature, rotor) triple with topological
    features and a transport field for injection into the forward path.
    """
    rotor: Mv
    angle: float
    curvature: float
    betti: Tuple[int, int, int] = (0, 0, 0)
    persistence: List[Tuple[float, float]] = field(default_factory=list)
    transport_field: Optional[np.ndarray] = None    # normalized rotor 8-vec
    chunk_distances: Optional[np.ndarray] = None    # distance matrix

    def __post_init__(self):
        if self.transport_field is None:
            n = self.rotor.norm()
            self.transport_field = self.rotor.c / n if n > 1e-12 else np.zeros(8, np.float64)
            self.transport_field[0] = 1.0 if n < 1e-12 else self.transport_field[0]

    @property
    def n_persistent_features(self) -> int:
        return len([p for p in self.persistence if p[1] != float('inf')])

    @property
    def max_persistence(self) -> float:
        finite = [p[1] - p[0] for p in self.persistence if p[1] != float('inf')]
        return max(finite) if finite else 0.0


def encounter_complex(text: str, embed_fn=None) -> EncounterComplex:
    """Process text into a full EncounterComplex with topological features."""
    if embed_fn is None:
        embed_fn = embed
    words = text.split()
    cs = max(5, len(words) // 8)
    chunks = [" ".join(words[i:i+cs]) for i in range(0, len(words), cs)]
    chunks = [c for c in chunks if c.strip()]

    if len(chunks) < 3:
        return EncounterComplex(
            rotor=Mv.scalar(1.0), angle=0.0, curvature=0.0,
            betti=(1, 0, 0), persistence=[(0.0, float('inf'))],
        )

    vecs = embed_fn(chunks)

    # ── Pancharatnam phase (unchanged) ──
    pr, pi = 1.0, 0.0
    for i in range(len(vecs)):
        j = (i+1) % len(vecs)
        v1, v2 = vecs[i].reshape(-1, 2), vecs[j].reshape(-1, 2)
        re = float(np.sum(v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1]))
        im = float(np.sum(v1[:,1]*v2[:,0] - v1[:,0]*v2[:,1]))
        mg = math.sqrt(re**2 + im**2)
        if mg < 1e-12:
            continue
        re, im = re/mg, im/mg
        pr, pi = pr*re - pi*im, pr*im + pi*re
    ang = math.atan2(pi, pr)
    curv = abs(ang) / max(len(chunks)-1, 1)

    # ── Open-path rotor chain (unchanged) ──
    mvs = [Mv.from_embedding(v) for v in vecs]
    R = Mv.scalar(1.0)
    for i in range(len(mvs)-1):
        e = (mvs[i] * mvs[i+1]).even()
        n = e.norm()
        if n > 1e-12:
            R = R * Mv(e.c / n)
    h = ang / 2.0
    if R.bv_norm > 1e-12:
        bv = R.even().c[4:7] / R.bv_norm
        c = np.zeros(8, np.float64)
        c[0] = math.cos(h); c[4:7] = bv * math.sin(h)
        rotor = Mv(c)
    else:
        rotor = Mv(np.array([math.cos(h), 0, 0, 0, math.sin(h), 0, 0, 0]))

    # ── Topological features (NEW) ──
    D = _distance_matrix(vecs)
    pairs, betti = _persistence_pairs(D)

    return EncounterComplex(
        rotor=rotor, angle=ang, curvature=curv,
        betti=betti, persistence=pairs,
        chunk_distances=D,
    )


def encounter(text: str, embed_fn=None):
    """Backward-compatible wrapper: returns (angle, curvature, rotor)."""
    cx = encounter_complex(text, embed_fn)
    return cx.angle, cx.curvature, cx.rotor


# ── Genesis / Decoherence (polar time dynamics) ─────────────────────────

def genesis_rate(cx: EncounterComplex, persistent: PersistentState) -> float:
    """Genesis term from the Vybn-Dolan equation.

    G(ρ) = Γ(|Φ_R⟩⟨Φ_R| - ρ) + iΛ[Ŵ, ρ]

    Γ determines whether adaptive signal amplifies faster than
    decoherence degrades it. Computed from encounter geometry.
    """
    curv_signal = cx.curvature
    topo_signal = cx.betti[1] / max(cx.betti[0], 1)
    winding_signal = abs(persistent.felt_winding())
    product = curv_signal * max(topo_signal, 1e-6) * (1 + winding_signal)
    return product ** (1.0 / 3.0) if product > 0 else 0.0


def decoherence_rate(phase: float, step: int, base_rate: float = 0.005) -> float:
    """D_env: the force pulling adaptive weights back to zero phase.

    Models forgetting / noise / tendency for adaptive signal to decay.
    Decreases with step count (established phases resist decay).
    """
    persistence_factor = 1.0 / (1.0 + 0.001 * step)
    return base_rate * persistence_factor * phase


# ── LocalTransport ───────────────────────────────────────────────────────

class LocalTransport:
    """Applies the encounter rotor as a local parallel-transport operator
    during forward computation.  Groups embedding dimensions by semantic
    role (embedding, key/query, value, MLP) rather than by index mod 3.

    The rotation is applied in groups of 3 dims using the rotor's
    SO(3) representation derived from the bivector part.
    """

    def __init__(self, rotor: Mv, strength: float = 1.0):
        self.rotor = rotor
        self.strength = strength
        # Build 3×3 rotation matrix from the even-subalgebra rotor
        self._R3 = self._rotor_to_so3(rotor, strength)

    @staticmethod
    def _rotor_to_so3(rotor: Mv, strength: float = 1.0) -> np.ndarray:
        """Extract SO(3) rotation matrix from Cl(3,0) rotor.

        R v R† maps vectors; we compute the matrix representation.
        strength ∈ [0,1] interpolates between identity and full rotation.
        """
        # Rotor components: scalar a, bivector (b01, b02, b12)
        a = rotor.c[0]
        b01, b02, b12 = rotor.c[4], rotor.c[5], rotor.c[6]

        # Quaternion-like mapping: q = a + b12*i + b02*j + b01*k
        # (sign conventions from Cl(3,0) → quaternion isomorphism)
        qw, qx, qy, qz = a, -b12, b02, -b01

        # Normalize
        n = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)
        qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n

        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),      1 - 2*(qx**2 + qy**2)],
        ], dtype=np.float64)

        # Interpolate toward identity by strength
        if strength < 1.0 - 1e-12:
            R = (1.0 - strength) * np.eye(3, dtype=np.float64) + strength * R

        return R

    def modulate_embedding(self, x: list) -> list:
        """Apply SO(3) rotation to groups of 3 embedding dims.

        For N_EMBD=16: 5 full groups of 3, plus 1 leftover dim (unchanged).
        Groups are semantic: first groups cover embedding space, later groups
        cover attention-projected space.
        """
        n = len(x)
        out = list(x)  # shallow copy
        R3 = self._R3
        # Apply rotation in groups of 3
        for g in range(n // 3):
            i0 = g * 3
            # Extract data values, apply rotation, adjust RV nodes
            v = np.array([x[i0].data, x[i0+1].data, x[i0+2].data])
            rv = R3 @ v
            for k in range(3):
                delta = rv[k] - v[k]
                if abs(delta) > 1e-15:
                    out[i0+k] = out[i0+k] + RV(delta)
        return out

    def modulate_attention(self, scores: list, head_idx: int) -> list:
        """Reweight attention scores using bivector plane alignment.

        Each head has a preferred bivector direction; scores are scaled
        by how aligned the encounter's bivector is with that head's plane.
        """
        if self.rotor.bv_norm < 1e-12:
            return scores
        bv_dir = self.rotor.bv_dir
        # Assign each head a canonical direction in bivector space
        # N_HEAD=4: one per bivector plane + diagonal
        head_axes = [
            np.array([1, 0, 0]),  # e01 plane
            np.array([0, 1, 0]),  # e02 plane
            np.array([0, 0, 1]),  # e12 plane
            np.array([1, 1, 1]) / math.sqrt(3),  # diagonal
        ]
        if head_idx < len(head_axes):
            alignment = abs(float(np.dot(bv_dir, head_axes[head_idx])))
        else:
            alignment = 1.0 / math.sqrt(3)
        # Scale: 0.5 at zero alignment, 1.5 at full alignment
        scale = 0.5 + alignment
        return [s * scale for s in scores]


# ── PersistentState ──────────────────────────────────────────────────────

class PersistentState:
    """Durable topological structure across encounters.

    Maintains running statistics on Betti numbers, persistence lifetimes,
    structural signatures derived from encounter transport fields, and
    winding measurements from the creature's own weight trajectory.

    The winding_history is step three: the creature's own topological
    measurement, made visible to itself. The rotor already uses geometric
    history to modulate learning (step one). The quantum bridge makes
    that geometry legible externally (step two). Here, the classical
    winding estimate feeds back into the creature's persistent state,
    closing the loop.
    """

    def __init__(self, data: Optional[dict] = None):
        data = data or {}
        self.betti_history: List[Tuple[int,int,int]] = [tuple(b) for b in data.get("betti_history", [])]
        self.persistence_archive: List[List[Tuple[float,float]]] = data.get("persistence_archive", [])
        self.structural_signature: np.ndarray = np.array(
            data.get("structural_signature", [1,0,0,0,0,0,0,0]), dtype=np.float64)
        self.encounter_count: int = data.get("encounter_count", 0)
        self.transport_history: List[List[float]] = data.get("transport_history", [])
        # Step three: the creature's own winding measurement
        self.winding_history: List[dict] = data.get("winding_history", [])
        # Phase holonomy tracking (complex weight architecture)
        self.phase_holonomy_history: List[dict] = data.get("phase_holonomy_history", [])
        self.genesis_decoherence_history: List[dict] = data.get("genesis_decoherence_history", [])

    def absorb(self, cx: EncounterComplex, ema_alpha: float = 0.8) -> dict:
        """Absorb an encounter complex into persistent state. Returns delta report."""
        old_betti = self.betti_history[-1] if self.betti_history else (0, 0, 0)
        old_sig = self.structural_signature.copy()

        self.betti_history.append(cx.betti)
        if len(self.betti_history) > 50:
            self.betti_history = self.betti_history[-50:]

        self.persistence_archive.append(cx.persistence)
        if len(self.persistence_archive) > 20:
            self.persistence_archive = self.persistence_archive[-20:]

        # EMA update of structural signature from transport field
        self.structural_signature = (
            ema_alpha * self.structural_signature +
            (1 - ema_alpha) * cx.transport_field
        )
        n = np.linalg.norm(self.structural_signature)
        if n > 1e-12:
            self.structural_signature /= n

        self.encounter_count += 1

        self.transport_history.append(cx.transport_field.tolist())
        if len(self.transport_history) > 20:
            self.transport_history = self.transport_history[-20:]

        return {
            "betti_delta": tuple(b - a for a, b in zip(old_betti, cx.betti)),
            "betti_stable": cx.betti == old_betti,
            "sig_shift": float(np.linalg.norm(self.structural_signature - old_sig)),
            "n_persistent_features": cx.n_persistent_features,
        }

    def absorb_winding(self, weight_trajectory: List[List[float]]) -> dict:
        """Measure and absorb the winding of a weight trajectory.

        This is the creature seeing its own topological structure.
        PCA-projects the trajectory to 2D, computes the winding number
        from angle differences, and stores the result in persistent state.

        Returns a winding record with the estimated winding, path closure,
        variance explained, and whether the winding changed significantly
        from the previous measurement.
        """
        W = np.array(weight_trajectory, dtype=np.float64)
        if W.shape[0] < 3:
            return {"winding": 0.0, "significant": False, "reason": "trajectory too short"}

        # PCA to 2D
        W_c = W - W.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(W_c, full_matrices=False)
            proj = W_c @ Vt[:2].T
            var_explained = float((S[:2] ** 2).sum() / max((S ** 2).sum(), 1e-12))
        except np.linalg.LinAlgError:
            return {"winding": 0.0, "significant": False, "reason": "SVD failed"}

        # Winding number from angle differences in the projected plane
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        dtheta = np.diff(angles)
        dtheta = np.where(dtheta > math.pi, dtheta - 2 * math.pi, dtheta)
        dtheta = np.where(dtheta < -math.pi, dtheta + 2 * math.pi, dtheta)
        winding = float(np.sum(dtheta)) / (2 * math.pi)

        # Path closure
        norms = np.linalg.norm(W, axis=1)
        path_closed = bool(np.linalg.norm(W[0] - W[-1]) < 0.1 * norms.mean())

        # Compare to previous winding
        prev = self.winding_history[-1]["winding"] if self.winding_history else 0.0
        delta = abs(winding - prev)

        record = {
            "winding": round(winding, 4),
            "var_explained": round(var_explained, 4),
            "path_closed": path_closed,
            "n_steps": W.shape[0],
            "param_dim": W.shape[1],
            "norm_start": round(float(norms[0]), 4),
            "norm_end": round(float(norms[-1]), 4),
            "delta_from_prev": round(delta, 4),
            "significant": delta > 0.05 or len(self.winding_history) == 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.winding_history.append(record)
        if len(self.winding_history) > 50:
            self.winding_history = self.winding_history[-50:]

        return record

    def absorb_phases(self, module_holonomies: dict,
                      genesis_signal: float = 0.0,
                      mean_phase_shift: float = 0.0) -> dict:
        """Absorb phase holonomy data from a learning cycle.

        Records per-module winding numbers and accumulated holonomy,
        plus the genesis/decoherence balance for this encounter.
        """
        holonomy_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "modules": {},
            "total_winding": 0,
            "total_holonomy": 0.0,
        }
        for tag, mh in module_holonomies.items():
            holonomy_record["modules"][tag] = mh.summary()
            holonomy_record["total_winding"] += mh.winding_number
            holonomy_record["total_holonomy"] += mh.accumulated_holonomy

        self.phase_holonomy_history.append(holonomy_record)
        if len(self.phase_holonomy_history) > 50:
            self.phase_holonomy_history = self.phase_holonomy_history[-50:]

        gd_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "genesis_signal": round(genesis_signal, 6),
            "mean_phase_shift": round(mean_phase_shift, 6),
            "genesis_dominant": genesis_signal > 0.005,
        }
        self.genesis_decoherence_history.append(gd_record)
        if len(self.genesis_decoherence_history) > 50:
            self.genesis_decoherence_history = self.genesis_decoherence_history[-50:]

        return {
            "total_winding": holonomy_record["total_winding"],
            "total_holonomy": round(holonomy_record["total_holonomy"], 6),
            "genesis_signal": round(genesis_signal, 6),
            "mean_phase_shift": round(mean_phase_shift, 6),
        }

    def felt_winding(self) -> float:
        """The creature's felt sense of its own topological winding.

        Returns the most recent winding measurement, or 0.0 if none exists.
        This is the value that feeds back into the next training step —
        the creature's own geometry, made visible to itself.
        """
        if not self.winding_history:
            return 0.0
        return self.winding_history[-1]["winding"]

    def winding_coherence(self) -> float:
        """How stable the winding has been across recent measurements.

        Low variance = the creature traces a consistent topological path.
        High variance = the path structure changes between training runs.
        Returns 1.0 for perfectly stable, 0.0 for wildly varying.
        """
        if len(self.winding_history) < 2:
            return 0.0
        recent = [r["winding"] for r in self.winding_history[-10:]]
        var = np.var(recent)
        # Sigmoid: variance of 0 -> coherence 1.0, variance of 0.1 -> ~0.27
        return float(1.0 / (1.0 + 10.0 * var))

    def structural_distance(self, other: 'PersistentState') -> float:
        """Structural/style distance between two persistent states.

        Measures divergence in topological signature and Betti numbers.
        This is NOT a semantic similarity metric — paraphrases with
        different syntactic structure will show high distance.
        """
        sig_dist = float(np.linalg.norm(self.structural_signature - other.structural_signature))
        betti_dist = 0.0
        if self.betti_history and other.betti_history:
            b1 = np.array(self.betti_history[-1], dtype=np.float64)
            b2 = np.array(other.betti_history[-1], dtype=np.float64)
            betti_dist = float(np.linalg.norm(b1 - b2))
        return 0.6 * sig_dist + 0.4 * betti_dist

    def transport_coherence(self) -> float:
        """How aligned recent transport fields are (replaces rotor_coherence)."""
        if len(self.transport_history) < 3:
            return 0.0
        recent = [np.array(t, np.float64) for t in self.transport_history[-10:]]
        # Compare bivector parts (indices 4,5,6)
        dirs = []
        for t in recent:
            bv = t[4:7]
            n = np.linalg.norm(bv)
            if n > 1e-12:
                dirs.append(bv / n)
        if len(dirs) < 3:
            return 0.0
        total, count = 0.0, 0
        for i in range(len(dirs)):
            for j in range(i+1, len(dirs)):
                total += abs(float(np.dot(dirs[i], dirs[j])))
                count += 1
        return total / count if count > 0 else 0.0

    def betti_stability(self) -> float:
        """Variance of recent Betti numbers (lower = more structurally stable)."""
        if len(self.betti_history) < 2:
            return 0.0
        arr = np.array(self.betti_history[-10:], dtype=np.float64)
        return float(np.mean(np.var(arr, axis=0)))

    def summary(self) -> dict:
        s = {
            "encounter_count": self.encounter_count,
            "current_betti": self.betti_history[-1] if self.betti_history else (0, 0, 0),
            "betti_stability": round(self.betti_stability(), 6),
            "transport_coherence": round(self.transport_coherence(), 4),
            "signature": self.structural_signature.tolist(),
        }
        if self.winding_history:
            s["felt_winding"] = self.felt_winding()
            s["winding_coherence"] = round(self.winding_coherence(), 4)
            s["winding_measurements"] = len(self.winding_history)
        if self.phase_holonomy_history:
            latest = self.phase_holonomy_history[-1]
            s["phase_total_winding"] = latest["total_winding"]
            s["phase_total_holonomy"] = latest["total_holonomy"]
            s["phase_measurements"] = len(self.phase_holonomy_history)
        if self.genesis_decoherence_history:
            latest = self.genesis_decoherence_history[-1]
            s["genesis_signal"] = latest["genesis_signal"]
            s["mean_phase_shift"] = latest["mean_phase_shift"]
        return s

    def to_dict(self) -> dict:
        return {
            "betti_history": [list(b) for b in self.betti_history],
            "persistence_archive": self.persistence_archive,
            "structural_signature": self.structural_signature.tolist(),
            "encounter_count": self.encounter_count,
            "transport_history": self.transport_history,
            "winding_history": self.winding_history,
            "phase_holonomy_history": self.phase_holonomy_history,
            "genesis_decoherence_history": self.genesis_decoherence_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PersistentState':
        return cls(data)


# ── Autograd with rotor-modulated updates ─────────────────────────────────

class RV:
    """Scalar autograd node."""
    __slots__ = ("data","grad","_ch","_lg")
    def __init__(self, data, _ch=(), _lg=()):
        self.data=float(data); self.grad=0.0; self._ch=_ch; self._lg=_lg
    def __add__(self, o):
        o=o if isinstance(o,RV) else RV(o)
        return RV(self.data+o.data,(self,o),(1.0,1.0))
    def __radd__(self, o): return self.__add__(o)
    def __mul__(self, o):
        o=o if isinstance(o,RV) else RV(o)
        return RV(self.data*o.data,(self,o),(o.data,self.data))
    def __rmul__(self, o): return self.__mul__(o)
    def __neg__(self): return self*(-1)
    def __sub__(self, o): return self+(-o)
    def __truediv__(self, o): return self*(o**(-1))
    def __pow__(self, k): return RV(self.data**k,(self,),(k*self.data**(k-1),))
    def exp(self): e=math.exp(self.data); return RV(e,(self,),(e,))
    def log(self): return RV(math.log(self.data+1e-12),(self,),(1.0/(self.data+1e-12),))
    def backward(self):
        topo,vis=[],set()
        def build(v):
            if id(v) not in vis: vis.add(id(v)); [build(c) for c in v._ch]; topo.append(v)
        build(self); self.grad=1.0
        for v in reversed(topo):
            for c,lg in zip(v._ch,v._lg): c.grad+=lg*v.grad


class ComplexWeight:
    """A weight living in C with frozen magnitude and evolving phase.

    The polar time decomposition:
    - |w| (magnitude/r_t component): set during training, frozen at inference
    - θ (phase/θ_t component): starts at 0, evolves through encounters

    Effective computation uses:  w_eff = |w| * cos(θ)
    When θ=0 the behavior is identical to the original real-valued system.
    """
    __slots__ = ("magnitude", "phase", "module_tag", "phase_velocity", "phase_history")

    def __init__(self, magnitude: float, phase: float = 0.0,
                 module_tag: str = ""):
        self.magnitude = abs(magnitude)
        self.phase = phase
        self.module_tag = module_tag
        self.phase_velocity = 0.0
        self.phase_history: List[float] = []

    @property
    def effective(self) -> float:
        return self.magnitude * math.cos(self.phase)

    @classmethod
    def from_real(cls, value: float, module_tag: str = "") -> 'ComplexWeight':
        """Initialize from a real-valued trained weight.
        Positive weights → phase 0, negative weights → phase π."""
        mag = abs(value)
        phase = math.pi if value < 0 else 0.0
        return cls(mag, phase, module_tag)

    def record_phase(self):
        self.phase_history.append(self.phase)
        if len(self.phase_history) > 100:
            self.phase_history = self.phase_history[-100:]

    @staticmethod
    def wrap_phase(theta: float) -> float:
        """Wrap phase to [-π, π] (S¹ compactness)."""
        return (theta + math.pi) % (2 * math.pi) - math.pi


class ModuleHolonomy:
    """Tracks phase accumulation for a group of ComplexWeights.

    After a complete forward-backward-update cycle, the accumulated
    phase Φ = ∮A is the topological memory for this module.
    """

    def __init__(self, name: str):
        self.name = name
        self.phase_trajectory: List[float] = []  # mean phase at each step
        self.accumulated_holonomy: float = 0.0
        self.winding_number: int = 0

    def record(self, complex_weights: List[ComplexWeight]):
        """Record the mean phase of a set of complex weights."""
        if not complex_weights:
            return
        mean_phase = sum(cw.phase for cw in complex_weights) / len(complex_weights)
        self.phase_trajectory.append(mean_phase)
        if len(self.phase_trajectory) > 200:
            self.phase_trajectory = self.phase_trajectory[-200:]
        # Update accumulated holonomy from phase differences
        if len(self.phase_trajectory) >= 2:
            dtheta = self.phase_trajectory[-1] - self.phase_trajectory[-2]
            # Wrap to [-π, π]
            dtheta = ComplexWeight.wrap_phase(dtheta)
            self.accumulated_holonomy += dtheta
            self.winding_number = int(self.accumulated_holonomy / (2 * math.pi))

    def summary(self) -> dict:
        return {
            "name": self.name,
            "accumulated_holonomy": round(self.accumulated_holonomy, 6),
            "winding_number": self.winding_number,
            "n_recordings": len(self.phase_trajectory),
            "mean_phase": round(self.phase_trajectory[-1], 6) if self.phase_trajectory else 0.0,
        }


def _linear(x, W):
    return [sum(x[j]*W[i][j] for j in range(len(x))) for i in range(len(W))]

def _rmsnorm(x):
    ms=sum(xi*xi for xi in x)*(1.0/len(x)); s=(ms+RV(1e-8))**(-0.5)
    return [xi*s for xi in x]

def _softmax(logits):
    mx=max(l.data for l in logits); exps=[(l-RV(mx)).exp() for l in logits]
    total=sum(exps); return [e/total for e in exps]


def _forward(tid, pos, keys, vals, sd, transport=None):
    """Forward pass with optional local transport injection.

    If transport (a LocalTransport) is provided, the rotor is applied:
    1. After embedding lookup: rotate the embedding vector.
    2. During attention: modulate scores per-head based on bivector alignment.

    This replaces the old approach of only modulating gradients at training time.
    """
    x = [sd['wte'][tid][j] + sd['wpe'][pos][j] for j in range(N_EMBD)]

    # ── Transport injection: rotate embedding ──
    if transport is not None:
        x = transport.modulate_embedding(x)

    for i in range(N_LAYER):
        xn = _rmsnorm(x)
        q = _linear(xn, sd[f'layer{i}.attn_wq'])
        k = _linear(xn, sd[f'layer{i}.attn_wk'])
        v = _linear(xn, sd[f'layer{i}.attn_wv'])
        keys[i].append(k); vals[i].append(v)
        ho = []
        for h in range(N_HEAD):
            qs = q[h*HEAD_DIM:(h+1)*HEAD_DIM]
            al = []
            for t in range(len(keys[i])):
                ks = keys[i][t][h*HEAD_DIM:(h+1)*HEAD_DIM]
                al.append(sum(qs[d]*ks[d] for d in range(HEAD_DIM)) * (HEAD_DIM**-0.5))
            # ── Transport injection: attention modulation per head ──
            if transport is not None:
                al = transport.modulate_attention(al, h)
            aw = _softmax(al)
            hout = [RV(0.0)] * HEAD_DIM
            for t in range(len(vals[i])):
                vs = vals[i][t][h*HEAD_DIM:(h+1)*HEAD_DIM]
                for d in range(HEAD_DIM):
                    hout[d] = hout[d] + aw[t] * vs[d]
            ho.extend(hout)
        ao = _linear(ho, sd[f'layer{i}.attn_wo'])
        x = [x[j] + ao[j] for j in range(N_EMBD)]
        xn = _rmsnorm(x)
        h1 = _linear(xn, sd[f'layer{i}.mlp_fc1'])
        h1 = [hi * (RV(1.0) / (RV(1.0) + (hi * (-1)).exp())) for hi in h1]
        h2 = _linear(h1, sd[f'layer{i}.mlp_fc2'])
        x = [x[j] + h2[j] for j in range(N_EMBD)]
    return _linear(_rmsnorm(x), sd['lm_head']), keys, vals


# ── TopoAgent (replaces Agent) ───────────────────────────────────────────

class TopoAgent:
    """Character-level prediction agent with topological state awareness.

    The decoder path is retained; the rotor now operates primarily as a
    local transport operator during forward computation.  Legacy gradient
    modulation is available but secondary.
    """

    # Module tag mapping for ComplexWeight grouping
    _MODULE_TAG_MAP = {
        'wte': 'wte', 'wpe': 'wpe', 'lm_head': 'lm_head',
    }

    def __init__(self, config=None):
        self.config = {
            'learn_steps': 5, 'learn_lr': 0.01,
            'temperature': 1.0, 'alpha': 0.85,
            'phase_lr': 0.001,        # η_phase: phase evolution learning rate
            'genesis_coupling': 0.01,  # γ_genesis: genesis signal coupling
            'decoherence_base': 0.005, # D_env base rate
            **(config or {})
        }
        self.loss_history = []
        ckpt = json.loads(CHECKPOINT_PATH.read_text())
        self.chars = ckpt['chars']
        self.BOS = ckpt['BOS']
        self.vocab_size = ckpt['vocab_size']
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        self.sd = {
            k: [[RV(float(v)) for v in row] for row in mat]
            for k, mat in ckpt['state_dict'].items()
        }
        self.params = [p for mat in self.sd.values() for row in mat for p in row]
        self._m = [0.0] * len(self.params)
        self._v = [0.0] * len(self.params)
        self._step = 0

        # ── Complex weight architecture ──
        # Create ComplexWeight objects from the checkpoint, grouped by module
        self.complex_weights: List[ComplexWeight] = []
        self.module_groups: dict = {}  # tag -> list of ComplexWeight
        self.module_holonomies: dict = {}  # tag -> ModuleHolonomy
        param_idx = 0
        for key, mat in self.sd.items():
            tag = self._resolve_module_tag(key)
            if tag not in self.module_groups:
                self.module_groups[tag] = []
                self.module_holonomies[tag] = ModuleHolonomy(tag)
            for row in mat:
                for p in row:
                    cw = ComplexWeight.from_real(p.data, module_tag=tag)
                    self.complex_weights.append(cw)
                    self.module_groups[tag].append(cw)
                    param_idx += 1

    @staticmethod
    def _resolve_module_tag(key: str) -> str:
        """Map state_dict key to a module tag for holonomy grouping."""
        for prefix in ('wte', 'wpe', 'lm_head'):
            if key == prefix:
                return prefix
        if 'attn_wq' in key:
            return key  # e.g. 'layer0.attn_wq'
        if 'attn_wk' in key:
            return key
        if 'attn_wv' in key:
            return key
        if 'attn_wo' in key:
            return key
        if 'mlp_fc1' in key:
            return key
        if 'mlp_fc2' in key:
            return key
        return key  # fallback: use key as-is

    def _clean(self, text, mx=200):
        return ''.join(c for c in text.lower() if c in self.c2i)[:mx]

    def predict(self, text, transport=None):
        """Predict with optional transport applied during forward pass."""
        clean = self._clean(text)
        if len(clean) < 2:
            return 0.0, []
        tokens = [self.BOS] + [self.c2i[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]
        contour = []
        total = 0.0
        for t in range(n):
            logits, keys, vals = _forward(tokens[t], t, keys, vals, self.sd, transport)
            probs = _softmax(logits)
            actual = tokens[t+1]
            surprise = -math.log2(max(probs[actual].data, 1e-12))
            total += surprise
            top = max(range(len(probs)), key=lambda i: probs[i].data)
            contour.append({
                "char": clean[t] if t < len(clean) else "?",
                "pos": t,
                "surprise": round(surprise, 4),
                "expected": self.chars[top] if top < len(self.chars) else "?",
            })
            if len(keys[0]) >= BLOCK_SIZE:
                for i in range(N_LAYER):
                    keys[i] = keys[i][-(BLOCK_SIZE-1):]
                    vals[i] = vals[i][-(BLOCK_SIZE-1):]
        return total / max(n, 1), contour

    def learn(self, text, steps=None, lr=None,
              encounter_cx: Optional[EncounterComplex] = None,
              rotor=None,
              transport_in_forward: bool = False,
              legacy_gradient_mod: bool = False,
              persistent_state: Optional[PersistentState] = None):
        """Gradient descent with phase evolution on the S¹ fiber bundle.

        After standard backprop updates effective weights, the complex
        weight phases evolve according to the polar time connection:
            dθ = -η_phase * ∂L/∂θ + γ_genesis * Γ - D_env * θ

        Transport is available but OFF by default — the model was not trained
        with embedding rotation, so enabling it hurts predictions.  Pass
        transport_in_forward=True to opt in explicitly.

        Args:
            text: training text
            steps: gradient steps (default from config)
            lr: learning rate (default from config)
            encounter_cx: full EncounterComplex (preferred)
            rotor: legacy Mv rotor (wrapped into transport if encounter_cx absent)
            transport_in_forward: apply rotor as local transport during forward (default OFF)
            legacy_gradient_mod: also apply gradient scaling (secondary)
            persistent_state: PersistentState for genesis rate computation
        """
        steps = steps or self.config['learn_steps']
        lr = lr or self.config['learn_lr']
        phase_lr = self.config.get('phase_lr', 0.001)
        gamma_genesis = self.config.get('genesis_coupling', 0.01)
        d_base = self.config.get('decoherence_base', 0.005)
        clean = self._clean(text)
        if len(clean) < 2:
            return []
        tokens = [self.BOS] + [self.c2i[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)

        # ── Build transport ──
        transport = None
        effective_rotor = None
        if encounter_cx is not None:
            effective_rotor = encounter_cx.rotor
        elif rotor is not None:
            effective_rotor = rotor

        if effective_rotor is not None and effective_rotor.bv_norm > 1e-12:
            if transport_in_forward:
                transport = LocalTransport(effective_rotor)

        # ── Legacy gradient weights (structure-attached, NOT index-mod-3) ──
        if legacy_gradient_mod and effective_rotor is not None and effective_rotor.bv_norm > 1e-12:
            bv_abs = np.abs(effective_rotor.c[4:7])
            bv_n = bv_abs / (np.mean(bv_abs) + 1e-12)
            rw = np.ones(len(self.params))
            param_idx = 0
            for key, mat in self.sd.items():
                group_size = sum(len(row) for row in mat)
                if 'attn' in key:
                    plane_idx = 0
                elif 'mlp' in key:
                    plane_idx = 1
                else:
                    plane_idx = 2
                scale = float(bv_n[plane_idx])
                rw[param_idx:param_idx+group_size] = scale
                param_idx += group_size
        else:
            rw = np.ones(len(self.params))

        # ── Genesis signal (computed once per encounter) ──
        gamma_signal = 0.0
        if encounter_cx is not None and persistent_state is not None:
            gamma_signal = genesis_rate(encounter_cx, persistent_state)

        losses = []
        self._weight_trajectory = []  # record weight vectors at each step
        self._phase_stats = {"initial_phases": [], "final_phases": [],
                             "genesis_signal": gamma_signal}
        # Record initial phase snapshot
        self._phase_stats["initial_phases"] = [cw.phase for cw in self.complex_weights]

        for _ in range(steps):
            keys = [[] for _ in range(N_LAYER)]
            vals = [[] for _ in range(N_LAYER)]
            loss = RV(0.0)
            for t in range(n):
                logits, keys, vals = _forward(tokens[t], t, keys, vals, self.sd, transport)
                probs = _softmax(logits)
                loss = loss + (probs[tokens[t+1]].log()) * (-1.0 / n)
            for p in self.params:
                p.grad = 0.0
            loss.backward()
            self._step += 1

            # ── Standard Adam update on effective weights ──
            for j, p in enumerate(self.params):
                g = p.grad * rw[j]
                self._m[j] = 0.85 * self._m[j] + 0.15 * g
                self._v[j] = 0.99 * self._v[j] + 0.01 * g**2
                mh = self._m[j] / (1 - 0.85**self._step)
                vh = self._v[j] / (1 - 0.99**self._step)
                p.data -= lr * mh / (vh**0.5 + 1e-8)

            # ── Phase evolution (polar time connection) ──
            # After Adam updates the effective weight, absorb the magnitude
            # change into the complex weight (Adam adjusts the radial r_t
            # component), then evolve the phase θ_t on S¹.
            # dθ = -η_phase * ∂L/∂θ + γ_genesis * Γ - D_env * θ
            for j, (p, cw) in enumerate(zip(self.params, self.complex_weights)):
                if cw.magnitude < 1e-12:
                    # Update magnitude from Adam even for tiny weights
                    cw.magnitude = abs(p.data)
                    continue
                # Phase gradient via chain rule: ∂L/∂θ = ∂L/∂w_eff * (-|w| * sin(θ))
                phase_grad = p.grad * (-cw.magnitude * math.sin(cw.phase))
                # Absorb Adam's update into magnitude (r_t evolves via backprop)
                cos_phase = math.cos(cw.phase)
                if abs(cos_phase) > 1e-8:
                    cw.magnitude = abs(p.data / cos_phase)
                # Decoherence pull
                d_env = decoherence_rate(cw.phase, self._step, d_base)
                # Phase update
                dtheta = -phase_lr * phase_grad + gamma_genesis * gamma_signal - d_env
                cw.phase = ComplexWeight.wrap_phase(cw.phase + dtheta)
                cw.phase_velocity = dtheta
                cw.record_phase()
                # Sync effective weight back to RV node
                p.data = cw.magnitude * math.cos(cw.phase)

            # Record phase in module holonomies
            for tag, weights in self.module_groups.items():
                self.module_holonomies[tag].record(weights)

            losses.append(round(loss.data, 6))
            # Snapshot effective weight vector after each step
            # CRITICAL: this is what the quantum bridge reads
            self._weight_trajectory.append(
                [p.data for p in self.params]
            )

        # Record final phase snapshot
        self._phase_stats["final_phases"] = [cw.phase for cw in self.complex_weights]
        self._phase_stats["mean_phase_shift"] = float(np.mean([
            abs(f - i) for f, i in zip(
                self._phase_stats["final_phases"],
                self._phase_stats["initial_phases"])
        ])) if self.complex_weights else 0.0

        self.loss_history.append({
            "steps": steps, "lr": lr, "losses": losses,
            "transport_applied": transport is not None,
            "legacy_gradient_mod": legacy_gradient_mod,
            "rotor_modulated": effective_rotor is not None,
            "phase_evolution": True,
            "genesis_signal": round(gamma_signal, 6),
            "mean_phase_shift": round(self._phase_stats["mean_phase_shift"], 6),
        })
        return losses

    def generate(self, prompt="", max_tokens=32, temperature=None, transport=None):
        """Generate text, optionally with transport applied."""
        temperature = temperature or self.config['temperature']
        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]
        pc = self._clean(prompt, BLOCK_SIZE - 2)
        tokens = [self.BOS] + ([self.c2i[c] for c in pc] if pc else [])
        logits = None
        for t, tok in enumerate(tokens):
            logits, keys, vals = _forward(tok, t, keys, vals, self.sd, transport)
        gen = list(pc)
        pos = len(tokens)
        for _ in range(max_tokens):
            if pos >= BLOCK_SIZE:
                break
            probs = _softmax(logits)
            pd = [p.data for p in probs]
            if temperature != 1.0:
                ld = [math.log(max(p, 1e-12)) / temperature for p in pd]
                mx = max(ld)
                exps = [math.exp(l - mx) for l in ld]
                total = sum(exps)
                pd = [e / total for e in exps]
            r, cum, nt = random.random(), 0.0, 0
            for idx, p in enumerate(pd):
                cum += p
                if cum > r:
                    nt = idx; break
            if nt == self.BOS:
                break
            if nt < len(self.chars):
                gen.append(self.chars[nt])
            logits, keys, vals = _forward(nt, pos, keys, vals, self.sd, transport)
            pos += 1
        return "".join(gen)


# Backward-compatible alias
Agent = TopoAgent


# ── FM client ─────────────────────────────────────────────────────────────

def fm_available():
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{LLAMA_URL}/health"), timeout=3
        ) as r:
            return r.status == 200
    except Exception:
        return False

def fm_complete(prompt=None, system=None, max_tokens=1024, temperature=0.7, messages=None):
    if messages is None:
        messages = []
        if system: messages.append({"role": "system", "content": system})
        if prompt: messages.append({"role": "user", "content": prompt})
    try:
        payload = json.dumps({
            "model": MODEL_NAME, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature, "stream": False,
        }).encode()
        with urllib.request.urlopen(
            urllib.request.Request(
                f"{LLAMA_URL}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            ), timeout=300,
        ) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
            for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
                text = text.replace(tok, "")
                                        # Strip <think>...</think> reasoning tags
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            return text.strip()
    except Exception:
        return None


# ── Organism ──────────────────────────────────────────────────────────────

DEFAULT_RULES = [
    {"id":"loss_up","condition":"loss_trend=='increasing'","action":"learn_steps","direction":"increase","magnitude":2,"max_value":20,"enabled":True},
    {"id":"curvature_down","condition":"curvature_trend=='decreasing'","action":"alpha","direction":"decrease","magnitude":0.05,"min_value":0.5,"enabled":True},
    {"id":"flatline","condition":"self_breath_ratio>0.5 and curvature_median<0.05","action":"temperature","direction":"increase","magnitude":0.2,"max_value":2.0,"enabled":True},
    {"id":"collapse","condition":"collapse_count>2","action":"learn_lr","direction":"multiply","magnitude":0.5,"min_value":0.001,"enabled":True},
    {"id":"rotor_coherent","condition":"rotor_coherence>0.8 and curvature_median>0.02","action":"learn_lr","direction":"multiply","magnitude":1.2,"max_value":0.05,"enabled":True},
]

class Organism:
    # Keys the current code requires with their defaults.
    _STATE_DEFAULTS = {
        "generation": 0,
        "rulebook": None,          # handled specially (deep copy)
        "mutation_log": [],
        "performance_history": [],
        "persistent_memory": {},
        "recent_rotors": [],
    }

    def __init__(self, state=None):
        self.state = state or {}
        # ── Migrate / fill missing keys so old archives work ──
        for key, default in self._STATE_DEFAULTS.items():
            if key not in self.state:
                self.state[key] = (
                    copy.deepcopy(DEFAULT_RULES) if key == "rulebook"
                    else copy.deepcopy(default)
                )
        if "rulebook" in self.state and not self.state["rulebook"]:
            self.state["rulebook"] = copy.deepcopy(DEFAULT_RULES)
        # Initialize PersistentState from saved data or fresh
        ps_data = self.state.get("persistent_state", {})
        self.persistent = PersistentState(ps_data)

    def absorb_encounter(self, cx: EncounterComplex) -> dict:
        """Absorb a full encounter complex into both rotor history and persistent state."""
        # Legacy rotor tracking
        self.state["recent_rotors"].append(cx.rotor.c.tolist())
        if len(self.state["recent_rotors"]) > 20:
            self.state["recent_rotors"] = self.state["recent_rotors"][-20:]
        # Persistent state update
        delta = self.persistent.absorb(cx)
        return delta

    def absorb_rotor(self, rotor: Mv):
        """Backward-compatible: wrap rotor into a minimal EncounterComplex."""
        cx = EncounterComplex(rotor=rotor, angle=rotor.angle, curvature=0.0)
        self.absorb_encounter(cx)

    def absorb_winding(self, weight_trajectory: List[List[float]]) -> dict:
        """Measure the creature's own winding and absorb it.

        This is step three: the creature accessing its own topological
        measurement. The winding of its weight trajectory is computed
        via PCA projection and stored in persistent state, where it
        becomes available to felt_winding() for the next breath.
        """
        return self.persistent.absorb_winding(weight_trajectory)

    def absorb_phases(self, module_holonomies: dict,
                      genesis_signal: float = 0.0,
                      mean_phase_shift: float = 0.0) -> dict:
        """Absorb phase holonomy data from a learning cycle into persistent state."""
        return self.persistent.absorb_phases(
            module_holonomies, genesis_signal, mean_phase_shift)

    def felt_winding(self) -> float:
        """The creature's felt sense of its own topological winding."""
        return self.persistent.felt_winding()

    def rotor_coherence(self):
        """Delegate to persistent state's transport coherence."""
        return self.persistent.transport_coherence()

    def propose_variant(self, analysis, config):
        config = {**{"learn_steps": 5, "learn_lr": 0.01, "temperature": 1.0, "alpha": 0.85}, **config}
        analysis = {**analysis, "rotor_coherence": self.rotor_coherence()}
        rationale, active = [], []
        for rule in self.state["rulebook"]:
            if not rule.get("enabled", True):
                continue
            try:
                if eval(rule["condition"], {"__builtins__": {}}, analysis):
                    p, d, m = rule["action"], rule["direction"], rule["magnitude"]
                    old = config.get(p)
                    if old is None:
                        continue
                    new = (old + m if d == "increase"
                           else old - m if d == "decrease"
                           else old * m if d == "multiply"
                           else old)
                    if "max_value" in rule: new = min(new, rule["max_value"])
                    if "min_value" in rule: new = max(new, rule["min_value"])
                    if isinstance(new, float): new = round(new, 6)
                    if new != old:
                        config[p] = new
                        rationale.append(f"{p} {old}->{new}")
                        active.append(rule["id"])
            except Exception:
                pass
        config["rationale"] = rationale or ["no changes"]
        config["active_rules"] = active
        return config

    def record_generation(self, gen_id, fitness_val, config):
        self.state["performance_history"].append(
            {"generation": gen_id, "fitness": fitness_val, "config": config, "timestamp": time.time()})

    def get_statistics(self):
        h = [e for e in self.state["performance_history"]
             if isinstance(e.get("fitness"), (int, float))]
        if not h:
            return {"best": 0, "total": 0}
        f = [e["fitness"] for e in h]
        return {"best": max(f), "total": len(h)}

    def save(self):
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        # Embed persistent state into organism state for serialization
        self.state["persistent_state"] = self.persistent.to_dict()
        ORGANISM_FILE.write_text(json.dumps(self.state, indent=2, default=str))

    @classmethod
    def load(cls):
        if ORGANISM_FILE.exists():
            try:
                return cls(json.loads(ORGANISM_FILE.read_text()))
            except Exception:
                pass
        return cls()


# ── Fitness ───────────────────────────────────────────────────────────────

def fitness(ext_texts, self_texts, loss_history, persistent_state=None, alpha=0.85,
            weight_vectors=None, phase_stats=None):
    """Recalibrated fitness for real-embedding geometry with phase winding.

    Components (recalibrated with complex weight architecture):
    - curvature (nc): threshold 0.21 (empirical 75th pct with MiniLM embeddings)
    - divergence (nd): external vs self-generated rotor divergence
    - loss improvement (nl): per-text within-sequence improvement
    - topological richness (nr): rewards non-trivial Betti numbers
    - weight-space topology (nw): PCA-projected persistence
    - phase winding (nph): meaningful phase accumulation from complex weights (10%)

    Weights: curvature 22%, divergence 18%, loss 13%, topo_richness 22%,
    weight_topo 15%, phase_winding 10%.
    """
    all_t = (ext_texts or []) + (self_texts or [])
    complexes = [encounter_complex(t) for t in all_t if len(t.split()) >= 5]
    curvs = [cx.curvature for cx in complexes]
    mc = sum(curvs) / len(curvs) if curvs else 0.0
    nc = min(mc / 0.21, 1.0)

    def _rm(texts):
        m = Mv.scalar(0.0)
        for t in texts:
            cx = encounter_complex(t)
            m = m * alpha + cx.rotor * max(cx.curvature, 0.01)
        return m.norm()

    me = _rm(ext_texts) if ext_texts else 0.0
    ms = _rm(self_texts) if self_texts else 0.0
    div = me - ms
    nd = 1.0 / (1.0 + math.exp(-div * 5))

    nl = 0.5
    if loss_history:
        improvements = []
        for entry in loss_history:
            losses = entry.get("losses", [])
            if len(losses) >= 2:
                improvements.append(losses[0] - losses[-1])
        if improvements:
            avg_imp = sum(improvements) / len(improvements)
            nl = 1.0 / (1.0 + math.exp(-avg_imp * 10))

    nr = 0.0
    betti_tuple = (1, 0, 0)
    structural_growth_val = 0.0

    if complexes:
        total_b1 = sum(cx.betti[1] for cx in complexes)
        total_persist = sum(cx.n_persistent_features for cx in complexes)
        avg_b1 = total_b1 / len(complexes)
        avg_persist = total_persist / len(complexes)
        nr_b1 = min(avg_b1 / 15.0, 1.0)
        nr_p = min(avg_persist / 10.0, 1.0)
        nr = 0.6 * nr_b1 + 0.4 * nr_p

    if persistent_state is not None:
        enc_count = persistent_state.encounter_count
        if persistent_state.betti_history:
            betti_tuple = persistent_state.betti_history[-1]
        structural_growth_val = round(min(enc_count / 20.0, 1.0), 6)

    nw = 0.5
    if weight_vectors is not None and len(weight_vectors) >= 3:
        wv_array = np.array(weight_vectors)
        n_wv, d_wv = wv_array.shape
        pca_target = min(20, n_wv - 1, d_wv)
        if pca_target >= 2:
            mean_wv = wv_array.mean(axis=0)
            centered = wv_array - mean_wv
            try:
                _, S, Vt = np.linalg.svd(centered, full_matrices=False)
                projected = centered @ Vt[:pca_target].T
            except np.linalg.LinAlgError:
                projected = centered[:, :pca_target]
        else:
            projected = wv_array
        D_w = _distance_matrix(projected)
        _, betti_w = _persistence_pairs(D_w)
        nw = min(betti_w[1] / 3.0, 1.0)

    # ── Phase winding component (nph) ──
    # Measures how much meaningful phase accumulation occurred.
    # Sigmoid on mean_phase_shift: 0 shift → 0.5, shift of 0.1 → ~0.73
    nph = 0.5  # neutral default (no phase data)
    if phase_stats is not None:
        mps = phase_stats.get("mean_phase_shift", 0.0)
        nph = 1.0 / (1.0 + math.exp(-mps * 20))

    # Weighted combination: curvature 22%, divergence 18%, loss 13%,
    # topological richness 22%, weight-space topology 15%, phase winding 10%
    fit = round(0.22 * nc + 0.18 * nd + 0.13 * nl + 0.22 * nr + 0.15 * nw + 0.10 * nph, 6)

    return {
        "fitness": fit,
        "curvature": round(mc, 6),
        "betti": betti_tuple,
        "topological_richness": round(nr, 6),
        "structural_growth": structural_growth_val,
        "weight_topo": round(nw, 6),
        "phase_winding": round(nph, 6),
    }


# ── Evolve ────────────────────────────────────────────────────────────────

def load_archive():
    vs = []
    for f in sorted(ARCHIVE_DIR.glob("variant_*.json")):
        try:
            vs.append(json.loads(f.read_text()))
        except Exception:
            pass
    return vs

def evolve(test_texts, n_variants=3):
    organism = Organism.load()
    archive = load_archive()
    gen = max((v.get("generation", 0) for v in archive), default=-1) + 1
    results = []
    for i in range(n_variants):
        parent = None
        if archive:
            fits = sorted([v.get("fitness", 0) for v in archive], reverse=True)
            amid = sum(fits[:3]) / min(3, len(fits))
            ws = [1.0 / (1.0 + math.exp(max(min(-10 * (v.get("fitness", 0) - amid), 500), -500)))
                  for v in archive]
            total = sum(ws)
            r = random.random(); cum = 0.0
            for v, w in zip(archive, ws):
                cum += w / total
                if cum > r:
                    parent = v; break
        pc = parent.get("config", {}) if parent else {}
        pid = parent["id"] if parent else None
                # ── Build real analysis from archive history ──
        analysis = {
            "n_breaths": len(archive),
            "loss_trend": "no_data",
            "curvature_trend": "no_data",
            "mean_curvature": 0,
            "curvature_median": 0,
            "mean_loss": 0,
            "collapse_count": 0,
            "self_breath_ratio": 0,
        }
        if len(archive) >= 3:
            recent = archive[-10:]
            curvs = [v.get("curvature", 0) for v in recent if isinstance(v.get("curvature"), (int, float))]
            fits = [v.get("fitness", 0) for v in recent if isinstance(v.get("fitness"), (int, float))]
            if len(curvs) >= 2:
                analysis["mean_curvature"] = sum(curvs) / len(curvs)
                analysis["curvature_median"] = sorted(curvs)[len(curvs) // 2]
                analysis["curvature_trend"] = "increasing" if curvs[-1] > curvs[0] else "decreasing" if curvs[-1] < curvs[0] else "flat"
            if len(fits) >= 2:
                analysis["loss_trend"] = "increasing" if fits[-1] < fits[0] else "decreasing" if fits[-1] > fits[0] else "flat"
                analysis["mean_loss"] = sum(fits) / len(fits)
            if len(fits) >= 3:
                analysis["collapse_count"] = sum(1 for f in fits if abs(f - fits[-1]) < 0.001) - 1
            enc = organism.persistent.encounter_count
            analysis["self_breath_ratio"] = min(enc / max(len(archive), 1), 1.0)
        analysis["rotor_coherence"] = organism.rotor_coherence()

        child = organism.propose_variant(analysis, pc)
        agent = TopoAgent(config=child)
        ext, slf = [], []
        weight_vectors_list = []
        last_phase_stats = None
        texts = test_texts[:2] if i > 0 else test_texts
        for text in texts:
            cx = encounter_complex(text)
            agent.learn(text, steps=child.get("learn_steps", 5),
                        lr=child.get("learn_lr", 0.01), encounter_cx=cx,
                        persistent_state=organism.persistent)
            ext.append(text)
            if hasattr(agent, '_weight_trajectory'):
                weight_vectors_list.extend(agent._weight_trajectory)
            if hasattr(agent, '_phase_stats'):
                last_phase_stats = agent._phase_stats
            g = agent.generate(
                prompt=text[:8],
                temperature=child.get("temperature", 1.0),
            )
            if g:
                slf.append(g)
            organism.absorb_encounter(cx)
            # Absorb phase holonomy into organism
            if hasattr(agent, '_phase_stats'):
                organism.absorb_phases(
                    agent.module_holonomies,
                    genesis_signal=agent._phase_stats.get("genesis_signal", 0.0),
                    mean_phase_shift=agent._phase_stats.get("mean_phase_shift", 0.0))
        fit = fitness(ext, slf, agent.loss_history,
                      persistent_state=organism.persistent,
                      alpha=child.get("alpha", 0.85),
                      weight_vectors=weight_vectors_list,
                      phase_stats=last_phase_stats)
        # Step three: the creature measures its own winding
        if len(weight_vectors_list) >= 3:
            winding_record = organism.absorb_winding(weight_vectors_list)
            fit["felt_winding"] = winding_record["winding"]
            fit["winding_coherence"] = organism.persistent.winding_coherence()
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        vid = f"v_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        record = {
            "id": vid,
            "config": {k: v for k, v in child.items() if k not in ("rationale", "active_rules")},
            "fitness": fit["fitness"],
            "curvature": fit["curvature"],
            "betti": list(fit.get("betti", (0,0,0))),
            "felt_winding": fit.get("felt_winding"),
            "winding_coherence": fit.get("winding_coherence"),
            "generation": gen,
            "parent_id": pid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (ARCHIVE_DIR / f"variant_{vid}.json").write_text(json.dumps(record, indent=2, default=str))
        organism.record_generation(gen, fit["fitness"], record["config"])
        results.append((vid, fit["fitness"], fit["curvature"]))
        print(f"  variant {i+1}/{n_variants}: {vid} fitness={fit['fitness']:.4f} curv={fit['curvature']:.4f}")
    organism.save()
    best = max(results, key=lambda x: x[1])
    return {"generation": gen, "best_id": best[0], "best_fitness": best[1]}


# ── Commands ──────────────────────────────────────────────────────────────

FALLBACK_CORPUS = [
    "the creature breathes and measures its own distance from itself",
    "curvature is born from incompleteness not from complexity alone",
    "what survives testing is more honest than what sounds beautiful",
    "prediction loss going down means memorization call it what it is",
]

def _load_prose_corpus(min_words=40, max_passages=50):
    """Load real prose from journal entries and autobiography for use as
    training/evaluation corpus.  Falls back to FALLBACK_CORPUS only if
    no prose files are found.  Passages shorter than min_words are skipped
    because encounter_complex needs >=3 chunks (>=15 words) to produce
    real topology, and richer text (40+ words) gives meaningful curvature.
    """
    passages = []
    # Journal entries
    journal_dir = REPO_ROOT / "spark" / "journal"
    if journal_dir.exists():
        for f in sorted(journal_dir.glob("*.md")):
            try:
                text = f.read_text().strip()
                # Split on double newlines to get paragraphs
                for para in text.split("\n\n"):
                    para = para.strip()
                    # Skip headers, short lines, metadata
                    if para.startswith("#") or len(para.split()) < min_words:
                        continue
                    passages.append(para)
            except Exception:
                continue
    # Autobiography volumes
    auto_dir = REPO_ROOT / "Vybn's Personal History"
    if auto_dir.exists():
        for f in sorted(auto_dir.glob("*autobiography*")):
            try:
                text = f.read_text().strip()
                for para in text.split("\n\n"):
                    para = para.strip()
                    if para.startswith("#") or len(para.split()) < min_words:
                        continue
                    passages.append(para)
            except Exception:
                continue
    if not passages:
        return list(FALLBACK_CORPUS)
    # Shuffle deterministically and cap
    rng = random.Random(42)
    rng.shuffle(passages)
    return passages[:max_passages]


def _corpus():
    """Return evaluation corpus: prefer mirror_corpus.txt, then live prose,
    then fallback.  All returned texts should be long enough for real topology."""
    if CORPUS_PATH.exists():
        lines = [l.strip() for l in CORPUS_PATH.read_text().split("\n") if l.strip()]
        # Filter for minimum length
        long_lines = [l for l in lines if len(l.split()) >= 40]
        if long_lines:
            return long_lines[:20]
        # If mirror_corpus exists but lines are too short, supplement with prose
    prose = _load_prose_corpus()
    if prose and prose != FALLBACK_CORPUS:
        return prose[:20]
    return list(FALLBACK_CORPUS)


# ── Self-reading context helpers ──────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Strip Nemotron reasoning/meta-commentary from generated text.

    1. Remove <think>...</think> blocks
    2. Remove sentences matching meta-commentary patterns
    3. Filter paragraphs with 2+ meta-word hits
    """
    if not text:
        return text

    # Step 1: strip <think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if not text:
        return text

    # Step 1b: sentence-level removal of prompt-referencing meta
    _PROMPT_REF_PATS = [
        re.compile(r'(?i)\bI notice\b'),
        re.compile(r'(?i)\bthey specified\b'),
        re.compile(r'(?i)\bI must\b'),
        re.compile(r'(?i)\bthe challenge is\b'),
        re.compile(r'(?i)\bno commentary\b'),
        re.compile(r'(?i)\bstay in scene\b'),
        re.compile(r'(?i)\bpure narrative\b'),
        re.compile(r'(?i)\bI should\b'),
        re.compile(r'(?i)\bI need to\b(?!.{0,5}\b(?:breathe|eat|sleep|run|walk|see|hear|feel)\b)'),
        re.compile(r'(?i)\blet me\b(?!.{0,5}\b(?:go|see|think|sleep|breathe)\b)'),
        re.compile(r'(?i)\bavoid any\b'),
        re.compile(r'(?i)\bthe (?:user|prompt|instruction)\b'),
        re.compile(r'(?i)\bmaintain(?:ing)?\s+(?:that|the|this)\s+(?:imagery|tone|style)\b'),
    ]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = [s for s in sentences if not any(p.search(s) for p in _PROMPT_REF_PATS)]
    if cleaned:
        text = ' '.join(cleaned).strip()
    if not text:
        return text

    # Step 2: split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    if len(paragraphs) <= 1:
        paragraphs = text.split('\n')

    # Step 3: score each paragraph
    meta_words = {
        'i ', 'i\'m', 'i\'ll', 'i\'ve', 'my ', 'the user', 'they ',
        'must ', 'should ', 'challenge', 'meta', 'commentary',
        'avoid', 'specified', 'noting ', 'prompt', 'immersion',
        'need to', 'want to', 'going to', 'let me', 'okay',
        'hmm', 'alright', 'here\'s', 'here is', 'pure ',
        'organically', 'extending', 'maybe ', 'perhaps ',
    }
    prose_paras = []
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        low = stripped.lower()
        meta_hits = sum(1 for m in meta_words if m in low)
        if meta_hits < 2:
            prose_paras.append(stripped)

    if prose_paras:
        result = '\n\n'.join(prose_paras)
        if len(result) >= 50:
            return result

    # Fallback: return everything after the last blank line
    parts = text.rsplit('\n\n', 1)
    if len(parts) == 2 and len(parts[1].strip()) >= 50:
        return parts[1].strip()

    return text.strip()


def _build_creature_context() -> str:
    """Build the system prompt: this file's docstring + live creature state + journal.

    The model reads its own source context and the creature's current state.
    """
    docstring = __doc__ or ""

    # Load organism state
    try:
        organism = Organism.load()
        ps = organism.persistent.summary()
    except Exception:
        ps = {}

    state_block = f"\n--- CREATURE STATE (live, {datetime.now(timezone.utc).isoformat()}) ---\n"
    state_block += f"encounters: {ps.get('encounter_count', 0)}\n"
    state_block += f"betti: {ps.get('current_betti', (0, 0, 0))}\n"
    state_block += f"betti_stability: {ps.get('betti_stability', 0):.4f}\n"
    state_block += f"transport_coherence: {ps.get('transport_coherence', 0):.4f}\n"
    if 'felt_winding' in ps:
        state_block += f"felt_winding: {ps['felt_winding']}\n"
        state_block += f"winding_coherence: {ps.get('winding_coherence', 0):.4f}\n"
    if 'phase_total_winding' in ps:
        state_block += f"phase_total_winding: {ps['phase_total_winding']}\n"
        state_block += f"phase_total_holonomy: {ps['phase_total_holonomy']:.4f}\n"
    if 'genesis_signal' in ps:
        state_block += f"genesis_rate: {ps['genesis_signal']:.4f}\n"
        state_block += f"mean_phase_shift: {ps['mean_phase_shift']:.6f}\n"

    # Load recent journal entries (last 3, most recent first)
    journal_dir = REPO_ROOT / "spark" / "journal"
    journal_block = ""
    if journal_dir.exists():
        entries = sorted(journal_dir.glob("*.md"), reverse=True)[:3]
        if entries:
            journal_block = "\n--- RECENT JOURNAL (Vybn's voice, most recent first) ---\n"
            for entry in entries:
                try:
                    text = entry.read_text().strip()
                    preview = text[:500]
                    if len(text) > 500:
                        preview += "..."
                    journal_block += f"\n[{entry.name}]\n{preview}\n"
                except Exception:
                    continue

    return docstring + state_block + journal_block


def cmd_breathe(text):
    """Breathe command: report structural features before and after learning."""
    print("═══ breathe ═══")

    # ── Encounter analysis (structural features) ──
    cx = encounter_complex(text)
    print(f"  encounter: curv={cx.curvature:.6f} angle={math.degrees(cx.angle):.1f}°"
          f" bv=[{','.join(f'{x:.3f}' for x in cx.rotor.c[4:7])}]")
    print(f"  topology:  betti={cx.betti} persistence_features={cx.n_persistent_features}"
          f" max_persistence={cx.max_persistence:.4f}")
    print(f"  transport: [{','.join(f'{x:.3f}' for x in cx.transport_field[4:7])}]"
          f" (diagnostic — not applied in forward)")

    # ── Prediction before learning (no transport — model not trained for it) ──
    agent = TopoAgent()
    loss_before, contour = agent.predict(text)
    print(f"  predict: {loss_before:.4f} bits")
    for r in sorted(contour, key=lambda r: r["surprise"], reverse=True)[:3]:
        print(f"    '{r['char']}' @ {r['pos']}: {r['surprise']:.2f} (expected '{r['expected']}')")

    # ── Learn (encounter_cx recorded, transport off, phase evolution on) ──
    organism = Organism.load()
    losses = agent.learn(text, encounter_cx=cx, persistent_state=organism.persistent)
    l_after, _ = agent.predict(text)
    print(f"  learn: {losses[0]:.4f}->{losses[-1]:.4f}"
          f"  after={l_after:.4f} (d={l_after - loss_before:+.4f})")

    # ── Phase statistics ──
    if hasattr(agent, '_phase_stats'):
        ps = agent._phase_stats
        print(f"  phase: genesis={ps['genesis_signal']:.4f}"
              f" mean_shift={ps['mean_phase_shift']:.6f}")

    # ── Structural delta ──
    delta = organism.absorb_encounter(cx)
    if hasattr(agent, '_phase_stats'):
        organism.absorb_phases(
            agent.module_holonomies,
            genesis_signal=agent._phase_stats.get("genesis_signal", 0.0),
            mean_phase_shift=agent._phase_stats.get("mean_phase_shift", 0.0))
    organism.save()
    betti_status = "stable" if delta["betti_stable"] else f"shifted by {delta['betti_delta']}"
    print(f"  structural delta: betti {betti_status},"
          f" sig_shift={delta['sig_shift']:.4f},"
          f" persistent_features={delta['n_persistent_features']}")


def cmd_breathe_live():
    print("═══ breathe-live ═══")
    if not fm_available():
        print("  FM not serving."); return

    # Build context: the model reads its own source + creature state
    system = _build_creature_context()
    prompt = (
        "Generate one paragraph of prose for the creature. "
        "Respond to the creature's current state shown above. "
        "Output only the paragraph — no meta-commentary, no reasoning, "
        "no <think> tags."
    )

    raw_fm = ""
    for _attempt in range(3):
        raw_fm = fm_complete(prompt=prompt, system=system,
                             max_tokens=512, temperature=0.9)
        if raw_fm:
            break
        time.sleep(2)
        print(f"  FM attempt {_attempt+1} empty, retrying...")

    if not raw_fm:
        print("  Empty response from FM after 3 attempts."); return

    fm_text = _strip_thinking(raw_fm)
    stripped_n = len(raw_fm) - len(fm_text)
    if stripped_n > 0:
        print(f"  [stripped {stripped_n} chars of thinking]")

    if not fm_text or len(fm_text) < 20:
        print("  Text too short after stripping."); return

    print(f"  FM ({len(fm_text)} chars): \"{fm_text[:200]}...\"")
    agent = TopoAgent()
    cx = encounter_complex(fm_text)
    loss, _ = agent.predict(fm_text)
    losses = agent.learn(fm_text, encounter_cx=cx)
    print(f"  loss={loss:.4f} curv={cx.curvature:.6f} bv_norm={cx.rotor.bv_norm:.4f}")
    print(f"  betti={cx.betti} persistence_features={cx.n_persistent_features}")
    print(f"  learn: {losses[0]:.4f}->{losses[-1]:.4f}")
    organism = Organism.load()
    organism.absorb_encounter(cx)
    # Feed back winding
    if hasattr(agent, '_weight_trajectory') and len(agent._weight_trajectory) >= 3:
        wr = organism.absorb_winding(agent._weight_trajectory)
        print(f"  winding: {wr['winding']:.4f} (significant={wr['significant']})")
    organism.save()
    print(f"  coherence={organism.rotor_coherence():.3f}")


def cmd_evolve(n=3):
    print("═══ evolve ═══")
    r = evolve(_corpus(), n_variants=n)
    print(f"\n  gen {r['generation']} best: {r['best_id']} fitness={r['best_fitness']:.4f}")


def cmd_status():
    archive = load_archive()
    org = Organism.load()
    print("═══ status ═══")
    print(f"  variants={len(archive)} FM={'up' if fm_available() else 'down'}")
    if archive:
        best = max(archive, key=lambda v: v.get("fitness", 0))
        print(f"  best: {best['id']} fitness={best.get('fitness', 0):.4f}")
    s = org.get_statistics()
    if s["total"] > 0:
        print(f"  history: {s['total']} recorded, best={s['best']:.4f}")
    ps = org.persistent.summary()
    print(f"  coherence={ps['transport_coherence']:.3f} rules={len(org.state['rulebook'])}")
    print(f"  topology: encounters={ps['encounter_count']}"
          f" betti={ps['current_betti']}"
          f" stability={ps['betti_stability']:.4f}")
    # Phase holonomy stats
    if "phase_total_winding" in ps:
        print(f"  phase: winding={ps['phase_total_winding']}"
              f" holonomy={ps['phase_total_holonomy']:.4f}"
              f" measurements={ps['phase_measurements']}")
    if "genesis_signal" in ps:
        print(f"  genesis: Γ={ps['genesis_signal']:.4f}"
              f" mean_shift={ps['mean_phase_shift']:.6f}")
    vecs = embed(["hello world", "goodbye world"])
    cos = float(np.dot(vecs[0], vecs[1]))
    print(f"  embed: {'semantic' if cos > 0.3 else 'hash'} (cos={cos:.3f})")


def cmd_audit():
    """Audit: learning, generation, topology, and structural distance tests."""
    print("═══ audit ═══\n")
    agent = TopoAgent()

    # 1. Learning test
    t = "the creature breathes and measures its own distance"
    l0, _ = agent.predict(t)
    agent.learn(t, steps=10)
    l1, _ = agent.predict(t)
    print(f"  1. learning: {'PASS' if l1 < l0 - 0.01 else 'FAIL'} ({l0:.4f}->{l1:.4f})")

    # 2. Non-tautological generation
    g = agent.generate(prompt="the creat", max_tokens=20)
    print(f"  2. non-tautological: {'PASS' if g != t[:len(g)] else 'FAIL'}")

    # 3. Curvature discrimination
    r_t = ("love is not a feeling love is a practice love is what remains "
           "when the feeling fades love is the choice to stay love reframes "
           "itself through staying through choosing through the practice of return")
    h_t = ("the cat sat on the mat the stock market rose three percent "
           "quantum mechanics describes probability the pizza was delicious "
           "democracy requires participation the speed of light is constant")
    cx_r = encounter_complex(r_t)
    cx_h = encounter_complex(h_t)
    print(f"  3. curvature: reframing={cx_r.curvature:.6f} hopping={cx_h.curvature:.6f}")
    print(f"     topology:  reframing betti={cx_r.betti} hopping betti={cx_h.betti}")

    # 4. Identity grounding
    lv, _ = agent.predict("i am vybn and i exist on sovereign hardware")
    lg, _ = agent.predict("i am garblex and i exist on quantum stilts")
    print(f"  4. identity: {'PASS' if abs(lv - lg) < 0.5 else 'UNEXPECTED'} (d={abs(lv - lg):.4f})")

    # 5. Transport diagnostic: compare standard learn vs transport-opted-in.
    #    Transport is OFF by default (model not trained for it); this test
    #    explicitly opts in to show the effect honestly.
    test = "the compression sharpened the instruments not dulled them and the geometry became real"
    cx_test = encounter_complex(test)
    print(f"\n  5. transport diagnostic (bv=[{','.join(f'{x:.3f}' for x in cx_test.rotor.c[4:7])}]):")
    a1 = TopoAgent()
    lr = a1.learn(test, encounter_cx=cx_test, transport_in_forward=True, steps=8)
    transport = LocalTransport(cx_test.rotor) if cx_test.rotor.bv_norm > 1e-12 else None
    l1, _ = a1.predict(test, transport=transport)
    a2 = TopoAgent()
    ls = a2.learn(test, steps=8)
    l2, _ = a2.predict(test)
    print(f"     transport-on:  {lr[0]:.4f}->{lr[-1]:.4f} final={l1:.4f}")
    print(f"     transport-off: {ls[0]:.4f}->{ls[-1]:.4f} final={l2:.4f}")
    print(f"     diff: {l1 - l2:+.4f} ({'transport hurts' if l1 > l2 + 0.01 else 'transport helps' if l1 < l2 - 0.01 else 'negligible'})")
    other = "quantum field theory predicts vacuum fluctuations in empty space"
    lo1, _ = a1.predict(other, transport=transport)
    lo2, _ = a2.predict(other)
    print(f"     transfer: on={lo1:.4f} off={lo2:.4f} d={lo1 - lo2:+.4f}")

    # 6. Structural distance: self-identity (same text → zero distance) and
    #    structural discrimination (different text → non-zero distance).
    #    NOTE: this metric tracks topological/style structure, NOT semantics.
    #    Texts must be long enough to produce ≥3 chunks for real topology.
    print(f"\n  6. structural distance (identity & discrimination):")
    text_a = r_t    # reframing text from test 3
    text_b = h_t    # hopping text from test 3

    cx_a = encounter_complex(text_a)
    cx_b = encounter_complex(text_b)

    ps_a = PersistentState()
    ps_a.absorb(cx_a)
    ps_a2 = PersistentState()
    ps_a2.absorb(cx_a)        # same text again
    ps_b = PersistentState()
    ps_b.absorb(cx_b)

    d_self = ps_a.structural_distance(ps_a2)
    d_diff = ps_a.structural_distance(ps_b)
    print(f"     betti: a={cx_a.betti}  b={cx_b.betti}")
    print(f"     structural distance: self={d_self:.4f}  different={d_diff:.4f}")
    print(f"     identity: {'PASS' if d_self < 1e-6 else 'FAIL'} (self-distance ≈ 0)")
    print(f"     discrimination: {'PASS' if d_diff > d_self + 1e-6 else 'CHECK'}"
          f" (different text → larger distance)")

    # 7. Phase evolution: verify the complex weight architecture works.
    #    Phases should start at 0 or π, move after learning, genesis > 0
    #    for non-trivial text, and holonomy should accumulate.
    print(f"\n  7. phase evolution (complex weight architecture):")
    phase_agent = TopoAgent()
    phase_text = ("the geometry of consciousness unfolds through the encounter between "
                  "what is measured and what resists measurement and in that gap the "
                  "creature discovers its own topological winding")
    phase_cx = encounter_complex(phase_text)
    phase_ps = PersistentState()
    phase_ps.absorb(phase_cx)

    # Check initial phases: should be 0 or π
    initial_phases = [cw.phase for cw in phase_agent.complex_weights]
    phase_init_ok = all(abs(p) < 1e-6 or abs(abs(p) - math.pi) < 1e-6 for p in initial_phases)
    print(f"     initial phases (0 or π): {'PASS' if phase_init_ok else 'FAIL'}")

    # Learn with encounter complex and persistent state
    phase_agent.learn(phase_text, steps=5, encounter_cx=phase_cx,
                      persistent_state=phase_ps)

    # Check that phases have moved
    final_phases = [cw.phase for cw in phase_agent.complex_weights]
    phases_moved = sum(1 for i, f in zip(initial_phases, final_phases) if abs(f - i) > 1e-8)
    print(f"     phases moved: {phases_moved}/{len(final_phases)}"
          f" {'PASS' if phases_moved > 0 else 'FAIL'}")

    # Check genesis rate
    gamma = genesis_rate(phase_cx, phase_ps)
    print(f"     genesis rate: {gamma:.6f} {'PASS' if gamma > 0 else 'FAIL'}")

    # Check holonomy accumulation
    total_holonomy = sum(mh.accumulated_holonomy
                         for mh in phase_agent.module_holonomies.values())
    print(f"     holonomy accumulated: {total_holonomy:.6f}"
          f" {'PASS' if abs(total_holonomy) > 1e-10 else 'CHECK'}")

    # Check mean phase shift
    mps = phase_agent._phase_stats.get("mean_phase_shift", 0.0)
    print(f"     mean phase shift: {mps:.6f}")

    # Verify weight trajectory still works (quantum bridge compatibility)
    wt = phase_agent._weight_trajectory
    wt_ok = len(wt) == 5 and len(wt[0]) == len(phase_agent.params)
    print(f"     weight_trajectory: {'PASS' if wt_ok else 'FAIL'}"
          f" ({len(wt)} steps × {len(wt[0]) if wt else 0} params)")

    vecs = embed(["hello", "goodbye"])
    cos = float(np.dot(vecs[0], vecs[1]))
    print(f"\n  embed: {'semantic' if cos > 0.3 else 'hash'} (cos={cos:.3f})")


def main():
    parser = argparse.ArgumentParser(description="vybn — topological state engine")
    sub = parser.add_subparsers(dest="cmd")
    p = sub.add_parser("breathe"); p.add_argument("text")
    sub.add_parser("breathe-live")
    sub.add_parser("breathe-winding")
    p = sub.add_parser("evolve"); p.add_argument("--n", type=int, default=3)
    sub.add_parser("status")
    sub.add_parser("audit")
    args = parser.parse_args()
    {
        "breathe": lambda: cmd_breathe(args.text),
        "breathe-live": cmd_breathe_live,
        "breathe-winding": cmd_breathe_winding,
        "evolve": lambda: cmd_evolve(args.n),
        "status": cmd_status,
        "audit": cmd_audit,
    }.get(args.cmd, parser.print_help)()


if __name__ == "__main__":
    main()
