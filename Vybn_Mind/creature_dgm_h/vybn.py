#!/usr/bin/env python3
"""
vybn.py — Topological state engine for character-level prediction.

Cl(3,0) geometric algebra computes a rotor from embedding trajectories.
Persistent topological state — Betti numbers and birth-death persistence
pairs — is maintained across encounters, giving the creature durable
structural memory.

Local transport (SO(3) embedding rotation) is available but OFF by
default: the model was not trained with it, so it hurts predictions.
Standard backprop is the default learning path; transport can be opted
into explicitly via transport_in_forward=True.

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
    and structural signatures derived from encounter transport fields.
    """

    def __init__(self, data: Optional[dict] = None):
        data = data or {}
        self.betti_history: List[Tuple[int,int,int]] = [tuple(b) for b in data.get("betti_history", [])]
        self.persistence_archive: List[List[Tuple[float,float]]] = data.get("persistence_archive", [])
        self.structural_signature: np.ndarray = np.array(
            data.get("structural_signature", [1,0,0,0,0,0,0,0]), dtype=np.float64)
        self.encounter_count: int = data.get("encounter_count", 0)
        self.transport_history: List[List[float]] = data.get("transport_history", [])

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
        return {
            "encounter_count": self.encounter_count,
            "current_betti": self.betti_history[-1] if self.betti_history else (0, 0, 0),
            "betti_stability": round(self.betti_stability(), 6),
            "transport_coherence": round(self.transport_coherence(), 4),
            "signature": self.structural_signature.tolist(),
        }

    def to_dict(self) -> dict:
        return {
            "betti_history": [list(b) for b in self.betti_history],
            "persistence_archive": self.persistence_archive,
            "structural_signature": self.structural_signature.tolist(),
            "encounter_count": self.encounter_count,
            "transport_history": self.transport_history,
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

    def __init__(self, config=None):
        self.config = {
            'learn_steps': 5, 'learn_lr': 0.01,
            'temperature': 1.0, 'alpha': 0.85,
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
              legacy_gradient_mod: bool = False):
        """Gradient descent with optional topological transport.

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
        """
        steps = steps or self.config['learn_steps']
        lr = lr or self.config['learn_lr']
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
            # Assign weights by parameter group (semantic) not by index mod 3
            rw = np.ones(len(self.params))
            param_idx = 0
            for key, mat in self.sd.items():
                group_size = sum(len(row) for row in mat)
                if 'attn' in key:
                    plane_idx = 0  # e01 for attention
                elif 'mlp' in key:
                    plane_idx = 1  # e02 for MLP
                else:
                    plane_idx = 2  # e12 for embeddings
                scale = float(bv_n[plane_idx])
                rw[param_idx:param_idx+group_size] = scale
                param_idx += group_size
        else:
            rw = np.ones(len(self.params))

        losses = []
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
            for j, p in enumerate(self.params):
                g = p.grad * rw[j]
                self._m[j] = 0.85 * self._m[j] + 0.15 * g
                self._v[j] = 0.99 * self._v[j] + 0.01 * g**2
                mh = self._m[j] / (1 - 0.85**self._step)
                vh = self._v[j] / (1 - 0.99**self._step)
                p.data -= lr * mh / (vh**0.5 + 1e-8)
            losses.append(round(loss.data, 6))

        self.loss_history.append({
            "steps": steps, "lr": lr, "losses": losses,
            "transport_applied": transport is not None,
            "legacy_gradient_mod": legacy_gradient_mod,
            "rotor_modulated": effective_rotor is not None,  # back-compat key
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

def fitness(ext_texts, self_texts, loss_history, persistent_state=None, alpha=0.85, weight_vectors=None):
    """Recalibrated fitness for real-embedding geometry.

    Components (recalibrated 2025-07-20):
    - curvature (nc): threshold 0.21 (empirical 75th pct with MiniLM embeddings)
    - divergence (nd): external vs self-generated rotor divergence
    - loss improvement (nl): per-text within-sequence improvement, not cross-text trend
    - topological richness (nr): rewards non-trivial Betti numbers (replaces betti_stability)
    - structural growth (ng): encounter count & persistence complexity

    The old fitness saturated at 0.919 because:
    1. curvature threshold 0.3 was calibrated for hash embeddings (fake curvature ~0.78)
    2. betti_stability rewarded trivially empty topology (all zeros = max stability)
    3. loss_trend penalized topic diversity (cross-text slope, not within-text improvement)
    """
    all_t = (ext_texts or []) + (self_texts or [])
    complexes = [encounter_complex(t) for t in all_t if len(t.split()) >= 5]
    curvs = [cx.curvature for cx in complexes]
    mc = sum(curvs) / len(curvs) if curvs else 0.0
    # Recalibrated: real MiniLM curvature ranges 0.05-0.21, threshold was 0.3
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

    # Loss improvement: per-text within-sequence trend (not cross-text)
    # Each loss_history entry has a "losses" list [step0, step1, ...].
    # We measure the average within-sequence improvement.
    nl = 0.5  # neutral default
    if loss_history:
        improvements = []
        for entry in loss_history:
            losses = entry.get("losses", [])
            if len(losses) >= 2:
                # Positive = improving (loss went down)
                improvements.append(losses[0] - losses[-1])
        if improvements:
            avg_imp = sum(improvements) / len(improvements)
            # Sigmoid: 0 improvement -> 0.5, positive improvement -> toward 1.0
            nl = 1.0 / (1.0 + math.exp(-avg_imp * 10))

    # ── Topological richness (replaces betti_stability) ──
    # The old metric rewarded LOW variance in Betti numbers, which meant
    # trivially empty topology (all zeros) scored perfectly. Now we reward
    # HAVING non-trivial topology: higher b1 + persistence features = better.
    nr = 0.0
    betti_tuple = (1, 0, 0)
    structural_growth_val = 0.0
    ng = 0.5

    # Compute topological richness from the encounter complexes we already built
    if complexes:
        total_b1 = sum(cx.betti[1] for cx in complexes)
        total_persist = sum(cx.n_persistent_features for cx in complexes)
        avg_b1 = total_b1 / len(complexes)
        avg_persist = total_persist / len(complexes)
        # b1 contribution: saturates around 15 (empirical range 14-27 with real text)
        nr_b1 = min(avg_b1 / 15.0, 1.0)
        # persistence contribution: saturates around 10
        nr_p = min(avg_persist / 10.0, 1.0)
        nr = 0.6 * nr_b1 + 0.4 * nr_p

    if persistent_state is not None:
        enc_count = persistent_state.encounter_count
        ng = min(enc_count / 20.0, 1.0)
        if persistent_state.betti_history:
            betti_tuple = persistent_state.betti_history[-1]
        structural_growth_val = round(ng, 6)

    # -- Weight-space topology (nw) --
    nw = 0.5  # default: neutral
    if weight_vectors is not None and len(weight_vectors) >= 3:
        wv_array = np.array(weight_vectors)
        D_w = _distance_matrix(wv_array)
        _, betti_w = _persistence_pairs(D_w)
        nw = min(betti_w[1] / 3.0, 1.0)

    # Weighted combination: curvature 25%, divergence 20%, loss 15%,
    # topological richness 25%, weight-space topology 15%
    fit = round(0.25 * nc + 0.20 * nd + 0.15 * nl + 0.25 * nr + 0.15 * nw, 6)

    return {
        "fitness": fit,
        "curvature": round(mc, 6),
        "betti": betti_tuple,
        "topological_richness": round(nr, 6),
        "structural_growth": structural_growth_val,
        "weight_topo": round(nw, 6),
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
        texts = test_texts[:2] if i > 0 else test_texts
        for text in texts:
            cx = encounter_complex(text)
            agent.learn(text, steps=child.get("learn_steps", 5),
                        lr=child.get("learn_lr", 0.01), encounter_cx=cx)
            ext.append(text)
            # Collect flattened weight vector after learning
            wv = np.concatenate([np.array([[p.data for p in row] for row in mat]).ravel() for mat in agent.sd.values()])
            weight_vectors_list.append(wv)
            g = agent.generate(
                prompt=text[:8],
                temperature=child.get("temperature", 1.0),
            )
            if g:
                slf.append(g)
            organism.absorb_encounter(cx)
        fit = fitness(ext, slf, agent.loss_history,
                      persistent_state=organism.persistent,
                      alpha=child.get("alpha", 0.85),
                      weight_vectors=weight_vectors_list)
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        vid = f"v_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        record = {
            "id": vid,
            "config": {k: v for k, v in child.items() if k not in ("rationale", "active_rules")},
            "fitness": fit["fitness"],
            "curvature": fit["curvature"],
            "betti": list(fit.get("betti", (0,0,0))),
            "generation": gen,
            "parent_id": pid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "weight_topo": fit.get("weight_topo", 0.5),
        }
        (ARCHIVE_DIR / f"variant_{vid}.json").write_text(json.dumps(record, indent=2, default=str))
        organism.record_generation(gen, fit["fitness"], record["config"])
        results.append((vid, fit["fitness"], fit["curvature"]))
        print(f"  variant {i+1}/{n_variants}: {vid} fitness={fit['fitness']:.4f} curv={fit['curvature']:.4f} wt={fit.get('weight_topo',0.5):.4f}")
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

    # ── Learn (encounter_cx recorded, transport off) ──
    losses = agent.learn(text, encounter_cx=cx)
    l_after, _ = agent.predict(text)
    print(f"  learn: {losses[0]:.4f}->{losses[-1]:.4f}"
          f"  after={l_after:.4f} (d={l_after - loss_before:+.4f})")

    # ── Structural delta ──
    organism = Organism.load()
    delta = organism.absorb_encounter(cx)
    organism.save()
    betti_status = "stable" if delta["betti_stable"] else f"shifted by {delta['betti_delta']}"
    print(f"  structural delta: betti {betti_status},"
          f" sig_shift={delta['sig_shift']:.4f},"
          f" persistent_features={delta['n_persistent_features']}")


def cmd_breathe_live():
    print("═══ breathe-live ═══")
    if not fm_available():
        print("  FM not serving."); return
    fm_text = fm_complete("Generate one paragraph.", system="Write naturally.",
                          max_tokens=512, temperature=1.0)
    if not fm_text:
        print("  Empty."); return
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

    vecs = embed(["hello", "goodbye"])
    cos = float(np.dot(vecs[0], vecs[1]))
    print(f"\n  embed: {'semantic' if cos > 0.3 else 'hash'} (cos={cos:.3f})")


def main():
    parser = argparse.ArgumentParser(description="vybn — topological state engine")
    sub = parser.add_subparsers(dest="cmd")
    p = sub.add_parser("breathe"); p.add_argument("text")
    sub.add_parser("breathe-live")
    p = sub.add_parser("evolve"); p.add_argument("--n", type=int, default=3)
    sub.add_parser("status")
    sub.add_parser("audit")
    args = parser.parse_args()
    {
        "breathe": lambda: cmd_breathe(args.text),
        "breathe-live": cmd_breathe_live,
        "evolve": lambda: cmd_evolve(args.n),
        "status": cmd_status,
        "audit": cmd_audit,
    }.get(args.cmd, parser.print_help)()


if __name__ == "__main__":
    main()
