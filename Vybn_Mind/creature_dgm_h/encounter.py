"""
encounter.py — Text to topology.

The representation map: text |-> D, where D is the space of
encounter complexes in Cl(3,0).

This is one half of the diagonal. The other half is the agent
producing text from an encounter. Together they form the
endomorphism f: D -> D whose fixed points and gaps are the
creature's incompleteness structure.
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from .algebra import Mv
except ImportError:
    from algebra import Mv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# -- Embedding --

def _hash_embed(texts):
    vecs = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % 2**31)
        v = rng.randn(384).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        vecs.append(v)
    return np.array(vecs)


def _make_embed_fn():
    try:
        sys.path.insert(0, str(REPO_ROOT / "spark"))
        from local_embedder import embed
        embed(["test"])
        return embed
    except Exception:
        return _hash_embed


embed = _make_embed_fn()


# -- Persistence homology (lightweight) --

def _distance_matrix(vecs):
    n = len(vecs)
    D = np.zeros((n, n), np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(vecs[i] - vecs[j]))
            D[i, j] = D[j, i] = d
    return D


def _persistence_pairs(D):
    n = len(D)
    if n == 0:
        return [], (0, 0, 0)
    edges = sorted((D[i, j], i, j) for i in range(n) for j in range(i + 1, n))
    parent = list(range(n))
    rank = [0] * n
    birth = {i: 0.0 for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    n_triangles = 0
    for dist, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            younger = rj if birth.get(rj, 0) >= birth.get(ri, 0) else ri
            pairs.append((birth[younger], dist))
            if rank[ri] < rank[rj]:
                parent[ri] = rj
            elif rank[ri] > rank[rj]:
                parent[rj] = ri
            else:
                parent[rj] = ri
                rank[ri] += 1
        else:
            n_triangles += 1

    components = set(find(i) for i in range(n))
    for c in components:
        pairs.append((birth.get(c, 0.0), float('inf')))

    # Betti at median threshold
    if edges:
        med_thresh = edges[len(edges) // 2][0]
    else:
        med_thresh = 0.0
    uf2 = list(range(n))
    def find2(x):
        while uf2[x] != x:
            uf2[x] = uf2[uf2[x]]
            x = uf2[x]
        return x
    for dist, i, j in edges:
        if dist > med_thresh:
            break
        ri, rj = find2(i), find2(j)
        if ri != rj:
            if rank[ri] < rank[rj]:
                uf2[ri] = rj
            else:
                uf2[rj] = ri
    b0 = len(set(find2(i) for i in range(n)))
    b1 = max(0, n_triangles - (n - b0))
    return pairs, (b0, b1, 0)


# -- EncounterComplex --

@dataclass
class EncounterComplex:
    """The topological signature of a text encounter.

    This is an element of D — the domain the creature maps to itself.
    """
    rotor: Mv
    angle: float
    curvature: float
    betti: Tuple[int, int, int] = (0, 0, 0)
    persistence: List[Tuple[float, float]] = field(default_factory=list)
    transport_field: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.transport_field is None:
            n = self.rotor.norm()
            self.transport_field = self.rotor.c / n if n > 1e-12 else np.zeros(8, np.float64)
            if n < 1e-12:
                self.transport_field[0] = 1.0

    @property
    def n_persistent_features(self):
        return len([p for p in self.persistence if p[1] != float('inf')])

    @property
    def max_persistence(self):
        finite = [p[1] - p[0] for p in self.persistence if p[1] != float('inf')]
        return max(finite) if finite else 0.0


def encounter_complex(text, embed_fn=None):
    """The representation map: text |-> D."""
    if embed_fn is None:
        embed_fn = embed

    words = text.split()
    cs = max(5, len(words) // 8)
    chunks = [" ".join(words[i:i + cs]) for i in range(0, len(words), cs)]
    chunks = [c for c in chunks if c.strip()]

    if len(chunks) < 3:
        return EncounterComplex(
            rotor=Mv.scalar(1.0), angle=0.0, curvature=0.0,
            betti=(1, 0, 0), persistence=[(0.0, float('inf'))],
        )

    vecs = embed_fn(chunks)

    # Pancharatnam phase
    pr, pi = 1.0, 0.0
    for i in range(len(vecs)):
        j = (i + 1) % len(vecs)
        v1, v2 = vecs[i].reshape(-1, 2), vecs[j].reshape(-1, 2)
        re = float(np.sum(v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]))
        im = float(np.sum(v1[:, 1] * v2[:, 0] - v1[:, 0] * v2[:, 1]))
        mg = math.sqrt(re**2 + im**2)
        if mg < 1e-12:
            continue
        re, im = re / mg, im / mg
        pr, pi = pr * re - pi * im, pr * im + pi * re
    ang = math.atan2(pi, pr)
    curv = abs(ang) / max(len(chunks) - 1, 1)

    # Open-path rotor chain
    mvs = [Mv.from_embedding(v) for v in vecs]
    R = Mv.scalar(1.0)
    for i in range(len(mvs) - 1):
        e = (mvs[i] * mvs[i + 1]).even()
        n = e.norm()
        if n > 1e-12:
            R = R * Mv(e.c / n)
    h = ang / 2.0
    if R.bv_norm > 1e-12:
        bv = R.even().c[4:7] / R.bv_norm
        c = np.zeros(8, np.float64)
        c[0] = math.cos(h)
        c[4:7] = bv * math.sin(h)
        rotor = Mv(c)
    else:
        rotor = Mv(np.array([math.cos(h), 0, 0, 0, math.sin(h), 0, 0, 0]))

    # Persistence
    D = _distance_matrix(vecs)
    pairs, betti = _persistence_pairs(D)

    return EncounterComplex(
        rotor=rotor, angle=ang, curvature=curv,
        betti=betti, persistence=pairs,
    )


def encounter(text, embed_fn=None):
    """Backward-compatible: returns (angle, curvature, rotor)."""
    cx = encounter_complex(text, embed_fn)
    return cx.angle, cx.curvature, cx.rotor
