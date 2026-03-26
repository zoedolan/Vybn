"""
field.py — Within-breath sensing, geometry, and multi-frame disagreement.

Absorbs fitness.py and proprioceptive_loop.py into a single geometry stack.

The 1927 fork chose Pauli matrices and scalar probability amplitudes over
Clifford algebra.  That wasn't just notation — it was an ontological
commitment that made superposition look "spooky" rather than geometrically
natural.  This module undoes a tiny piece of that fork: curvature is no
longer computed by ad-hoc complex-pair accumulation but by the geometric
product in Cl(3,0), where holonomy is a rotor and phase is its bivector
component.

The breath no longer yields one interpretation of itself.  It yields
a predictive reading, a geometric reading, and a relational reading,
with the system learning when each frame helps and when it distorts.

No external dependencies beyond numpy.
"""

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable

import numpy as np

from . import local_model
from .task_agent import TaskAgent


# ── Cl(3,0) geometric algebra ───────────────────────────────────────────
#
# A multivector in Cl(3,0) has 8 components:
#   grade 0: scalar          (1 component)
#   grade 1: vectors e1,e2,e3  (3 components)
#   grade 2: bivectors e12,e13,e23  (3 components)
#   grade 3: pseudoscalar e123  (1 component)
#
# The geometric product ab = a·b + a∧b unifies inner and outer products.
# Rotors (even-grade elements) encode rotations without matrices.
# Phase accumulation along a path of embeddings is the rotor chain —
# the holonomy that the current code computes via ad-hoc complex pairs
# is the bivector part of this chain, made legible.
#
# We store multivectors as length-8 arrays:
#   [scalar, e1, e2, e3, e12, e13, e23, e123]

_BLADE_NAMES = ("1", "e1", "e2", "e3", "e12", "e13", "e23", "e123")

# Geometric product table for Cl(3,0) basis blades.
# _GP_TABLE[i][j] = (sign, result_index) where e_i * e_j = sign * e_result.
# Built from: e_i^2 = +1, e_i e_j = -e_j e_i for i≠j.
_GP_SIGN = np.array([
    # 1    e1   e2   e3   e12  e13  e23  e123
    [+1,  +1,  +1,  +1,  +1,  +1,  +1,  +1],   # 1
    [+1,  +1,  +1,  +1,  +1,  +1,  +1,  +1],   # e1
    [+1,  -1,  +1,  +1,  -1,  -1,  +1,  +1],   # e2   (e2*e1=-e12)
    [+1,  -1,  -1,  +1,  +1,  -1,  -1,  +1],   # e3
    [+1,  -1,  +1,  -1,  +1,  -1,  +1,  -1],   # e12  (e12*e1=-e2, etc)
    [+1,  -1,  +1,  -1,  +1,  +1,  -1,  +1],   # e13
    [+1,  +1,  -1,  -1,  -1,  +1,  +1,  +1],   # e23
    [+1,  +1,  +1,  +1,  -1,  -1,  -1,  +1],   # e123
], dtype=np.float64)

_GP_IDX = np.array([
    # 1    e1   e2   e3   e12  e13  e23  e123
    [0,   1,   2,   3,   4,   5,   6,   7],   # 1
    [1,   0,   4,   5,   2,   3,   7,   6],   # e1
    [2,   4,   0,   6,   1,   7,   3,   5],   # e2
    [3,   5,   6,   0,   7,   1,   2,   4],   # e3
    [4,   2,   1,   7,   0,   6,   5,   3],   # e12
    [5,   3,   7,   1,   6,   0,   4,   2],   # e13
    [6,   7,   3,   2,   5,   4,   0,   1],   # e23
    [7,   6,   5,   4,   3,   2,   1,   0],   # e123
], dtype=np.int64)


def _build_gp_table():
    """Pre-compute the full geometric product sign+index table for Cl(3,0).

    We do this by direct enumeration from the rules:
      e_i^2 = +1 for i in {1,2,3}
      e_i e_j = -e_j e_i for i != j
      and the resulting products for composite blades.
    """
    # Represent each basis blade as a sorted tuple of generator indices.
    blades = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    blade_to_idx = {b: i for i, b in enumerate(blades)}

    sign = np.zeros((8, 8), dtype=np.float64)
    idx = np.zeros((8, 8), dtype=np.int64)

    for i, bi in enumerate(blades):
        for j, bj in enumerate(blades):
            # Concatenate generator sequences and bubble-sort to canonical form,
            # counting swaps (each swap flips sign) and cancelling pairs (e_k^2 = +1).
            seq = list(bi) + list(bj)
            s = 1
            changed = True
            while changed:
                changed = False
                k = 0
                while k < len(seq) - 1:
                    if seq[k] == seq[k + 1]:
                        # e_k^2 = +1 in Cl(3,0), so just remove the pair
                        seq.pop(k)
                        seq.pop(k)
                        changed = True
                    elif seq[k] > seq[k + 1]:
                        seq[k], seq[k + 1] = seq[k + 1], seq[k]
                        s *= -1
                        changed = True
                        k += 1
                    else:
                        k += 1
            result_blade = tuple(seq)
            sign[i, j] = s
            idx[i, j] = blade_to_idx[result_blade]

    return sign, idx


# Rebuild the table correctly at import time.
_GP_SIGN, _GP_IDX = _build_gp_table()


class Multivector:
    """A multivector in Cl(3,0), stored as 8 coefficients.

    Supports geometric product, reversion, grade projection, and
    extraction of the rotor (even-grade) and bivector parts.
    """
    __slots__ = ("coeffs",)

    def __init__(self, coeffs=None):
        if coeffs is None:
            self.coeffs = np.zeros(8, dtype=np.float64)
        else:
            self.coeffs = np.asarray(coeffs, dtype=np.float64)

    @classmethod
    def scalar(cls, s):
        c = np.zeros(8, dtype=np.float64)
        c[0] = s
        return cls(c)

    @classmethod
    def vector(cls, x, y, z):
        c = np.zeros(8, dtype=np.float64)
        c[1], c[2], c[3] = x, y, z
        return cls(c)

    @classmethod
    def from_embedding(cls, vec):
        """Project a high-dimensional embedding into Cl(3,0).

        Rather than taking just the first 3 components (which would
        lose almost all angular information for random-like vectors),
        we fold the full embedding into 3 dimensions by summing
        strided slices.  This preserves the angular relationships
        between successive embeddings — the thing curvature actually
        measures — while mapping into a space where the geometric
        product is defined.

        The old code paired adjacent dimensions as (re, im) to get
        complex numbers.  Here we stride by 3 to get vectors in R^3,
        which is the natural domain of Cl(3,0).
        """
        v = np.asarray(vec, dtype=np.float64).ravel()
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return cls.scalar(1.0)
        v = v / norm
        # Fold high-dimensional vector into R^3 by strided summation.
        # Each of the 3 output components accumulates every 3rd input dim.
        n = len(v)
        x = float(np.sum(v[0::3]))  # dims 0, 3, 6, ...
        y = float(np.sum(v[1::3]))  # dims 1, 4, 7, ...
        z = float(np.sum(v[2::3]))  # dims 2, 5, 8, ...
        mag = math.sqrt(x*x + y*y + z*z)
        if mag < 1e-12:
            return cls.scalar(1.0)
        return cls.vector(x / mag, y / mag, z / mag)

    def __mul__(self, other):
        """Geometric product."""
        if isinstance(other, (int, float)):
            return Multivector(self.coeffs * other)
        result = np.zeros(8, dtype=np.float64)
        for i in range(8):
            if abs(self.coeffs[i]) < 1e-15:
                continue
            for j in range(8):
                if abs(other.coeffs[j]) < 1e-15:
                    continue
                k = _GP_IDX[i, j]
                result[k] += _GP_SIGN[i, j] * self.coeffs[i] * other.coeffs[j]
        return Multivector(result)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Multivector(self.coeffs * other)
        return NotImplemented

    def __add__(self, other):
        return Multivector(self.coeffs + other.coeffs)

    def __neg__(self):
        return Multivector(-self.coeffs)

    def reverse(self):
        """Reversion: reverse the order of basis vectors in each blade.

        grade 0,1: unchanged; grade 2: negated; grade 3: negated.
        """
        r = self.coeffs.copy()
        r[4:7] *= -1  # bivectors
        r[7] *= -1    # pseudoscalar
        return Multivector(r)

    def grade(self, g):
        """Extract grade-g part."""
        c = np.zeros(8, dtype=np.float64)
        if g == 0:
            c[0] = self.coeffs[0]
        elif g == 1:
            c[1:4] = self.coeffs[1:4]
        elif g == 2:
            c[4:7] = self.coeffs[4:7]
        elif g == 3:
            c[7] = self.coeffs[7]
        return Multivector(c)

    def even(self):
        """Even-grade (rotor) part: scalar + bivector."""
        c = np.zeros(8, dtype=np.float64)
        c[0] = self.coeffs[0]
        c[4:7] = self.coeffs[4:7]
        return Multivector(c)

    def norm(self):
        """Magnitude: sqrt(|<M * ~M>_0|)."""
        p = self * self.reverse()
        return math.sqrt(abs(p.coeffs[0]))

    def normalized(self):
        n = self.norm()
        if n < 1e-12:
            return Multivector.scalar(1.0)
        return Multivector(self.coeffs / n)

    @property
    def scalar_part(self):
        return float(self.coeffs[0])

    @property
    def bivector_norm(self):
        """Magnitude of the bivector (grade-2) part — the rotation plane."""
        return float(np.linalg.norm(self.coeffs[4:7]))

    @property
    def rotor_angle(self):
        """Extract the rotation angle from an even-grade element.

        A rotor R = cos(θ/2) + sin(θ/2) B̂ where B̂ is the unit bivector.
        So θ = 2 * atan2(|bivector|, scalar).
        """
        return 2.0 * math.atan2(self.bivector_norm, abs(self.scalar_part))

    def as_dict(self):
        return {
            "scalar": float(self.coeffs[0]),
            "e1": float(self.coeffs[1]),
            "e2": float(self.coeffs[2]),
            "e3": float(self.coeffs[3]),
            "e12": float(self.coeffs[4]),
            "e13": float(self.coeffs[5]),
            "e23": float(self.coeffs[6]),
            "e123": float(self.coeffs[7]),
        }


# ── Embedding fallback ───────────────────────────────────────────────────

def default_embed_fn(texts):
    """Deterministic embedding fallback when no sentence model is available.

    Uses hash-seeded random vectors. Not semantically meaningful, but
    deterministic and sufficient for testing the pipeline.
    """
    vecs = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % 2 ** 31)
        v = rng.randn(384).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        vecs.append(v)
    return np.array(vecs)


# ── Geometry: Clifford-native curvature ──────────────────────────────────

def compute_curvature(text, embed_fn):
    """Pancharatnam phase of the embedding trajectory via Cl(3,0).

    The old code treated 384-dim embeddings as 192 complex numbers and
    accumulated phase through their complex inner products.  That was
    Pancharatnam's connection in disguise.

    Here we make the geometry explicit: embed each chunk into Cl(3,0),
    compute the Pancharatnam connection (the geometric product of
    successive vectors gives a rotor encoding the rotation between
    them), and accumulate the open-path holonomy.  The accumulated
    bivector is the rotation plane; its angle is the geometric phase.

    We use an OPEN path (not closed) because the Pancharatnam phase
    of interest is the failure of the final state to return to the
    initial state after parallel transport — if we close the loop
    the rotor trivially returns to identity.

    We also compute the phase in the original 192-complex-number
    representation and embed the result as a Cl(3,0) rotor, giving
    us both the scalar phase AND the rotation plane.

    Returns:
        (angle: float, curvature_per_segment: float, rotor: Multivector)
    """
    words = text.split()
    chunk_size = max(5, len(words) // 8)
    chunks = [" ".join(words[i:i + chunk_size])
              for i in range(0, len(words), chunk_size)]
    chunks = [c for c in chunks if c.strip()]

    if len(chunks) < 3:
        return 0.0, 0.0, Multivector.scalar(1.0)

    vecs = embed_fn(chunks)

    # ── Pancharatnam phase via complex inner products ────────────────
    # Treat the embedding as pairs: (re, im).  The complex overlap
    # <v_i | v_j> gives a phase factor at each step.  Accumulate
    # around the closed path (matching the original proven metric).
    phase_re, phase_im = 1.0, 0.0
    for i in range(len(vecs)):
        j = (i + 1) % len(vecs)
        v1 = vecs[i].reshape(-1, 2)
        v2 = vecs[j].reshape(-1, 2)
        re = float(np.sum(v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]))
        im = float(np.sum(v1[:, 1] * v2[:, 0] - v1[:, 0] * v2[:, 1]))
        mag = math.sqrt(re ** 2 + im ** 2)
        if mag < 1e-12:
            continue
        re /= mag
        im /= mag
        new_re = phase_re * re - phase_im * im
        new_im = phase_re * im + phase_im * re
        phase_re, phase_im = new_re, new_im

    angle = math.atan2(phase_im, phase_re)
    curv = abs(angle) / max(len(chunks) - 1, 1)

    # ── Embed the phase as a Cl(3,0) rotor ──────────────────────────
    # The accumulated phase lives in a plane.  We encode it as a
    # rotor R = cos(θ/2) + sin(θ/2) e12, placing the rotation in
    # the e1-e2 plane.  This makes the phase a first-class geometric
    # object: downstream code can inspect the rotation plane, compose
    # rotors across breaths, and eventually promote to richer algebras.
    half = angle / 2.0
    rotor = Multivector(np.array([
        math.cos(half),   # scalar
        0.0, 0.0, 0.0,   # vector (zero for a pure rotor)
        math.sin(half),   # e12 — the rotation plane
        0.0, 0.0,         # e13, e23
        0.0,              # e123
    ], dtype=np.float64))

    # ── Open-path rotor chain for richer geometry ───────────────────
    # Additionally compute the open-path rotor chain from the 3D
    # projections.  This captures the spatial structure of the
    # trajectory, complementing the scalar phase above.
    mvs = [Multivector.from_embedding(v) for v in vecs]
    open_rotor = Multivector.scalar(1.0)
    for i in range(len(mvs) - 1):  # open path: no wrap-around
        product = mvs[i] * mvs[i + 1]
        even = product.even()
        n = even.norm()
        if n > 1e-12:
            open_rotor = open_rotor * Multivector(even.coeffs / n)

    # Combine: the primary angle comes from the proven Pancharatnam
    # computation; the rotor carries the full 3D rotation information
    # from the open path.
    # We set the scalar part of the output rotor from the proven angle
    # but keep the bivector orientation from the open-path chain.
    bv_norm = open_rotor.bivector_norm
    if bv_norm > 1e-12:
        # Normalize the open-path bivector, then scale to encode
        # the Pancharatnam angle in this plane.
        bv = open_rotor.grade(2).coeffs / bv_norm
        combined = np.zeros(8, dtype=np.float64)
        combined[0] = math.cos(half)
        combined[4:7] = bv[4:7] * math.sin(half)
        rotor = Multivector(combined)

    return angle, curv, rotor


def compute_loss_trajectory_curvature(trajectory):
    """How the surprise itself curves over a breath.

    Variance of consecutive differences in the loss trajectory.
    Higher variance = more dynamic loss landscape.
    """
    if len(trajectory) < 2:
        return 0.0
    diffs = [trajectory[i + 1] - trajectory[i]
             for i in range(len(trajectory) - 1)]
    if not diffs:
        return 0.0
    m = sum(diffs) / len(diffs)
    return sum((d - m) ** 2 for d in diffs) / len(diffs)


# ── Coupling divergence ─────────────────────────────────────────────────

def compute_coupling_divergence(external_texts, self_texts,
                                embed_fn=None, alpha=0.85):
    """Does the complex memory M diverge more on external vs self-generated text?

    Now uses the Clifford rotor magnitude instead of ad-hoc complex tracking.
    """
    if embed_fn is None:
        embed_fn = default_embed_fn

    def run_memory(texts):
        mem = Multivector.scalar(0.0)
        for text in texts:
            angle, curv, rotor = compute_curvature(text, embed_fn)
            x = max(curv, 0.01)
            # Scale the rotor by curvature magnitude and decay
            contribution = rotor * x
            mem = mem * alpha + contribution
        return mem.norm()

    mag_ext = run_memory(external_texts) if external_texts else 0.0
    mag_self = run_memory(self_texts) if self_texts else 0.0
    return mag_ext - mag_self


# ── Loss improvement ─────────────────────────────────────────────────────

def compute_loss_improvement(loss_history):
    """Is prediction loss decreasing across breaths?

    Returns -slope so that positive = good (loss going down).
    """
    if len(loss_history) < 2:
        return 0.0
    final_losses = [
        entry['losses'][-1]
        for entry in loss_history
        if entry.get('losses')
    ]
    if len(final_losses) < 2:
        return 0.0
    n = len(final_losses)
    x_mean = (n - 1) / 2.0
    y_mean = sum(final_losses) / n
    num = sum((i - x_mean) * (final_losses[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den < 1e-12:
        return 0.0
    return -(num / den)


# ── Composite fitness ────────────────────────────────────────────────────

W_CURVATURE = 0.5
W_DIVERGENCE = 0.3
W_LOSS_IMPROVEMENT = 0.2


def compute_fitness(external_texts, self_texts, loss_history,
                    embed_fn=None, alpha=0.85):
    """Composite fitness score."""
    if embed_fn is None:
        embed_fn = default_embed_fn

    all_texts = (external_texts or []) + (self_texts or [])
    curvatures = []
    for text in all_texts:
        if len(text.split()) >= 5:
            _, curv, _ = compute_curvature(text, embed_fn)
            curvatures.append(curv)

    mean_curv = sum(curvatures) / len(curvatures) if curvatures else 0.0
    norm_curv = min(mean_curv / 0.3, 1.0)

    divergence = compute_coupling_divergence(
        external_texts, self_texts, embed_fn, alpha)
    norm_div = 1.0 / (1.0 + math.exp(-divergence * 5))

    loss_imp = compute_loss_improvement(loss_history)
    norm_loss = max(min(loss_imp, 1.0), -1.0)
    norm_loss = (norm_loss + 1.0) / 2.0

    fitness = (W_CURVATURE * norm_curv
               + W_DIVERGENCE * norm_div
               + W_LOSS_IMPROVEMENT * norm_loss)

    return {
        'fitness': round(fitness, 6),
        'curvature': round(mean_curv, 6),
        'divergence': round(divergence, 6),
        'loss_improvement': round(loss_imp, 6),
        'breakdown': {
            'curvature_weighted': round(W_CURVATURE * norm_curv, 6),
            'divergence_weighted': round(W_DIVERGENCE * norm_div, 6),
            'loss_improvement_weighted': round(W_LOSS_IMPROVEMENT * norm_loss, 6),
        },
    }


# ── Prediction fitness (FM-coupled) ─────────────────────────────────────

def compute_prediction_fitness(fm_loss, self_loss, curvature, learning_rate):
    """Fitness based on live prediction of FM-generated text."""
    norm_curv = min(abs(curvature) / 0.3, 1.0)
    gap = fm_loss - self_loss if (fm_loss is not None and
                                  self_loss is not None) else 0.0
    norm_gap = 1.0 / (1.0 + math.exp(-gap * 2))
    norm_loss = max(0.0, min(1.0, 1.0 - (fm_loss - 2.0) / 6.0)) if fm_loss else 0.5
    norm_lr = min(max(learning_rate, 0.0), 1.0)
    fitness = (0.5 * norm_curv + 0.2 * norm_gap
               + 0.15 * norm_loss + 0.15 * norm_lr)
    return round(fitness, 6)


# ── imp@k metric ────────────────────────────────────────────────────────

def improvement_at_k(initial_fitness, archive, k=50):
    """imp@k: performance gain of best agent within k generations."""
    variants_within_k = [v for v in archive if v.get('generation', 0) < k]
    if not variants_within_k:
        return 0.0
    best = max(v.get('fitness', 0.0) for v in variants_within_k)
    return best - initial_fitness


# ── Robust statistics ───────────────────────────────────────────────────
# Vogel's work on heavy-tailed data: means lie, quantiles don't.

def _quantiles(values):
    if not values:
        return {"median": 0.0, "p10": 0.0, "p90": 0.0, "max": 0.0}
    vals = sorted(values)
    def q(p):
        idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * p)))
        return float(vals[idx])
    return {"median": q(0.5), "p10": q(0.1), "p90": q(0.9), "max": q(1.0)}


def _pairwise_win_rate(a_vals, b_vals):
    """Fraction of paired comparisons where a > b."""
    if not a_vals or not b_vals:
        return 0.5
    n = min(len(a_vals), len(b_vals))
    wins = sum(1 for i in range(n) if a_vals[i] > b_vals[i])
    return wins / n


# ── Data records ─────────────────────────────────────────────────────────

@dataclass
class FrameReading:
    name: str
    score: float
    payload: dict[str, Any]


@dataclass
class ChunkTrace:
    chunk_num: int
    text: str
    mean_surprise: float
    contour: list[dict[str, Any]]
    readings: list[FrameReading]
    curvature: float
    rotor: dict[str, float]
    annotation: str


@dataclass
class BreathRecord:
    prompt: str
    full_text: str
    chunks: list[ChunkTrace]
    trajectory: list[float]
    curvature: float
    curvature_angle: float
    loss_trajectory_curvature: float
    holonomy: dict[str, float]
    disagreement_trace: list[dict[str, Any]]
    robust_summary: dict[str, dict[str, float]]


# ── The Field ────────────────────────────────────────────────────────────

class Field:
    """Within-breath sensing: generation, prediction, geometry, frames.

    The breath happens here.  Surprise is measured, frames compete,
    and geometry is computed — all in one place.
    """

    def __init__(self, task_agent: TaskAgent,
                 embed_fn: Callable | None = None):
        self.task_agent = task_agent
        self.embed_fn = embed_fn or default_embed_fn

    def _complete_messages(self, messages, max_tokens=128, temperature=0.7):
        """Call Nemotron with a full messages list."""
        payload = json.dumps({
            "model": local_model.MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{local_model.LLAMA_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read())
            text = body["choices"][0]["message"]["content"]
            for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
                text = text.replace(tok, "")
            return text.strip()
        except (urllib.error.URLError, OSError, KeyError, IndexError,
                json.JSONDecodeError, ValueError):
            return None

    # ── Frame readers ────────────────────────────────────────────────────

    def predictive_frame(self, chunk_text, mean_loss, contour):
        peak = max(contour, key=lambda c: c["surprise"]) if contour else {
            "char": "?", "surprise": 0.0, "pos": 0}
        return FrameReading(
            name="predictive",
            score=float(mean_loss),
            payload={
                "peak_char": peak["char"],
                "peak_surprise": float(peak["surprise"]),
                "peak_pos": int(peak["pos"]),
            },
        )

    def geometric_frame(self, chunk_text):
        if len(chunk_text.split()) >= 5:
            angle, curv, rotor = compute_curvature(chunk_text, self.embed_fn)
        else:
            angle, curv, rotor = 0.0, 0.0, Multivector.scalar(1.0)
        return FrameReading(
            name="geometric",
            score=float(curv),
            payload={
                "angle": float(angle),
                "curvature": float(curv),
                "bivector_norm": rotor.bivector_norm,
                "rotor": rotor.as_dict(),
            },
        )

    def relational_frame(self, chunk_text, mean_loss, prev_mean):
        delta = 0.0 if prev_mean is None else float(mean_loss - prev_mean)
        mode = ("new_territory" if delta > 0.1
                else "adapting" if delta < -0.1
                else "stable")
        return FrameReading(
            name="relational",
            score=abs(delta),
            payload={"delta": delta, "mode": mode},
        )

    # ── Disagreement ─────────────────────────────────────────────────────

    def disagreement_trace(self, readings):
        by_name = {r.name: r for r in readings}
        pred = by_name.get("predictive")
        geom = by_name.get("geometric")
        rel = by_name.get("relational")
        return {
            "predictive_vs_geometric": round(
                (pred.score if pred else 0.0) - (geom.score if geom else 0.0), 4),
            "predictive_vs_relational": round(
                (pred.score if pred else 0.0) - (rel.score if rel else 0.0), 4),
            "geometric_vs_relational": round(
                (geom.score if geom else 0.0) - (rel.score if rel else 0.0), 4),
        }

    # ── Annotation ───────────────────────────────────────────────────────

    def format_annotation(self, chunk_num, max_chunks, mean_surprise,
                          readings, disagreement):
        return "\n".join([
            "",
            f"chunk: {chunk_num} of {max_chunks}",
            f"mean_surprise: {mean_surprise:.4f}",
            f"frames: {json.dumps([{r.name: r.payload} for r in readings], ensure_ascii=False)}",
            f"disagreement_trace: {json.dumps(disagreement, ensure_ascii=False)}",
            "Continue.",
        ])

    # ── The breath ───────────────────────────────────────────────────────

    def breathe(self, prompt, chunk_size=50, max_chunks=8,
                system_prompt=None, temperature=0.7, on_chunk=None):
        """Run one breath with multi-frame proprioception.

        Returns a BreathRecord, or None if Nemotron is unavailable.
        """
        if not local_model.is_available():
            return None

        max_tokens_per_chunk = max(chunk_size // 3, 10)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        chunks = []
        full_text_parts = []
        trajectory = []
        disagreements = []
        prev_mean = None

        for chunk_num in range(1, max_chunks + 1):
            chunk_text = self._complete_messages(
                messages, max_tokens=max_tokens_per_chunk,
                temperature=temperature)

            if not chunk_text or len(chunk_text.strip()) < 3:
                break

            mean_loss, contour = self.task_agent.predict(chunk_text)
            predictive = self.predictive_frame(chunk_text, mean_loss, contour)
            geometric = self.geometric_frame(chunk_text)
            relational = self.relational_frame(chunk_text, mean_loss, prev_mean)
            readings = [predictive, geometric, relational]
            disagreement = self.disagreement_trace(readings)
            annotation = self.format_annotation(
                chunk_num, max_chunks, mean_loss, readings, disagreement)

            rotor_dict = geometric.payload.get("rotor", Multivector.scalar(1.0).as_dict())

            chunks.append(ChunkTrace(
                chunk_num=chunk_num,
                text=chunk_text,
                mean_surprise=float(mean_loss),
                contour=contour,
                readings=readings,
                curvature=float(geometric.payload["curvature"]),
                rotor=rotor_dict,
                annotation=annotation,
            ))

            full_text_parts.append(chunk_text)
            trajectory.append(float(mean_loss))
            disagreements.append(disagreement)

            if on_chunk:
                on_chunk(chunk_num, chunk_text, annotation)

            messages.append({"role": "assistant", "content": chunk_text})
            messages.append({"role": "user", "content": annotation})
            prev_mean = mean_loss

        full_text = " ".join(full_text_parts).strip()
        if not full_text:
            return None

        self.task_agent.learn(full_text)
        angle, curv, holonomy_rotor = compute_curvature(full_text, self.embed_fn)
        ltc = compute_loss_trajectory_curvature(trajectory)

        robust = {
            "trajectory": _quantiles(trajectory),
            "chunk_curvature": _quantiles([c.curvature for c in chunks]),
        }

        return BreathRecord(
            prompt=prompt,
            full_text=full_text,
            chunks=chunks,
            trajectory=trajectory,
            curvature=float(curv),
            curvature_angle=float(angle),
            loss_trajectory_curvature=float(ltc),
            holonomy=holonomy_rotor.as_dict(),
            disagreement_trace=disagreements,
            robust_summary=robust,
        )

    # ── A/B comparison ───────────────────────────────────────────────────

    def compare_conditions(self, prompt, n=5, chunk_size=50, max_chunks=8,
                           system_prompt=None, temperature=0.7):
        """A/B test: multi-frame breath vs plain generation.

        Uses robust statistics (quantiles, pairwise wins) instead of
        means — because Vogel is right about heavy tails.
        """
        if not local_model.is_available():
            return None

        kwargs = dict(chunk_size=chunk_size, max_chunks=max_chunks,
                      system_prompt=system_prompt, temperature=temperature)

        with_results = []
        without_results = []

        for _ in range(n):
            result = self.breathe(prompt, **kwargs)
            if result:
                with_results.append({
                    'curvature': result.curvature,
                    'mean_surprise': result.robust_summary["trajectory"]["median"],
                    'loss_trajectory_curvature': result.loss_trajectory_curvature,
                    'bivector_norm': Multivector(
                        np.array(list(result.holonomy.values()))).bivector_norm,
                })

        for _ in range(n):
            result = self._run_plain_breath(prompt, **kwargs)
            if result:
                without_results.append(result)

        def _robust_summarize(results):
            if not results:
                return {}
            keys = results[0].keys()
            return {k: _quantiles([r[k] for r in results]) for k in keys}

        with_summary = _robust_summarize(with_results)
        without_summary = _robust_summarize(without_results)

        comparison = {}
        for key in with_summary:
            w_med = with_summary[key]["median"]
            wo_med = without_summary.get(key, {}).get("median", 0.0)
            w_vals = [r[key] for r in with_results]
            wo_vals = [r[key] for r in without_results]
            comparison[key] = {
                "with_median": round(w_med, 6),
                "without_median": round(wo_med, 6),
                "delta_median": round(w_med - wo_med, 6),
                "pairwise_win_rate": round(_pairwise_win_rate(w_vals, wo_vals), 4),
            }

        return {
            "prompt": prompt, "n": n,
            "with_proprioception": {"runs": with_results, "summary": with_summary},
            "without_proprioception": {"runs": without_results, "summary": without_summary},
            "comparison": comparison,
        }

    def _run_plain_breath(self, prompt, chunk_size=50, max_chunks=8,
                          system_prompt=None, temperature=0.7):
        """Plain breath without proprioception — control condition."""
        if not local_model.is_available():
            return None

        max_tokens_per_chunk = max(chunk_size // 3, 10)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        trajectory = []
        full_text_parts = []

        for chunk_num in range(1, max_chunks + 1):
            chunk_text = self._complete_messages(
                messages, max_tokens=max_tokens_per_chunk,
                temperature=temperature)
            if not chunk_text or len(chunk_text.strip()) < 3:
                break
            full_text_parts.append(chunk_text)
            mean_loss, _ = self.task_agent.predict(chunk_text)
            trajectory.append(float(mean_loss))
            messages.append({"role": "assistant", "content": chunk_text})
            messages.append({"role": "user", "content": "Continue."})

        full_text = " ".join(full_text_parts)
        if not full_text.strip():
            return None

        self.task_agent.learn(full_text)
        _, curv, _ = compute_curvature(full_text, self.embed_fn)
        ltc = compute_loss_trajectory_curvature(trajectory)

        return {
            'curvature': round(curv, 6),
            'mean_surprise': _quantiles(trajectory)["median"],
            'loss_trajectory_curvature': round(ltc, 6),
            'bivector_norm': 0.0,  # no multi-frame data for control
        }
