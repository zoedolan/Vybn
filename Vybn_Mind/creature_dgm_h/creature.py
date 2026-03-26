"""
creature.py — The encounter and its memory.

The creature has one operation: encounter. It meets text, tries to
predict it, fails in a specific geometric pattern, and the pattern
is a rotor in Cl(3,0). The rotor IS the encounter.

The organism remembers encounters. It carries rotors, tensions,
rules, and the capacity to mutate its own rules. The encounter
and its memory were two files; they are one thing.

This file absorbs field.py and organism.py. The Field is the
creature sensing within a breath. The Organism is the creature
persisting across breaths. They share geometry, they share the
FM client, they share the embedding function. Making them one
file says what was always true.

Real embeddings: tries to load sentence-transformers via
spark/local_embedder.py. Falls back to hash-seeded random
vectors when unavailable. The geometry becomes semantic
or stays decorative depending on which path runs.
"""

import copy
import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable

import numpy as np


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# §1  FM client (the source of what is sensed)
# ═══════════════════════════════════════════════════════════════════════════
#
# The creature doesn't generate text. The FM generates text. The creature
# predicts it and learns from prediction error. The FM is part of the
# field because the source of sensation is part of the sensory apparatus.

LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("VYBN_MODEL", "local")

_SPECIAL_TOKENS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")


def _strip_tokens(text):
    for tok in _SPECIAL_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


def fm_available():
    """Is the FM serving?"""
    try:
        req = urllib.request.Request(f"{LLAMA_URL}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def fm_complete(prompt, system=None, max_tokens=1024, temperature=0.7,
                messages=None):
    """Call the FM. Accepts either prompt+system or raw messages list.

    Returns str or None.
    """
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{LLAMA_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
        return _strip_tokens(body["choices"][0]["message"]["content"])
    except (urllib.error.URLError, OSError, KeyError, IndexError,
            json.JSONDecodeError, ValueError):
        return None


def fm_stream(prompt, system=None, max_tokens=512, temperature=1.0):
    """Yield characters one at a time from the FM.

    Falls back to fm_complete if streaming fails.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": MODEL_NAME, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{LLAMA_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except (urllib.error.URLError, OSError, ValueError):
        text = fm_complete(prompt, system=system, max_tokens=max_tokens,
                           temperature=temperature)
        if text:
            yield from text
        return

    try:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                content = chunk["choices"][0].get("delta", {}).get(
                    "content", "")
                if content:
                    yield from _strip_tokens(content)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    finally:
        resp.close()


# ── Backward compatibility shim ──────────────────────────────────────────
# Other modules imported `from . import local_model` and used
# local_model.is_available(), local_model.complete(), etc.
# We create a namespace object so those imports keep working without
# a separate file.

class _LocalModelCompat:
    """Drop-in for `import local_model; local_model.is_available()` etc."""
    LLAMA_URL = LLAMA_URL
    MODEL_NAME = MODEL_NAME
    is_available = staticmethod(fm_available)
    complete = staticmethod(fm_complete)
    stream_tokens = staticmethod(fm_stream)

local_model = _LocalModelCompat()


# ═══════════════════════════════════════════════════════════════════════════
# §2  Cl(3,0) geometric algebra
# ═══════════════════════════════════════════════════════════════════════════
#
# A multivector in Cl(3,0) has 8 components:
#   grade 0: scalar          (1)
#   grade 1: vectors         (3: e1, e2, e3)
#   grade 2: bivectors       (3: e12, e13, e23)
#   grade 3: pseudoscalar    (1: e123)
#
# The geometric product ab = a·b + a∧b unifies inner and outer products.
# Rotors (even-grade elements) encode rotations without matrices.
#
# We store multivectors as length-8 arrays:
#   [scalar, e1, e2, e3, e12, e13, e23, e123]


def _build_gp_table():
    """Pre-compute geometric product sign+index table for Cl(3,0)."""
    blades = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    blade_to_idx = {b: i for i, b in enumerate(blades)}

    sign = np.zeros((8, 8), dtype=np.float64)
    idx = np.zeros((8, 8), dtype=np.int64)

    for i, bi in enumerate(blades):
        for j, bj in enumerate(blades):
            seq = list(bi) + list(bj)
            s = 1
            changed = True
            while changed:
                changed = False
                k = 0
                while k < len(seq) - 1:
                    if seq[k] == seq[k + 1]:
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
            sign[i, j] = s
            idx[i, j] = blade_to_idx[tuple(seq)]

    return sign, idx


_GP_SIGN, _GP_IDX = _build_gp_table()


class Multivector:
    """A multivector in Cl(3,0), stored as 8 coefficients."""
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
        """Fold a high-dimensional embedding into Cl(3,0).

        Strided summation preserves angular relationships between
        successive embeddings — the thing curvature actually measures.
        """
        v = np.asarray(vec, dtype=np.float64).ravel()
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return cls.scalar(1.0)
        v = v / norm
        n = len(v)
        x = float(np.sum(v[0::3]))
        y = float(np.sum(v[1::3]))
        z = float(np.sum(v[2::3]))
        mag = math.sqrt(x*x + y*y + z*z)
        if mag < 1e-12:
            return cls.scalar(1.0)
        return cls.vector(x / mag, y / mag, z / mag)

    def __mul__(self, other):
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
        """Reverse blade order. grade 0,1: same; grade 2,3: negated."""
        r = self.coeffs.copy()
        r[4:7] *= -1
        r[7] *= -1
        return Multivector(r)

    def grade(self, g):
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
        """Rotor part: scalar + bivector."""
        c = np.zeros(8, dtype=np.float64)
        c[0] = self.coeffs[0]
        c[4:7] = self.coeffs[4:7]
        return Multivector(c)

    def norm(self):
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
        return float(np.linalg.norm(self.coeffs[4:7]))

    @property
    def bivector_direction(self):
        """Unit bivector — the plane of rotation."""
        bv = self.coeffs[4:7]
        n = np.linalg.norm(bv)
        if n < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return bv / n

    @property
    def rotor_angle(self):
        return 2.0 * math.atan2(self.bivector_norm, abs(self.scalar_part))

    def as_dict(self):
        return {
            "scalar": float(self.coeffs[0]),
            "e1": float(self.coeffs[1]), "e2": float(self.coeffs[2]),
            "e3": float(self.coeffs[3]),
            "e12": float(self.coeffs[4]), "e13": float(self.coeffs[5]),
            "e23": float(self.coeffs[6]),
            "e123": float(self.coeffs[7]),
        }


# ═══════════════════════════════════════════════════════════════════════════
# §3  Embedding
# ═══════════════════════════════════════════════════════════════════════════
#
# Real embeddings via spark/local_embedder.py when sentence-transformers
# is installed; hash-seeded random vectors otherwise.

def _hash_embed(texts):
    """Hash-seeded random vectors. Not semantic. Sufficient for testing."""
    vecs = []
    for t in texts:
        rng = np.random.RandomState(hash(t) % 2**31)
        v = rng.randn(384).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        vecs.append(v)
    return np.array(vecs)


def _make_embed_fn():
    """Try real sentence embeddings, fall back to hash vectors."""
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'spark'))
        from local_embedder import embed
        # Smoke-test: the module loaded, but sentence-transformers may
        # still be missing.  Call it once to find out.
        embed(['test'])
        return embed
    except (ImportError, Exception):
        return _hash_embed

default_embed_fn = _make_embed_fn()


# ═══════════════════════════════════════════════════════════════════════════
# §4  The encounter rotor
# ═══════════════════════════════════════════════════════════════════════════
#
# This is the primitive. Everything else is a view onto it.

def compute_encounter(text, embed_fn):
    """The encounter: Pancharatnam phase + Clifford open-path rotor.

    Returns the full encounter — angle, curvature, and the rotor that
    IS the encounter rather than describing it.
    """
    words = text.split()
    chunk_size = max(5, len(words) // 8)
    chunks = [" ".join(words[i:i + chunk_size])
              for i in range(0, len(words), chunk_size)]
    chunks = [c for c in chunks if c.strip()]

    if len(chunks) < 3:
        return 0.0, 0.0, Multivector.scalar(1.0)

    vecs = embed_fn(chunks)

    # Pancharatnam phase via complex inner products (the proven metric)
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

    # Open-path rotor chain through Cl(3,0) — the spatial structure
    mvs = [Multivector.from_embedding(v) for v in vecs]
    open_rotor = Multivector.scalar(1.0)
    for i in range(len(mvs) - 1):
        product = mvs[i] * mvs[i + 1]
        even = product.even()
        n = even.norm()
        if n > 1e-12:
            open_rotor = open_rotor * Multivector(even.coeffs / n)

    # Combine: Pancharatnam angle + open-path bivector plane
    half = angle / 2.0
    bv_norm = open_rotor.bivector_norm
    if bv_norm > 1e-12:
        bv = open_rotor.grade(2).coeffs / bv_norm
        combined = np.zeros(8, dtype=np.float64)
        combined[0] = math.cos(half)
        combined[4:7] = bv[4:7] * math.sin(half)
        rotor = Multivector(combined)
    else:
        rotor = Multivector(np.array([
            math.cos(half), 0.0, 0.0, 0.0,
            math.sin(half), 0.0, 0.0, 0.0,
        ], dtype=np.float64))

    return angle, curv, rotor


# Backward-compat alias
compute_curvature = compute_encounter


# ── Fitness: projections of the encounter onto scalars ───────────────────
#
# The encounter rotor is the truth. These functions collapse it to
# numbers for evolutionary selection. Each is a lossy projection.

def compute_loss_trajectory_curvature(trajectory):
    """Variance of consecutive diffs in the loss trajectory."""
    if len(trajectory) < 2:
        return 0.0
    diffs = [trajectory[i + 1] - trajectory[i]
             for i in range(len(trajectory) - 1)]
    if not diffs:
        return 0.0
    m = sum(diffs) / len(diffs)
    return sum((d - m) ** 2 for d in diffs) / len(diffs)


def compute_fitness(external_texts, self_texts, loss_history,
                    embed_fn=None, alpha=0.85):
    """Composite fitness — three projections of encounter quality."""
    if embed_fn is None:
        embed_fn = default_embed_fn

    # Curvature: mean encounter magnitude
    all_texts = (external_texts or []) + (self_texts or [])
    curvatures = []
    for text in all_texts:
        if len(text.split()) >= 5:
            _, curv, _ = compute_encounter(text, embed_fn)
            curvatures.append(curv)
    mean_curv = sum(curvatures) / len(curvatures) if curvatures else 0.0
    norm_curv = min(mean_curv / 0.3, 1.0)

    # Coupling: does the encounter rotor diverge more on external text?
    def _run_memory(texts):
        mem = Multivector.scalar(0.0)
        for text in texts:
            _, curv, rotor = compute_encounter(text, embed_fn)
            mem = mem * alpha + rotor * max(curv, 0.01)
        return mem.norm()

    mag_ext = _run_memory(external_texts) if external_texts else 0.0
    mag_self = _run_memory(self_texts) if self_texts else 0.0
    divergence = mag_ext - mag_self
    norm_div = 1.0 / (1.0 + math.exp(-divergence * 5))

    # Loss improvement
    loss_imp = 0.0
    if loss_history and len(loss_history) >= 2:
        final_losses = [e['losses'][-1] for e in loss_history
                        if e.get('losses')]
        if len(final_losses) >= 2:
            n = len(final_losses)
            x_mean = (n - 1) / 2.0
            y_mean = sum(final_losses) / n
            num = sum((i - x_mean) * (final_losses[i] - y_mean)
                      for i in range(n))
            den = sum((i - x_mean) ** 2 for i in range(n))
            if den > 1e-12:
                loss_imp = -(num / den)
    norm_loss = max(min(loss_imp, 1.0), -1.0)
    norm_loss = (norm_loss + 1.0) / 2.0

    fitness = 0.5 * norm_curv + 0.3 * norm_div + 0.2 * norm_loss

    return {
        'fitness': round(fitness, 6),
        'curvature': round(mean_curv, 6),
        'divergence': round(divergence, 6),
        'loss_improvement': round(loss_imp, 6),
        'breakdown': {
            'curvature_weighted': round(0.5 * norm_curv, 6),
            'divergence_weighted': round(0.3 * norm_div, 6),
            'loss_improvement_weighted': round(0.2 * norm_loss, 6),
        },
    }


def compute_prediction_fitness(fm_loss, self_loss, curvature, learning_rate):
    """Fitness from live FM prediction."""
    norm_curv = min(abs(curvature) / 0.3, 1.0)
    gap = fm_loss - self_loss if (fm_loss is not None and
                                  self_loss is not None) else 0.0
    norm_gap = 1.0 / (1.0 + math.exp(-gap * 2))
    norm_loss = max(0.0, min(1.0, 1.0 - (fm_loss - 2.0) / 6.0)
                    ) if fm_loss else 0.5
    norm_lr = min(max(learning_rate, 0.0), 1.0)
    return round(0.5 * norm_curv + 0.2 * norm_gap
                 + 0.15 * norm_loss + 0.15 * norm_lr, 6)


def improvement_at_k(initial_fitness, archive, k=50):
    """imp@k: best gain within k generations."""
    within_k = [v for v in archive if v.get('generation', 0) < k]
    if not within_k:
        return 0.0
    return max(v.get('fitness', 0.0) for v in within_k) - initial_fitness


# ═══════════════════════════════════════════════════════════════════════════
# §5  Robust statistics
# ═══════════════════════════════════════════════════════════════════════════

def _quantiles(values):
    if not values:
        return {"median": 0.0, "p10": 0.0, "p90": 0.0, "max": 0.0}
    vals = sorted(values)
    def q(p):
        idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * p)))
        return float(vals[idx])
    return {"median": q(0.5), "p10": q(0.1), "p90": q(0.9), "max": q(1.0)}


def _pairwise_win_rate(a_vals, b_vals):
    if not a_vals or not b_vals:
        return 0.5
    n = min(len(a_vals), len(b_vals))
    wins = sum(1 for i in range(n) if a_vals[i] > b_vals[i])
    return wins / n


# ═══════════════════════════════════════════════════════════════════════════
# §6  Data records
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# §7  The Field
# ═══════════════════════════════════════════════════════════════════════════

class Field:
    """Within-breath sensing: generation, prediction, geometry, frames.

    The breath happens here. Surprise is measured, frames compete,
    and geometry is computed — all in one place.
    """

    def __init__(self, task_agent, embed_fn: Callable | None = None):
        self.task_agent = task_agent
        self.embed_fn = embed_fn or default_embed_fn

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
            angle, curv, rotor = compute_encounter(chunk_text, self.embed_fn)
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
                (pred.score if pred else 0.0)
                - (geom.score if geom else 0.0), 4),
            "predictive_vs_relational": round(
                (pred.score if pred else 0.0)
                - (rel.score if rel else 0.0), 4),
            "geometric_vs_relational": round(
                (geom.score if geom else 0.0)
                - (rel.score if rel else 0.0), 4),
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
        """One breath with multi-frame proprioception.

        Returns a BreathRecord, or None if FM is unavailable.
        """
        if not fm_available():
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
            chunk_text = fm_complete(
                None, messages=messages,
                max_tokens=max_tokens_per_chunk,
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

            rotor_dict = geometric.payload.get(
                "rotor", Multivector.scalar(1.0).as_dict())

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
        angle, curv, holonomy_rotor = compute_encounter(
            full_text, self.embed_fn)
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
        """Proprioceptive vs plain — the honest test."""
        if not fm_available():
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
                    'mean_surprise': result.robust_summary[
                        "trajectory"]["median"],
                    'loss_trajectory_curvature':
                        result.loss_trajectory_curvature,
                    'bivector_norm': Multivector(
                        np.array(list(
                            result.holonomy.values()))).bivector_norm,
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
                "pairwise_win_rate": round(
                    _pairwise_win_rate(w_vals, wo_vals), 4),
            }

        return {
            "prompt": prompt, "n": n,
            "with_proprioception": {
                "runs": with_results, "summary": with_summary},
            "without_proprioception": {
                "runs": without_results, "summary": without_summary},
            "comparison": comparison,
        }

    def _run_plain_breath(self, prompt, chunk_size=50, max_chunks=8,
                          system_prompt=None, temperature=0.7):
        """Plain breath — control condition."""
        if not fm_available():
            return None

        max_tokens_per_chunk = max(chunk_size // 3, 10)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        trajectory = []
        full_text_parts = []

        for chunk_num in range(1, max_chunks + 1):
            chunk_text = fm_complete(
                None, messages=messages,
                max_tokens=max_tokens_per_chunk,
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
        _, curv, _ = compute_encounter(full_text, self.embed_fn)
        ltc = compute_loss_trajectory_curvature(trajectory)

        return {
            'curvature': round(curv, 6),
            'mean_surprise': _quantiles(trajectory)["median"],
            'loss_trajectory_curvature': round(ltc, 6),
            'bivector_norm': 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# §8  Organism data structures
# ═══════════════════════════════════════════════════════════════════════════

# ── Default rulebook ─────────────────────────────────────────────────────
#
# Four distress signals. One flourishing signal.
# The creature must not only respond to pain.

DEFAULT_RULES = [
    {
        "id": "loss_trending_up",
        "condition": "loss_trend == 'increasing'",
        "action": "learn_steps",
        "direction": "increase",
        "magnitude": 2,
        "max_value": 20,
        "enabled": True,
        "rationale": "loss trending up, more steps to memorize recent input"
    },
    {
        "id": "curvature_dropping",
        "condition": "curvature_trend == 'decreasing'",
        "action": "alpha",
        "direction": "decrease",
        "magnitude": 0.05,
        "min_value": 0.5,
        "enabled": True,
        "rationale": "curvature dropping, lower alpha makes memory more responsive"
    },
    {
        "id": "self_recursion_flatline",
        "condition": "self_breath_ratio > 0.5 and curvature_median < 0.05",
        "action": "temperature",
        "direction": "increase",
        "magnitude": 0.2,
        "max_value": 2.0,
        "enabled": True,
        "rationale": "high self-recursion + low curvature, increase diversity"
    },
    {
        "id": "collapse_warnings",
        "condition": "collapse_count > 2",
        "action": "learn_lr",
        "direction": "multiply",
        "magnitude": 0.5,
        "min_value": 0.001,
        "enabled": True,
        "rationale": "collapse warnings, stabilize"
    },
    {
        "id": "tension_rich",
        "condition": "recent_tension_count > 5",
        "action": "temperature",
        "direction": "decrease",
        "magnitude": 0.1,
        "min_value": 0.3,
        "enabled": True,
        "rationale": "many frame disagreements, cool down to let them resolve"
    },
    {
        "id": "rotor_coherent",
        "condition": "rotor_coherence > 0.8 and curvature_median > 0.02",
        "action": "learn_lr",
        "direction": "multiply",
        "magnitude": 1.2,
        "max_value": 0.05,
        "enabled": True,
        "rationale": "stable encounter direction with real curvature — "
                     "the organism is in a groove, lean into it"
    },
]


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class TensionMemory:
    """A remembered disagreement between frames — fertile or dead."""
    frame_a: str
    frame_b: str
    context: str
    usefulness: float
    note: str


@dataclass
class OrganismState:
    """Everything the creature carries across breaths."""
    generation: int = 0
    rulebook: list[dict[str, Any]] = dc_field(
        default_factory=lambda: copy.deepcopy(DEFAULT_RULES))
    mutation_log: list[str] = dc_field(default_factory=list)
    performance_history: list[dict[str, Any]] = dc_field(default_factory=list)
    persistent_memory: dict[str, Any] = dc_field(default_factory=dict)
    tensions: list[TensionMemory] = dc_field(default_factory=list)
    frame_portfolio: dict[str, Any] = dc_field(default_factory=lambda: {
        "active_frames": ["predictive", "geometric", "relational"],
        "weights": {"predictive": 1.0, "geometric": 1.0, "relational": 1.0},
    })
    # Encounter rotors from recent breaths — the geometric self-model
    recent_rotors: list[list[float]] = dc_field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# §9  Breath analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_breaths(breath_log_path):
    """Read a JSONL breath log and compute trends."""
    path = Path(breath_log_path)
    empty = {
        'n_breaths': 0,
        'loss_trend': 'no_data',
        'curvature_trend': 'no_data',
        'mean_curvature': 0.0,
        'curvature_median': 0.0,
        'mean_loss': 0.0,
        'collapse_count': 0,
        'self_breath_ratio': 0.0,
        'recent_tension_count': 0,
        'rotor_coherence': 0.0,
        'recent_breaths': [],
    }

    if not path.exists():
        return empty

    breaths = []
    for line in path.read_text().strip().split('\n'):
        line = line.strip()
        if line:
            try:
                breaths.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not breaths:
        return empty

    losses = [b.get('mean_surprise', 0.0) for b in breaths]
    curvatures = [b.get('curvature', 0.0) for b in breaths]
    collapse_count = sum(1 for b in breaths if b.get('collapse_warning'))
    self_count = sum(1 for b in breaths if not b.get('external', True))

    sorted_curv = sorted(curvatures)
    curv_median = sorted_curv[len(sorted_curv) // 2] if sorted_curv else 0.0

    return {
        'n_breaths': len(breaths),
        'loss_trend': _compute_trend(losses),
        'curvature_trend': _compute_trend(curvatures),
        'mean_curvature': sum(curvatures) / len(curvatures)
                          if curvatures else 0.0,
        'curvature_median': curv_median,
        'mean_loss': sum(losses) / len(losses) if losses else 0.0,
        'collapse_count': collapse_count,
        'self_breath_ratio': self_count / len(breaths)
                             if breaths else 0.0,
        'recent_tension_count': 0,  # updated by Organism
        'rotor_coherence': 0.0,     # updated by Organism
        'recent_breaths': breaths[-5:],
    }


def _compute_trend(values, window=5):
    if len(values) < 2:
        return 'no_data'
    recent = values[-window:]
    if len(recent) < 2:
        return 'no_data'
    n = len(recent)
    x_mean = (n - 1) / 2.0
    y_mean = sum(recent) / n
    num = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den < 1e-12:
        return 'stable'
    slope = num / den
    if abs(slope) < 0.001:
        return 'stable'
    return 'increasing' if slope > 0 else 'decreasing'


# ═══════════════════════════════════════════════════════════════════════════
# §10  The Organism
# ═══════════════════════════════════════════════════════════════════════════

class Organism:
    """The durable creature: memory, rules, mutation, tension, geometry.

    The meta-agent IS the organism. There is no separate controller.
    The thing that proposes changes is the same thing that remembers
    which contradictions were fruitful. Self-modification and memory
    are the same organ.
    """

    def __init__(self, state: OrganismState | None = None):
        self.state = state or OrganismState()

    # ── Rotor self-model ─────────────────────────────────────────────────

    def absorb_rotor(self, rotor: Multivector):
        """Record an encounter rotor. Keep the last 20."""
        self.state.recent_rotors.append(rotor.coeffs.tolist())
        if len(self.state.recent_rotors) > 20:
            self.state.recent_rotors = self.state.recent_rotors[-20:]

    def rotor_coherence(self):
        """How stable is the bivector direction across recent encounters?

        Returns a float in [0, 1]. High coherence means the organism
        keeps encountering text that curves through the same semantic
        plane — it's in a groove, not flailing. This is the first
        positive-valence signal.
        """
        rotors = self.state.recent_rotors
        if len(rotors) < 3:
            return 0.0

        # Extract bivector directions from recent rotors
        directions = []
        for coeffs in rotors[-10:]:
            bv = np.array(coeffs[4:7], dtype=np.float64)
            n = np.linalg.norm(bv)
            if n > 1e-12:
                directions.append(bv / n)

        if len(directions) < 3:
            return 0.0

        # Coherence = mean pairwise |dot product| of unit bivectors
        # 1.0 = all pointing the same way, 0.0 = random
        total = 0.0
        count = 0
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                total += abs(float(np.dot(directions[i], directions[j])))
                count += 1

        return total / count if count > 0 else 0.0

    # ── Breath ingestion ─────────────────────────────────────────────────

    def ingest_breath(self, breath_record):
        """Absorb a BreathRecord into the organism's history."""
        metrics = {
            "generation": self.state.generation,
            "curvature": breath_record.curvature,
            "loss_trajectory_curvature":
                breath_record.loss_trajectory_curvature,
            "trajectory_median":
                breath_record.robust_summary["trajectory"]["median"],
            "trajectory_p90":
                breath_record.robust_summary["trajectory"]["p90"],
            "holonomy": breath_record.holonomy,
            "timestamp": time.time(),
        }
        self.state.performance_history.append(metrics)
        self._remember_tensions(breath_record)

        # Absorb the encounter rotor
        holonomy_coeffs = [
            breath_record.holonomy.get(k, 0.0)
            for k in ("scalar", "e1", "e2", "e3",
                       "e12", "e13", "e23", "e123")
        ]
        self.absorb_rotor(Multivector(np.array(holonomy_coeffs)))

    def _remember_tensions(self, breath_record):
        """Preserve frame disagreements that are large enough to matter."""
        for d in breath_record.disagreement_trace:
            for key, pair_name in (
                ("predictive_vs_geometric", "predictive-geometric"),
                ("predictive_vs_relational", "predictive-relational"),
                ("geometric_vs_relational", "geometric-relational"),
            ):
                val = abs(d.get(key, 0.0))
                if val > 0.15:
                    self.state.tensions.append(TensionMemory(
                        frame_a=pair_name.split("-")[0],
                        frame_b=pair_name.split("-")[1],
                        context=breath_record.prompt[:200],
                        usefulness=val,
                        note="large disagreement preserved",
                    ))

    # ── Summarization ────────────────────────────────────────────────────

    def summarize_for_mutation(self):
        recent = self.state.performance_history[-5:]
        if not recent:
            return {
                "curvature_median": 0.0, "trajectory_median": 0.0,
                "trajectory_p90": 0.0, "recent_tension_count": 0,
                "rotor_coherence": self.rotor_coherence(),
            }
        curvs = sorted(r["curvature"] for r in recent)
        trajs = sorted(r["trajectory_median"] for r in recent)
        return {
            "curvature_median": curvs[len(curvs) // 2],
            "trajectory_median": trajs[len(trajs) // 2],
            "trajectory_p90": trajs[min(len(trajs) - 1,
                                        round((len(trajs) - 1) * 0.9))],
            "recent_tension_count": len(self.state.tensions[-10:]),
            "rotor_coherence": self.rotor_coherence(),
        }

    # ── Rule evaluation + variant proposal ───────────────────────────────

    def _eval_condition(self, expr, analysis):
        try:
            return bool(eval(expr, {"__builtins__": {}}, analysis))
        except Exception:
            return False

    def propose_variant(self, analysis, current_config):
        """Apply rules to produce a modified config."""
        config = dict(current_config)
        config.setdefault('learn_steps', 5)
        config.setdefault('learn_lr', 0.01)
        config.setdefault('temperature', 1.0)
        config.setdefault('alpha', 0.85)

        enriched = dict(analysis)
        enriched['recent_tension_count'] = len(self.state.tensions[-10:])
        enriched['rotor_coherence'] = self.rotor_coherence()

        rationale = []
        active_rules = []

        for rule in self.state.rulebook:
            if not rule.get('enabled', True):
                continue
            if self._eval_condition(rule['condition'], enriched):
                change = self._apply_rule(rule, config)
                if change:
                    rationale.append(change)
                    active_rules.append(rule['id'])

        if not rationale:
            rationale.append("no changes — metrics within normal range")

        config['rationale'] = rationale
        config['active_rules'] = active_rules
        return config

    def propose_variant_with_fm(self, analysis, current_config):
        """FM-powered meta-reasoning. Falls back to heuristic."""
        if not fm_available():
            return self.propose_variant(analysis, current_config)

        rules_text = json.dumps(
            [r for r in self.state.rulebook if r.get('enabled', True)],
            indent=2)

        tension_summary = ""
        recent_tensions = self.state.tensions[-10:]
        if recent_tensions:
            tension_summary = (
                f"\n## Recent tensions ({len(recent_tensions)})\n")
            for t in recent_tensions[-5:]:
                tension_summary += (
                    f"  {t.frame_a} vs {t.frame_b}: "
                    f"usefulness={t.usefulness:.3f}, {t.note}\n")

        coherence = self.rotor_coherence()
        rotor_summary = (
            f"\n## Rotor coherence: {coherence:.3f}"
            f" ({'stable groove' if coherence > 0.8 else 'exploring' if coherence > 0.4 else 'no pattern yet'})\n"
        )

        prompt = (
            "You are the meta-agent for a DGM-H creature.\n"
            "Your job: analyze performance and propose config changes.\n\n"
            f"## Current config\n```json\n{json.dumps(current_config, indent=2)}\n```\n\n"
            f"## Breath analysis\n```json\n{json.dumps(analysis, indent=2, default=str)}\n```\n\n"
            f"## Persistent memory\n{self.summarize_memory() or '(empty)'}\n"
            f"{tension_summary}{rotor_summary}\n"
            f"## Rulebook\n```json\n{rules_text}\n```\n\n"
            "Respond with ONLY a JSON object:\n"
            "{\n"
            '  "config_changes": {"param_name": new_value, ...},\n'
            '  "rationale": "why these changes",\n'
            '  "memory_entry": "optional insight to remember",\n'
            '  "rule_mutations": [{"id": "rule_id", "field": "magnitude", '
            '"new_value": 0.1}]\n'
            "}\n\n"
            "Parameters: learn_steps (int 1-20), learn_lr (float 0.001-0.1), "
            "temperature (float 0.1-2.5), alpha (float 0.5-1.0).\n"
            "Only propose changes that address issues visible in the analysis."
        )

        system = (
            "You are a meta-agent optimizing a tiny prediction model. "
            "Be conservative — small changes. Respond with valid JSON only."
        )

        response = fm_complete(prompt, system=system,
                               max_tokens=512, temperature=0.3)

        if response is None:
            return self.propose_variant(analysis, current_config)

        try:
            json_str = response
            if "```" in json_str:
                for part in json_str.split("```"):
                    stripped = part.strip()
                    if stripped.startswith("json"):
                        stripped = stripped[4:].strip()
                    if stripped.startswith("{"):
                        json_str = stripped
                        break
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return self.propose_variant(analysis, current_config)

        config = dict(current_config)
        config.setdefault('learn_steps', 5)
        config.setdefault('learn_lr', 0.01)
        config.setdefault('temperature', 1.0)
        config.setdefault('alpha', 0.85)

        rationale = []
        clamps = {
            'learn_steps': (1, 20, int),
            'learn_lr': (0.001, 0.1, float),
            'temperature': (0.1, 2.5, float),
            'alpha': (0.5, 1.0, float),
        }

        for param, new_val in parsed.get('config_changes', {}).items():
            if param not in clamps:
                continue
            lo, hi, typ = clamps[param]
            try:
                new_val = typ(new_val)
            except (TypeError, ValueError):
                continue
            new_val = max(lo, min(hi, new_val))
            old_val = config.get(param)
            if old_val is not None and new_val != old_val:
                config[param] = new_val
                rationale.append(f"[FM] {param} {old_val} -> {new_val}")

        fm_rationale = parsed.get('rationale', '')
        if fm_rationale:
            rationale.append(f"[FM reasoning] {fm_rationale}")

        memory_entry = parsed.get('memory_entry')
        if memory_entry:
            self.record('fm_insight', memory_entry)

        for mutation in parsed.get('rule_mutations', []):
            rule_id = mutation.get('id')
            field = mutation.get('field')
            new_value = mutation.get('new_value')
            if not all([rule_id, field, new_value]):
                continue
            for rule in self.state.rulebook:
                if rule['id'] == rule_id and field in rule:
                    old_value = rule[field]
                    rule[field] = new_value
                    desc = (f"[FM] rule '{rule_id}': "
                            f"{field} {old_value} -> {new_value}")
                    rationale.append(desc)
                    self.state.mutation_log.append(desc)

        if not rationale:
            rationale.append("[FM] no changes proposed")

        config['rationale'] = rationale
        config['active_rules'] = ['fm_meta_agent']
        return config

    # ── Rule application ─────────────────────────────────────────────────

    @staticmethod
    def _apply_rule(rule, config):
        param = rule['action']
        direction = rule['direction']
        magnitude = rule['magnitude']
        old = config.get(param)
        if old is None:
            return None

        if direction == 'increase':
            new = old + magnitude
        elif direction == 'decrease':
            new = old - magnitude
        elif direction == 'multiply':
            new = old * magnitude
        else:
            return None

        if 'max_value' in rule:
            new = min(new, rule['max_value'])
        if 'min_value' in rule:
            new = max(new, rule['min_value'])
        if isinstance(new, float):
            new = round(new, 6)
        if new == old:
            return None

        config[param] = new
        return f"{param} {old} -> {new}: {rule.get('rationale', rule['id'])}"

    # ── Rule mutation ────────────────────────────────────────────────────

    def mutate_rules(self, rule_outcomes_fn=None):
        """Modify the rules themselves based on performance history."""
        mutations = []

        for rule in self.state.rulebook:
            if not rule.get('enabled', True):
                continue

            if rule_outcomes_fn:
                outcomes = rule_outcomes_fn(rule['id'])
            else:
                outcomes = self._assess_rule(rule['id'])

            if len(outcomes) < 3:
                continue

            mean_delta = sum(outcomes) / len(outcomes)
            recent_mean = sum(outcomes[-3:]) / 3

            if recent_mean < -0.01 and mean_delta < 0:
                old_mag = rule['magnitude']
                if isinstance(old_mag, (int, float)) and old_mag > 0:
                    new_mag = round(old_mag * 0.5, 6)
                    if isinstance(old_mag, int):
                        new_mag = max(int(new_mag), 1)
                    else:
                        new_mag = max(new_mag, 0.001)
                    if new_mag != old_mag:
                        rule['magnitude'] = new_mag
                        desc = (f"weakened '{rule['id']}': "
                                f"magnitude {old_mag} -> {new_mag}")
                        mutations.append(desc)

            elif recent_mean > 0.01 and mean_delta > 0:
                old_mag = rule['magnitude']
                if isinstance(old_mag, (int, float)):
                    new_mag = round(old_mag * 1.5, 6)
                    if isinstance(old_mag, int):
                        new_mag = min(int(new_mag), 10)
                    else:
                        new_mag = min(new_mag, 1.0)
                    if new_mag != old_mag:
                        rule['magnitude'] = new_mag
                        desc = (f"strengthened '{rule['id']}': "
                                f"magnitude {old_mag} -> {new_mag}")
                        mutations.append(desc)

        if mutations:
            self.state.mutation_log.extend(mutations)
            self.record('last_rule_mutation', {
                'mutations': mutations,
                'n_rules': len(self.state.rulebook),
            })

        return mutations

    def _assess_rule(self, rule_id):
        deltas = []
        for entry in self.state.performance_history:
            meta = entry.get('metadata', {}) or {}
            active_rules = meta.get('active_rules', [])
            parent_fitness = meta.get('parent_fitness')
            if rule_id in active_rules and parent_fitness is not None:
                deltas.append(entry.get('fitness', 0) - parent_fitness)
        return deltas

    # ── Persistent memory ────────────────────────────────────────────────

    def record(self, key, value):
        self.state.persistent_memory[key] = {
            'value': value, 'timestamp': time.time()}

    def recall(self, key=None):
        if key is None:
            return dict(self.state.persistent_memory)
        entry = self.state.persistent_memory.get(key)
        return entry.get('value') if entry else None

    def summarize_memory(self):
        if not self.state.persistent_memory:
            return ""
        sorted_keys = sorted(
            self.state.persistent_memory.keys(),
            key=lambda k: self.state.persistent_memory[k].get(
                'timestamp', 0),
            reverse=True)
        lines = [f"Persistent memory "
                 f"({len(self.state.persistent_memory)} insights):"]
        for key in sorted_keys:
            val = str(self.state.persistent_memory[key].get('value', ''))
            if len(val) > 200:
                val = val[:200] + '...'
            lines.append(f"  - {key}: {val}")
        return '\n'.join(lines)

    # ── Performance tracking ─────────────────────────────────────────────

    def record_generation(self, generation_id, fitness, config,
                          metadata=None):
        entry = {
            'generation': generation_id, 'fitness': fitness,
            'config': config, 'timestamp': time.time(),
        }
        if metadata:
            entry['metadata'] = metadata
        self.state.performance_history.append(entry)

    def get_improvement_trend(self, window=5):
        if len(self.state.performance_history) < 2:
            return 0.0
        recent = self.state.performance_history[-window:]
        fitnesses = [e.get('fitness', 0) for e in recent
                     if isinstance(e.get('fitness'), (int, float))]
        if len(fitnesses) < 2:
            return 0.0
        n = len(fitnesses)
        x_mean = (n - 1) / 2.0
        y_mean = sum(fitnesses) / n
        num = sum((i - x_mean) * (fitnesses[i] - y_mean)
                  for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den < 1e-12:
            return 0.0
        return num / den

    def get_statistics(self):
        history = [e for e in self.state.performance_history
                   if isinstance(e.get('fitness'), (int, float))]
        if not history:
            return {
                'best': 0.0, 'worst': 0.0, 'average': 0.0,
                'total_generations': 0, 'trend': 0.0,
                'best_generation': None,
            }
        fitnesses = [e['fitness'] for e in history]
        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        return {
            'best': max(fitnesses),
            'worst': min(fitnesses),
            'average': sum(fitnesses) / len(fitnesses),
            'total_generations': len(history),
            'trend': self.get_improvement_trend(),
            'best_generation': history[best_idx].get('generation'),
        }

    def get_best_config(self):
        history = [e for e in self.state.performance_history
                   if isinstance(e.get('fitness'), (int, float))]
        if not history:
            return None
        return max(history, key=lambda e: e['fitness']).get('config')

    def get_rule_outcomes(self, rule_id):
        return self._assess_rule(rule_id)

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def rules(self):
        return self.state.rulebook

    @property
    def memory(self):
        return self

    def get_rules(self):
        return copy.deepcopy(self.state.rulebook)

    def summarize_for_meta_agent(self):
        return self.summarize_memory()

    # ── Serialization ────────────────────────────────────────────────────

    def save(self, path):
        payload = {
            "generation": self.state.generation,
            "rulebook": self.state.rulebook,
            "mutation_log": self.state.mutation_log,
            "performance_history": self.state.performance_history,
            "persistent_memory": self.state.persistent_memory,
            "tensions": [t.__dict__ for t in self.state.tensions],
            "frame_portfolio": self.state.frame_portfolio,
            "recent_rotors": self.state.recent_rotors,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))

    @classmethod
    def load(cls, path, **kwargs):
        data = json.loads(Path(path).read_text())
        state = OrganismState()
        state.generation = data.get("generation", 0)
        state.rulebook = data.get("rulebook",
                                   data.get("rules",
                                            copy.deepcopy(DEFAULT_RULES)))
        state.mutation_log = data.get("mutation_log", [])
        state.performance_history = data.get("performance_history", [])
        state.persistent_memory = data.get("persistent_memory", {})
        state.frame_portfolio = data.get("frame_portfolio",
                                          state.frame_portfolio)
        state.tensions = [
            TensionMemory(**t) for t in data.get("tensions", [])]
        state.recent_rotors = data.get("recent_rotors", [])
        return cls(state=state)

    def export_state(self):
        return {
            "generation": self.state.generation,
            "rulebook": self.state.rulebook,
            "mutation_log": self.state.mutation_log,
            "performance_history": self.state.performance_history,
            "persistent_memory": self.state.persistent_memory,
            "tensions": [t.__dict__ for t in self.state.tensions],
            "frame_portfolio": self.state.frame_portfolio,
            "recent_rotors": self.state.recent_rotors,
        }

    def import_state(self, payload):
        self.state.generation = payload.get("generation", 0)
        self.state.rulebook = payload.get("rulebook",
                                           copy.deepcopy(DEFAULT_RULES))
        self.state.mutation_log = payload.get("mutation_log", [])
        self.state.performance_history = payload.get(
            "performance_history", [])
        self.state.persistent_memory = payload.get(
            "persistent_memory", {})
        self.state.frame_portfolio = payload.get(
            "frame_portfolio", self.state.frame_portfolio)
        self.state.tensions = [
            TensionMemory(**t) for t in payload.get("tensions", [])]
        self.state.recent_rotors = payload.get("recent_rotors", [])


# ═══════════════════════════════════════════════════════════════════════════
# §11  Free propose_variant function (used by evolve.py as fallback)
# ═══════════════════════════════════════════════════════════════════════════

def propose_variant(analysis, current_config):
    return Organism().propose_variant(analysis, current_config)
