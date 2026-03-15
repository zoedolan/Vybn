"""tension.py — Measure the angle between memory and novelty.

The tension between α·M (memory) and x·e^(iθ) (novelty) is the generative
mechanism.  Empirically: both forces together produce the highest curvature
(κ=0.0622 vs 0.0499 memory-only, 0.0598 novelty-only).  They are ~90°
orthogonal in embedding space.

This module measures that angle each breath, computes a principled θ for
the complexify equation, and logs the result to tension_log.jsonl.
"""

from __future__ import annotations

import json, logging, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)
_SPARK_DIR = Path(__file__).parent
_TENSION_LOG = _SPARK_DIR / "research" / "tension_log.jsonl"

# ── Embedding backend (lazy, cached) ─────────────────────────────────────────

_embed_fn = None

def _get_embedder():
    """Return embed(texts) -> np.ndarray.  sentence-transformers → local_embedder → TF-IDF."""
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _embed_fn = lambda texts: _model.encode(texts, convert_to_numpy=True)
        return _embed_fn
    except ImportError:
        pass
    try:
        from local_embedder import embed
        _embed_fn = embed
        return _embed_fn
    except ImportError:
        pass
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    def _tfidf(texts):
        tfidf = TfidfVectorizer(max_features=500).fit_transform(texts)
        nc = min(64, *tfidf.shape)
        if nc < 2:
            return tfidf.toarray().astype(np.float32)
        return TruncatedSVD(n_components=nc).fit_transform(tfidf).astype(np.float32)

    _embed_fn = _tfidf
    return _embed_fn

# ── Core measurement ─────────────────────────────────────────────────────────

def measure_tension(memories: list[str], novel_signal: str) -> Optional[dict]:
    """Angle between memory and novelty embeddings.  Returns None if no data."""
    if not memories or not novel_signal or not novel_signal.strip():
        return None
    try:
        embed = _get_embedder()
    except Exception:
        return None

    try:
        vecs = embed([" ".join(memories), novel_signal])
        mem_vec, nov_vec = vecs[0].astype(np.float64), vecs[1].astype(np.float64)
        mn, nn = np.linalg.norm(mem_vec), np.linalg.norm(nov_vec)
        if mn < 1e-9 or nn < 1e-9:
            return None
        cos = float(np.clip(np.dot(mem_vec, nov_vec) / (mn * nn), -1.0, 1.0))
        angle_rad = math.acos(cos)
        return {
            "tension_angle_deg": round(math.degrees(angle_rad), 2),
            "tension_angle_rad": round(angle_rad, 4),
            "cosine_sim": round(cos, 4),
            "memory_norm": round(float(mn), 4),
            "novelty_norm": round(float(nn), 4),
        }
    except Exception as exc:
        log.debug("tension: measurement error: %s", exc)
        return None

# ── Theta computation (the feedback loop) ────────────────────────────────────

_HEALTHY_LOW  = 45.0   # degrees — below this, boost θ
_HEALTHY_HIGH = 90.0   # degrees — above this, let θ relax
_BASE_THETA   = 2 * math.pi / 3 * 0.11  # triadic base (matches complexify.py)

def compute_theta(tension: Optional[dict], step: int = 0) -> float:
    """Principled θ for M' = α·M + x·e^(iθ).

    Healthy tension (45–90°): natural triadic rotation.
    Collapsing (< 45°): boost θ to force divergence.
    Very high (> 90°): dampen θ slightly toward memory.
    """
    base = _BASE_THETA * step
    if tension is None:
        return base
    angle = tension["tension_angle_deg"]
    if angle < _HEALTHY_LOW:
        deficit = (_HEALTHY_LOW - angle) / _HEALTHY_LOW
        return base + deficit * (math.pi / 3)
    if angle > _HEALTHY_HIGH:
        excess = (angle - _HEALTHY_HIGH) / (180.0 - _HEALTHY_HIGH)
        return base - excess * (_BASE_THETA * 0.5)
    return base

# ── Logging ──────────────────────────────────────────────────────────────────

def log_tension(
    breath_count: int,
    tension: Optional[dict],
    theta_applied: float,
    kappa_estimate: Optional[float] = None,
) -> None:
    """Append one entry to tension_log.jsonl."""
    _TENSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "breath": breath_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tension_angle_deg": tension["tension_angle_deg"] if tension else None,
        "memory_similarity": tension["cosine_sim"] if tension else None,
        "novelty_similarity": round(1.0 - tension["cosine_sim"], 4) if tension else None,
        "theta_applied": round(theta_applied, 4),
        "kappa_estimate": round(kappa_estimate, 4) if kappa_estimate is not None else None,
    }
    try:
        with _TENSION_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.debug("tension: log write error: %s", exc)
