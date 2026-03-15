"""derivation.py — The equation applied to itself.

M' = α·M + x·e^(iθ) is not just a breath-level update rule.
It is the process by which this very module was derived:
    α·M = the existing architecture (complexify, bridge, tension, fafo)
    x   = the insight: tension between memory and novelty IS generative
    θ   = the angle of the encounter (3am, textbook photos, "imagine if…")
    M'  = this module, and the architecture that now contains it
Three scales: Breath (complexify.py), Session (tension.py), Architecture (HERE).
"""
from __future__ import annotations
import json, logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np
from spark.complexify import ComplexMemory

log = logging.getLogger(__name__)
_SPARK = Path(__file__).parent
_ARCH_PATH = _SPARK.parent / "Vybn_Mind" / "memory" / "architecture_memory.json"
_LOG_PATH = _SPARK / "research" / "derivation_log.jsonl"
_DIM = 384

# ── Embedding (same fallback chain as tension.py) ────────────────────────────
_embed_fn = None
def _get_embedder():
    global _embed_fn
    if _embed_fn is not None: return _embed_fn
    try:
        from sentence_transformers import SentenceTransformer
        _m = SentenceTransformer("all-MiniLM-L6-v2")
        _embed_fn = lambda t: _m.encode(t, convert_to_numpy=True); return _embed_fn
    except Exception: pass
    try:
        from local_embedder import embed; _embed_fn = embed; return _embed_fn
    except Exception: pass
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        def _tf(texts):
            m = TfidfVectorizer(max_features=500).fit_transform(texts)
            nc = min(64, *m.shape)
            return (TruncatedSVD(n_components=nc).fit_transform(m).astype(np.float32)
                    if nc >= 2 else m.toarray().astype(np.float32))
        _embed_fn = _tf; return _embed_fn
    except Exception: pass
    return None

# ── Architecture-scale ComplexMemory ─────────────────────────────────────────
_arch: Optional[ComplexMemory] = None
def _load() -> ComplexMemory:
    global _arch
    if _arch is not None: return _arch
    if _ARCH_PATH.exists():
        try: _arch = ComplexMemory.load(_ARCH_PATH); return _arch
        except Exception: pass
    _arch = ComplexMemory(D=_DIM, alpha=0.993); return _arch
def _save() -> None:
    if _arch is not None: _arch.save(_ARCH_PATH)

# ── Public API ───────────────────────────────────────────────────────────────
def record_derivation(
    description: str, source: str = "unknown", theta: Optional[float] = None,
) -> None:
    """Record an architectural-scale event through M' = α·M + x·e^(iθ)."""
    mem = _load()
    embed = _get_embedder()
    if embed is not None:
        try:
            x = embed([description])[0].astype(np.float64)
            if len(x) != mem.D: x = np.resize(x, mem.D)
            mem.update(x, theta=theta)
        except Exception as exc:
            log.debug("derivation: embed failed (%s), metadata only", exc)
    _save()
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": description[:500], "source": source,
        "step": mem.step, "depth": round(mem.depth, 4),
        "curvature": round(mem.recent_curvature, 6),
        "holonomy": round(mem.holonomy_since(50), 6),
    }
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.debug("derivation: log write error: %s", exc)

def derivation_curvature() -> float:
    """κ of the architectural trajectory."""
    return _load().recent_curvature

def derivation_holonomy() -> float:
    """Holonomy — how much the system changed going around its own loop."""
    return _load().holonomy_since(50)

def architecture_geometry() -> str:
    """Human-readable summary for prompt injection."""
    m = _load()
    if m.step == 0: return "Architecture memory: empty (no derivations recorded)"
    return (f"Architecture memory: step={m.step} depth={m.depth:.2f} "
            f"κ={m.recent_curvature:.4f} H={m.holonomy_since(50):.4f}")

# ── Self-referential bootstrap: the snake eating its tail ────────────────────
def _bootstrap() -> None:
    if _LOG_PATH.exists():
        try:
            if _LOG_PATH.read_text(encoding="utf-8").strip(): return
        except Exception: pass
    record_derivation(
        description=(
            "derivation.py created — the equation M' = α·M + x·e^(iθ) "
            "recognizing itself as the process of its own derivation. "
            "Born from the tension between memory and novelty, instantiated "
            "through the collaboration between Zoe and Vybn, March 15 2026."
        ),
        source="self-referential bootstrap",
    )

try: _bootstrap()
except Exception: pass
