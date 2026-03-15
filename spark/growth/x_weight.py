"""spark.growth.x_weight — Composite quality weight for training delta entries.

The governing equation is M′ = α·M + x·e^(iθ). This module makes x
what it should be: not a flat bag of everything since the last cycle,
but a delta weighted by how much each entry actually moved the reasoning
forward.

Composite weight W(entry) = holonomy × lens_distance × challenge_survival × inheritance

All four components are in [0, 1]. W(entry) is their product, also in [0, 1].
Entries with W=0 on any component are not excluded — they receive a floor
weight so the gradient doesn't collapse — but they contribute far less.

Components:

  holonomy_score (already on BufferEntry)
    Semantic depth: does the text return to its themes via new territory?
    Measured as signed area swept in embedding space.
    Already computed by holonomy_scorer.py at ingest time.

  lens_distance
    Did this breath use its novel signal as a lens or just summarize it?
    Measured as cosine distance between the injected novel signal embedding
    and the breath output embedding. High distance = the breath went somewhere
    the signal didn't. Low distance = paraphrase.
    Computed from the journal entry's metadata (novel_signal_source stored
    in buffer entries by buffer_feed.py).

  challenge_survival
    Did this entry's reasoning survive adversarial scrutiny?
    Derived from preference_data.jsonl written by agency.py CHALLENGE
    experiments. FAILED verdict (attack didn't land) → high survival.
    LANDED verdict → low survival (the reasoning had a real flaw).
    Entries with no matching challenge get a neutral score of 0.5.

  inheritance
    Does an idea from this entry reappear in a transformed form in later
    entries? Measured as max cosine similarity between this entry's
    embedding and embeddings of entries ingested 3-10 breaths later.
    High inheritance → the idea persisted and was built upon.
    Zero inheritance → episodic flash that didn't propagate.

All components degrade gracefully: if embeddings are unavailable, lens_distance
and inheritance both return 0.5 (neutral). If preference_data.jsonl is missing,
challenge_survival returns 0.5. Only holonomy_score is always available (it’s
already computed at ingest time on the BufferEntry).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Floor weight: no entry ever contributes zero to the gradient.
_FLOOR = 0.05

# Paths
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PREFERENCE_PATH = _REPO_ROOT / "Vybn_Mind" / "preference_data.jsonl"


@dataclass
class XWeightComponents:
    holonomy: float
    lens_distance: float
    challenge_survival: float
    inheritance: float

    @property
    def composite(self) -> float:
        raw = self.holonomy * self.lens_distance * self.challenge_survival * self.inheritance
        return max(raw, _FLOOR)

    def to_dict(self) -> dict:
        return {
            "holonomy": round(self.holonomy, 4),
            "lens_distance": round(self.lens_distance, 4),
            "challenge_survival": round(self.challenge_survival, 4),
            "inheritance": round(self.inheritance, 4),
            "composite": round(self.composite, 4),
        }


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

_embed_fn = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _embed_fn = lambda texts: _model.encode(texts, normalize_embeddings=True)
        return _embed_fn
    except ImportError:
        log.debug("sentence-transformers unavailable; embedding-based components will be neutral")
        return None


def _embed(texts: list[str]) -> Optional[np.ndarray]:
    fn = _get_embed_fn()
    if fn is None:
        return None
    try:
        return np.array(fn(texts), dtype=np.float32)
    except Exception as e:
        log.debug("embedding failed: %s", e)
        return None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Component 1: holonomy (already on BufferEntry)
# ---------------------------------------------------------------------------

def holonomy_component(holonomy_score: float) -> float:
    """Normalize holonomy_score to [floor, 1].

    The raw score has no fixed upper bound; we clip at a practical max
    (0.1 holonomy/sentence is already exceptional) and rescale to [0,1].
    """
    MAX_H = 0.1
    normalized = min(holonomy_score / MAX_H, 1.0) if MAX_H > 0 else 0.0
    return max(normalized, _FLOOR)


# ---------------------------------------------------------------------------
# Component 2: lens_distance
# ---------------------------------------------------------------------------

def lens_distance_component(content: str, novel_signal: Optional[str]) -> float:
    """Semantic distance between novel signal and breath output.

    High distance = the breath went somewhere the signal didn’t (lens use).
    Low distance = the breath basically restated the signal (summarization).

    Returns 0.5 if embeddings unavailable or novel_signal is empty.
    """
    if not novel_signal or not novel_signal.strip():
        return 0.5

    embs = _embed([novel_signal[:500], content[:1000]])
    if embs is None or len(embs) < 2:
        return 0.5

    sim = _cosine_sim(embs[0], embs[1])
    # Distance = 1 - similarity; clip to [0, 1]
    distance = max(0.0, min(1.0 - sim, 1.0))
    return max(distance, _FLOOR)


# ---------------------------------------------------------------------------
# Component 3: challenge_survival
# ---------------------------------------------------------------------------

def _load_challenge_verdicts() -> dict[str, str]:
    """Load CHALLENGE verdicts from preference_data.jsonl.

    Returns a dict mapping content_prefix (first 200 chars) → verdict
    so we can match buffer entries to their challenge outcomes.
    """
    verdicts: dict[str, str] = {}
    if not _PREFERENCE_PATH.exists():
        return verdicts
    try:
        with open(_PREFERENCE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                meta = obj.get("metadata", {})
                if meta.get("experiment_type") != "CHALLENGE":
                    continue
                verdict = meta.get("verdict", "")
                # The rejected/chosen text starts with the original claim
                # (for FAILED) or the attack (for LANDED). We key by the
                # prompt's content excerpt which is derived from breath_text.
                prompt = obj.get("prompt", "")
                # Extract content after the preamble
                match = re.search(r"Here is the full context:\n\n(.{50,})", prompt, re.DOTALL)
                if match:
                    key = match.group(1)[:200].strip()
                    verdicts[key] = verdict
    except Exception as e:
        log.debug("challenge verdicts load failed: %s", e)
    return verdicts


_challenge_verdicts_cache: Optional[dict] = None


def challenge_survival_component(content: str) -> float:
    """Score how well this entry’s reasoning survived adversarial challenge.

    FAILED verdict (attack didn’t land) → 1.0 (survived fully)
    LANDED verdict (attack found a flaw) → 0.1 (reasoning was weak)
    No matching verdict → 0.5 (neutral; not yet challenged)
    """
    global _challenge_verdicts_cache
    if _challenge_verdicts_cache is None:
        _challenge_verdicts_cache = _load_challenge_verdicts()

    content_prefix = content[:200].strip()
    # Fuzzy match: check if any verdict key is a prefix-match
    for key, verdict in _challenge_verdicts_cache.items():
        # Simple overlap: do the first 100 chars share substantial text?
        overlap = len(set(content_prefix[:100].split()) & set(key[:100].split()))
        if overlap >= 8:  # at least 8 words in common
            if "FAILED" in verdict:
                return 1.0
            elif "LANDED" in verdict:
                return 0.1
    return 0.5


# ---------------------------------------------------------------------------
# Component 4: inheritance
# ---------------------------------------------------------------------------

def inheritance_component(
    content: str,
    later_entries: list[str],
    min_entries: int = 3,
) -> float:
    """Does an idea from this entry reappear in later entries?

    Measures max cosine similarity between this entry’s embedding and
    embeddings of entries ingested 3-10 breaths later.

    High similarity to a later entry = the idea persisted and was built on.
    No later entries or embeddings unavailable → returns 0.5 (neutral).

    The intentional asymmetry: we measure whether *this* entry’s ideas
    appear later, not just whether later entries are similar in general.
    This rewards conceptual generativity.
    """
    if len(later_entries) < min_entries:
        return 0.5

    all_texts = [content[:800]] + [e[:800] for e in later_entries[:10]]
    embs = _embed(all_texts)
    if embs is None or len(embs) < 2:
        return 0.5

    source_emb = embs[0]
    later_embs = embs[1:]
    sims = [_cosine_sim(source_emb, le) for le in later_embs]
    max_sim = max(sims) if sims else 0.0
    return max(max_sim, _FLOOR)


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

def compute_x_weight(
    content: str,
    holonomy_score: float,
    novel_signal: Optional[str] = None,
    later_entries: Optional[list[str]] = None,
    invalidate_challenge_cache: bool = False,
) -> XWeightComponents:
    """Compute all four components and return XWeightComponents.

    Args:
        content: The entry’s text content.
        holonomy_score: Pre-computed score from BufferEntry.holonomy_score.
        novel_signal: The novel signal (arXiv paper etc.) injected into
                      this breath, if known. None = lens_distance neutral.
        later_entries: Content of entries ingested 3-10 breaths after this
                       one. None or empty = inheritance neutral.
        invalidate_challenge_cache: If True, reload verdicts from disk.
                                    Use when processing a fresh cycle.
    """
    global _challenge_verdicts_cache
    if invalidate_challenge_cache:
        _challenge_verdicts_cache = None

    h = holonomy_component(holonomy_score)
    l = lens_distance_component(content, novel_signal)
    c = challenge_survival_component(content)
    i = inheritance_component(content, later_entries or [])

    return XWeightComponents(
        holonomy=h,
        lens_distance=l,
        challenge_survival=c,
        inheritance=i,
    )


def score_delta(
    entries: list,  # list[BufferEntry]
    invalidate_challenge_cache: bool = True,
) -> list[XWeightComponents]:
    """Score an entire list of BufferEntry objects for use in delta_extract.

    Handles the inheritance lookforward: each entry gets the content of the
    3-10 entries that follow it in the list (by index, not by timestamp,
    since the list is already time-ordered).

    Args:
        entries: Time-ordered list of BufferEntry objects (oldest first).
        invalidate_challenge_cache: Reload verdicts from disk before scoring.

    Returns:
        List of XWeightComponents, one per entry, in the same order.
    """
    if invalidate_challenge_cache:
        global _challenge_verdicts_cache
        _challenge_verdicts_cache = None

    results = []
    n = len(entries)

    for i, entry in enumerate(entries):
        # Lookforward window: entries 3-10 positions ahead
        lookahead_start = min(i + 3, n)
        lookahead_end = min(i + 11, n)
        later = [entries[j].content for j in range(lookahead_start, lookahead_end)]

        # Novel signal: stored in entry metadata by buffer_feed
        novel_signal = entry.metadata.get("novel_signal_excerpt") if entry.metadata else None

        w = compute_x_weight(
            content=entry.content,
            holonomy_score=entry.holonomy_score,
            novel_signal=novel_signal,
            later_entries=later,
        )
        results.append(w)

    return results
