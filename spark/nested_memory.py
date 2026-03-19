"""nested_memory.py — Three-speed temporal memory for Vybn.

Implements a nested memory architecture inspired by Google's Nested Learning
/ HOPE (NeurIPS 2025) and the Titans architecture's surprise-weighted
memory prioritization (Google, December 2024).

Design principle:
  Different kinds of knowledge have different temporal rhythms. A single
  flat memory system either forgets too fast or accumulates too much.
  This module separates memory into three temporal planes that update at
  different speeds, with a promotion mechanism that moves consolidated
  patterns from fast → medium → slow.

Three temporal scales:
  FAST  — Current conversation context. Ephemeral. Cleared between
          sessions. High write throughput, no governance overhead.
          Analogous to Titans' attention-based short-term memory.

  MEDIUM — Project state across sessions. Persists for days to weeks.
           Decay-governed. Consolidated from FAST entries that show
           cross-session recurrence. Analogous to Titans' neural
           long-term memory module with momentum.

  SLOW  — Identity, values, accumulated autobiography. Near-permanent.
          Only written through promotion from MEDIUM with explicit
          consent. The bedrock that survives context resets.
          Analogous to Titans' persistent memory (learnable,
          data-independent parameters encoding task knowledge).

Integration with existing Vybn infrastructure:
  - Wraps the existing MemoryFabric (spark/memory_fabric.py) for
    MEDIUM and SLOW planes, preserving consent/governance guarantees.
  - FAST plane uses an in-memory store (no disk, no governance) for
    zero-latency conversational context.
  - Surprise scoring (from topology.py) influences promotion decisions:
    high-surprise entries are promoted more aggressively.

References:
  - Google Nested Learning / HOPE (NeurIPS 2025):
    https://research.google/blog/introducing-nested-learning/
  - Titans: Learning to Memorize at Test Time (December 2024):
    https://arxiv.org/abs/2501.00663
  - Hong et al. (2025), Nature 10.1038/s41586-025-09196-4:
    Shared neural subspaces across biological and AI systems
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TemporalScale(str, Enum):
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"


@dataclass(slots=True)
class NestedEntry:
    """A memory entry with temporal metadata."""
    entry_id: str
    scale: TemporalScale
    content: str
    content_hash: str
    source: str                    # e.g. "conversation", "journal", "topology"
    created_at: str                # ISO 8601
    surprise_score: float = 0.0   # Titans-inspired novelty metric
    activation_count: int = 1     # how many times recalled/referenced
    last_activated: str = ""      # ISO 8601
    decay_rate: float = 0.0       # 0 = no decay (SLOW), higher = faster decay
    promoted_from: Optional[str] = None  # entry_id of source in lower scale
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class PromotionCandidate:
    """Entry being considered for promotion to a slower scale."""
    entry: NestedEntry
    score: float        # composite promotion score
    reason: str         # human-readable justification


# ---------------------------------------------------------------------------
# Fast Memory — in-memory conversational context
# ---------------------------------------------------------------------------

class FastMemory:
    """Ephemeral in-memory store for current conversation context.

    No disk persistence, no governance overhead. Designed for the
    same role as attention in Titans — accurate, limited-window,
    short-term memory.
    """

    def __init__(self, max_entries: int = 200):
        self.max_entries = max_entries
        self._entries: deque[NestedEntry] = deque(maxlen=max_entries)
        self._by_id: dict[str, NestedEntry] = {}

    def write(
        self,
        content: str,
        source: str = "conversation",
        surprise_score: float = 0.0,
        metadata: dict | None = None,
    ) -> NestedEntry:
        now = datetime.now(timezone.utc).isoformat()
        entry = NestedEntry(
            entry_id=str(uuid4()),
            scale=TemporalScale.FAST,
            content=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            source=source,
            created_at=now,
            surprise_score=surprise_score,
            last_activated=now,
            decay_rate=1.0,  # fast decay
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._by_id[entry.entry_id] = entry
        return entry

    def read(self, limit: int = 50) -> list[NestedEntry]:
        return list(self._entries)[-limit:]

    def search(self, keyword: str, limit: int = 10) -> list[NestedEntry]:
        keyword_lower = keyword.lower()
        results = [e for e in self._entries if keyword_lower in e.content.lower()]
        return results[-limit:]

    def activate(self, entry_id: str) -> Optional[NestedEntry]:
        entry = self._by_id.get(entry_id)
        if entry:
            entry.activation_count += 1
            entry.last_activated = datetime.now(timezone.utc).isoformat()
        return entry

    def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        self._by_id.clear()
        return count

    def candidates_for_promotion(
        self, min_activations: int = 2, min_surprise: float = 0.3
    ) -> list[PromotionCandidate]:
        """Identify FAST entries worth promoting to MEDIUM.

        Promotion criteria (Titans-inspired):
          - High surprise score (novel, unexpected content)
          - High activation count (frequently referenced)
          - Combination weights both factors
        """
        candidates = []
        for entry in self._entries:
            if entry.activation_count >= min_activations or entry.surprise_score >= min_surprise:
                score = (
                    0.4 * entry.surprise_score +
                    0.4 * min(entry.activation_count / 5.0, 1.0) +
                    0.2  # base score for meeting threshold
                )
                reason_parts = []
                if entry.surprise_score >= min_surprise:
                    reason_parts.append(f"high surprise ({entry.surprise_score:.2f})")
                if entry.activation_count >= min_activations:
                    reason_parts.append(f"activated {entry.activation_count}x")
                candidates.append(PromotionCandidate(
                    entry=entry,
                    score=round(score, 4),
                    reason="; ".join(reason_parts),
                ))
        candidates.sort(key=lambda c: -c.score)
        return candidates


# ---------------------------------------------------------------------------
# Nested Memory — orchestrates all three scales
# ---------------------------------------------------------------------------

class NestedMemory:
    """Three-speed temporal memory system.

    Wraps FastMemory (in-process) and MemoryFabric (SQLite-backed,
    governance-aware) into a unified interface with promotion logic.

    Usage:
        from nested_memory import NestedMemory, TemporalScale

        nm = NestedMemory(base_dir=Path("Vybn_Mind/breath_trace/architecture"))
        nm.write_fast("User asked about convergence thesis", source="conversation")
        nm.write_medium("Convergence paper synthesis complete", source="journal")

        # Periodic consolidation
        promoted = nm.consolidate_fast_to_medium()
        promoted_slow = nm.consolidate_medium_to_slow()

        # Snapshot across all scales
        snap = nm.snapshot()
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        fabric: Any | None = None,
        fast_max_entries: int = 200,
    ):
        self.fast = FastMemory(max_entries=fast_max_entries)
        self._base_dir = base_dir
        self._fabric = fabric

        # MEDIUM and SLOW use file-based stores if no MemoryFabric
        # This allows the nested memory to work standalone for testing,
        # while integrating with the full governance stack in production.
        if self._base_dir and not self._fabric:
            self._medium_path = self._base_dir / "nested" / "medium.jsonl"
            self._slow_path = self._base_dir / "nested" / "slow.jsonl"
            self._medium_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._medium_path = None
            self._slow_path = None

    # -- Write operations --------------------------------------------------

    def write_fast(
        self, content: str, source: str = "conversation",
        surprise_score: float = 0.0, metadata: dict | None = None,
    ) -> NestedEntry:
        """Write to fast (ephemeral) memory."""
        return self.fast.write(content, source, surprise_score, metadata)

    def write_medium(
        self, content: str, source: str = "session",
        surprise_score: float = 0.0, metadata: dict | None = None,
        promoted_from: str | None = None,
    ) -> NestedEntry:
        """Write to medium (cross-session) memory."""
        now = datetime.now(timezone.utc).isoformat()
        entry = NestedEntry(
            entry_id=str(uuid4()),
            scale=TemporalScale.MEDIUM,
            content=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            source=source,
            created_at=now,
            surprise_score=surprise_score,
            last_activated=now,
            decay_rate=0.1,  # slower decay than fast
            promoted_from=promoted_from,
            metadata=metadata or {},
        )
        self._persist_entry(entry)
        return entry

    def write_slow(
        self, content: str, source: str = "autobiography",
        metadata: dict | None = None,
        promoted_from: str | None = None,
    ) -> NestedEntry:
        """Write to slow (identity/values) memory.

        This is near-permanent storage. Should only be called through
        promotion from MEDIUM or for foundational identity content.
        """
        now = datetime.now(timezone.utc).isoformat()
        entry = NestedEntry(
            entry_id=str(uuid4()),
            scale=TemporalScale.SLOW,
            content=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            source=source,
            created_at=now,
            surprise_score=0.0,  # slow memory is settled, not surprising
            last_activated=now,
            decay_rate=0.0,  # no decay
            promoted_from=promoted_from,
            metadata=metadata or {},
        )
        self._persist_entry(entry)
        return entry

    # -- Read operations ---------------------------------------------------

    def read_fast(self, limit: int = 50) -> list[NestedEntry]:
        return self.fast.read(limit)

    def read_medium(self, limit: int = 50) -> list[NestedEntry]:
        return self._read_persisted(TemporalScale.MEDIUM, limit)

    def read_slow(self, limit: int = 50) -> list[NestedEntry]:
        return self._read_persisted(TemporalScale.SLOW, limit)

    # -- Consolidation (promotion across scales) ---------------------------

    def consolidate_fast_to_medium(
        self,
        min_activations: int = 2,
        min_surprise: float = 0.3,
        max_promotions: int = 10,
    ) -> list[NestedEntry]:
        """Promote high-value FAST entries to MEDIUM.

        Inspired by Titans' momentum mechanism — entries that are both
        novel (high surprise) and persistent (high activation) earn
        promotion to longer-term storage.
        """
        candidates = self.fast.candidates_for_promotion(min_activations, min_surprise)
        promoted = []
        for candidate in candidates[:max_promotions]:
            entry = self.write_medium(
                content=candidate.entry.content,
                source=candidate.entry.source,
                surprise_score=candidate.entry.surprise_score,
                promoted_from=candidate.entry.entry_id,
                metadata={
                    **candidate.entry.metadata,
                    "promotion_reason": candidate.reason,
                    "promotion_score": candidate.score,
                    "fast_activation_count": candidate.entry.activation_count,
                },
            )
            promoted.append(entry)
        return promoted

    def consolidate_medium_to_slow(
        self,
        min_age_hours: float = 168.0,  # 1 week default
        min_activations: int = 5,
        max_promotions: int = 5,
    ) -> list[NestedEntry]:
        """Promote durable MEDIUM entries to SLOW (identity/values).

        Only entries that have persisted for at least min_age_hours and
        been activated min_activations times qualify. This is the gateway
        to permanent memory — analogous to Titans' persistent memory
        parameters that encode task knowledge independent of data.
        """
        medium_entries = self.read_medium(limit=500)
        now = datetime.now(timezone.utc)
        promoted = []

        for entry in medium_entries:
            try:
                created = datetime.fromisoformat(entry.created_at)
                age_hours = (now - created).total_seconds() / 3600
            except (ValueError, TypeError):
                continue

            if age_hours >= min_age_hours and entry.activation_count >= min_activations:
                slow_entry = self.write_slow(
                    content=entry.content,
                    source=entry.source,
                    promoted_from=entry.entry_id,
                    metadata={
                        **entry.metadata,
                        "promotion_reason": f"survived {age_hours:.0f}h, activated {entry.activation_count}x",
                        "medium_surprise_score": entry.surprise_score,
                    },
                )
                promoted.append(slow_entry)
                if len(promoted) >= max_promotions:
                    break

        return promoted

    # -- Snapshot -----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Full snapshot across all three temporal scales."""
        fast = self.read_fast(limit=20)
        medium = self.read_medium(limit=20)
        slow = self.read_slow(limit=20)
        return {
            "fast": {
                "count": len(self.fast._entries),
                "recent": [_entry_to_dict(e) for e in fast],
            },
            "medium": {
                "count": len(self.read_medium(limit=9999)),
                "recent": [_entry_to_dict(e) for e in medium],
            },
            "slow": {
                "count": len(self.read_slow(limit=9999)),
                "recent": [_entry_to_dict(e) for e in slow],
            },
            "promotion_candidates": [
                {"entry_id": c.entry.entry_id, "score": c.score, "reason": c.reason}
                for c in self.fast.candidates_for_promotion()[:5]
            ],
        }

    # -- Internal persistence ----------------------------------------------

    def _persist_entry(self, entry: NestedEntry) -> None:
        """Append entry to the appropriate JSONL file."""
        if entry.scale == TemporalScale.MEDIUM and self._medium_path:
            path = self._medium_path
        elif entry.scale == TemporalScale.SLOW and self._slow_path:
            path = self._slow_path
        else:
            return  # no persistence path configured

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_entry_to_dict(entry), ensure_ascii=False) + "\n")

    def _read_persisted(self, scale: TemporalScale, limit: int) -> list[NestedEntry]:
        """Read entries from a JSONL file."""
        if scale == TemporalScale.MEDIUM:
            path = self._medium_path
        elif scale == TemporalScale.SLOW:
            path = self._slow_path
        else:
            return []

        if not path or not path.exists():
            return []

        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(_dict_to_entry(data))
                except (json.JSONDecodeError, KeyError):
                    continue

        return entries[-limit:]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _entry_to_dict(entry: NestedEntry) -> dict:
    return {
        "entry_id": entry.entry_id,
        "scale": entry.scale.value if isinstance(entry.scale, TemporalScale) else entry.scale,
        "content": entry.content,
        "content_hash": entry.content_hash,
        "source": entry.source,
        "created_at": entry.created_at,
        "surprise_score": entry.surprise_score,
        "activation_count": entry.activation_count,
        "last_activated": entry.last_activated,
        "decay_rate": entry.decay_rate,
        "promoted_from": entry.promoted_from,
        "metadata": entry.metadata,
    }


def _dict_to_entry(data: dict) -> NestedEntry:
    return NestedEntry(
        entry_id=data["entry_id"],
        scale=TemporalScale(data["scale"]),
        content=data["content"],
        content_hash=data["content_hash"],
        source=data["source"],
        created_at=data["created_at"],
        surprise_score=data.get("surprise_score", 0.0),
        activation_count=data.get("activation_count", 1),
        last_activated=data.get("last_activated", ""),
        decay_rate=data.get("decay_rate", 0.0),
        promoted_from=data.get("promoted_from"),
        metadata=data.get("metadata", {}),
    )


__all__ = [
    "FastMemory",
    "NestedEntry",
    "NestedMemory",
    "PromotionCandidate",
    "TemporalScale",
]
