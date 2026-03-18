"""spark.growth.growth_buffer — Experience buffer for the recursive growth engine.

Phase 3 (REMEMBER) of the growth cycle described in issue #2483.

The buffer pulls recent entries from NestedMemory, filters by surprise
score, and maintains a bounded rolling buffer for downstream training.

Integration points:
  - Reads from: NestedMemory (FAST tier entries written by the breath cycle)
  - Persists to: GROWTH_DIR / "buffer.jsonl"
  - Tracks trained set in: GROWTH_DIR / "trained_manifest.json"
"""

from __future__ import annotations

import json
import logging
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from spark.nested_memory import NestedEntry, NestedMemory

log = logging.getLogger(__name__)

# Default config
_DEFAULT_CFG = {
    "buffer_size": 500,
    "surprise_floor": 0.3,
    # Holonomy curation: weight for holonomy vs surprise in sampling.
    # 1.0 = pure holonomy, 0.0 = pure surprise, 0.5 = equal blend.
    # Validated: rho=1.0 convergence between intrinsic and extrinsic methods,
    # 6-sigma phase transition separating deep from flat texts. (#2507)
    "holonomy_weight": 0.5,
}


@dataclass(slots=True)
class BufferEntry:
    """A single entry in the growth buffer.

    Wraps a NestedEntry with growth-cycle metadata: which cycle (if any)
    trained on it, when it was ingested, its surprise score, and its
    holonomy score (semantic depth — how much the text returns to its
    themes via new territory in embedding space).
    """

    entry_id: str
    content: str
    source: str
    surprise_score: float
    ingested_at: str  # ISO-8601
    trained_in_cycle: Optional[str] = None
    nested_entry_scale: str = "FAST"
    metadata: dict = field(default_factory=dict)
    holonomy_score: float = 0.0  # semantic depth; 0.0 if scoring unavailable


def _buffer_entry_to_dict(entry: BufferEntry) -> dict:
    return {
        "entry_id": entry.entry_id,
        "content": entry.content,
        "source": entry.source,
        "surprise_score": entry.surprise_score,
        "holonomy_score": entry.holonomy_score,
        "ingested_at": entry.ingested_at,
        "trained_in_cycle": entry.trained_in_cycle,
        "nested_entry_scale": entry.nested_entry_scale,
        "metadata": entry.metadata,
    }


def _dict_to_buffer_entry(data: dict) -> BufferEntry:
    return BufferEntry(
        entry_id=data["entry_id"],
        content=data["content"],
        source=data["source"],
        surprise_score=data.get("surprise_score", 0.0),
        holonomy_score=data.get("holonomy_score", 0.0),
        ingested_at=data.get("ingested_at", ""),
        trained_in_cycle=data.get("trained_in_cycle"),
        nested_entry_scale=data.get("nested_entry_scale", "FAST"),
        metadata=data.get("metadata", {}),
    )


def _compute_holonomy_score(content: str) -> float:
    """Score text holonomy; returns 0.0 on any failure.

    Holonomy measures semantic depth: texts that return to their themes
    via new conceptual territory sweep area in embedding space. This is
    the validated signal from the Vybn-Dolan holonomic loss hypothesis
    (rho=1.0 intrinsic/extrinsic convergence, 6-sigma phase transition).
    """
    try:
        from spark.growth.holonomy_scorer import score_text
        report = score_text(content)
        return float(report.score)
    except ImportError:
        log.debug("holonomy_scorer unavailable (sentence-transformers not installed); skipping")
        return 0.0
    except Exception as exc:  # noqa: BLE001
        log.debug("holonomy scoring failed: %s", exc)
        return 0.0


class GrowthBuffer:
    """Experience buffer for the recursive growth engine.

    Pulls recent entries from NestedMemory, filters by surprise score,
    and maintains a bounded rolling buffer with combined surprise + holonomy
    weighted sampling for downstream training.

    This is Phase 3 (REMEMBER) of the growth cycle described in #2483.
    """

    def __init__(
        self,
        nested: NestedMemory,
        cfg: dict | None = None,
        buffer_dir: Path | None = None,
    ) -> None:
        """Initialize the growth buffer.

        Args:
            nested: NestedMemory instance to pull entries from.
            cfg: Config dict with keys: buffer_size, surprise_floor,
                 holonomy_weight (0.0–1.0, default 0.5).
            buffer_dir: Directory for buffer.jsonl and trained_manifest.json.
                        Defaults to spark/growth/.
        """
        self._nested = nested
        self._cfg = {**_DEFAULT_CFG, **(cfg or {})}
        self._buffer_dir = buffer_dir or Path(__file__).resolve().parent
        self._buffer_path = self._buffer_dir / "buffer.jsonl"
        self._manifest_path = self._buffer_dir / "trained_manifest.json"

        # In-memory ring buffer
        self._entries: deque[BufferEntry] = deque(maxlen=self._cfg["buffer_size"])
        self._by_id: dict[str, BufferEntry] = {}
        self._seen_ids: set[str] = set()
        self._last_ingest_ts: str = ""

        # Load any persisted entries
        self._load_persisted()

        # Load trained manifest
        self._trained_manifest: dict = self._load_manifest()

    # -- Core operations ---------------------------------------------------

    def ingest(self) -> int:
        """Pull recent entries from NestedMemory and add to buffer.

        Reads FAST and MEDIUM tier entries from nested memory, filters by
        surprise_score >= surprise_floor, scores holonomy, and adds new
        entries to the bounded buffer.

        Returns:
            Number of new entries ingested.
        """
        floor = self._cfg["surprise_floor"]
        fast_entries = self._nested.read_fast(limit=200)
        medium_entries = self._nested.read_medium(limit=200) if hasattr(self._nested, 'read_medium') else []
        all_entries = fast_entries + medium_entries
        count = 0

        for nested_entry in all_entries:
            if nested_entry.entry_id in self._seen_ids:
                continue
            if nested_entry.surprise_score < floor:
                continue

            now = datetime.now(timezone.utc).isoformat()
            holonomy = _compute_holonomy_score(nested_entry.content)

            buf_entry = BufferEntry(
                entry_id=nested_entry.entry_id,
                content=nested_entry.content,
                source=nested_entry.source,
                surprise_score=nested_entry.surprise_score,
                holonomy_score=holonomy,
                ingested_at=now,
                nested_entry_scale=nested_entry.scale.value if hasattr(nested_entry.scale, 'value') else str(nested_entry.scale),
                metadata=nested_entry.metadata,
            )

            self._entries.append(buf_entry)
            self._by_id[buf_entry.entry_id] = buf_entry
            self._seen_ids.add(buf_entry.entry_id)
            self._persist_entry(buf_entry)
            count += 1

        if count:
            self._last_ingest_ts = datetime.now(timezone.utc).isoformat()

        # Trim _by_id and _seen_ids if buffer evicted old entries
        live_ids = {e.entry_id for e in self._entries}
        stale = set(self._by_id.keys()) - live_ids
        for sid in stale:
            self._by_id.pop(sid, None)

        return count

    def sample(self, n: int, strategy: str = "depth_weighted") -> list[BufferEntry]:
        """Sample n entries from the buffer for replay during training.

        Args:
            n: Number of entries to sample.
            strategy: "depth_weighted" (holonomy_weight * holonomy +
                      (1-holonomy_weight) * surprise, from cfg),
                      "surprise_weighted" (surprise only, legacy),
                      or "uniform" (equal probability).

        Returns:
            List of BufferEntry objects sampled from the buffer.
        """
        if not self._entries:
            return []
        n = min(n, len(self._entries))

        if strategy == "uniform":
            return random.sample(list(self._entries), n)

        entries_list = list(self._entries)

        if strategy == "surprise_weighted":
            weights = [max(e.surprise_score, 0.01) for e in entries_list]
        else:
            # depth_weighted: blend surprise and holonomy per config
            hw = float(self._cfg.get("holonomy_weight", 0.5))
            sw = 1.0 - hw
            weights = [
                max(sw * e.surprise_score + hw * e.holonomy_score, 0.01)
                for e in entries_list
            ]

        sampled = random.choices(entries_list, weights=weights, k=n)
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[BufferEntry] = []
        for e in sampled:
            if e.entry_id not in seen:
                seen.add(e.entry_id)
                result.append(e)
        return result

    def delta_since_last_cycle(self) -> list[BufferEntry]:
        """Return entries added since the last completed growth cycle.

        These are entries NOT yet included in any training cycle —
        the growth delta.

        Returns:
            List of untrained BufferEntry objects.
        """
        return [e for e in self._entries if e.trained_in_cycle is None]

    def mark_trained(self, entry_ids: list[str] | None = None, cycle_id: str | None = None) -> None:
        """Mark entries as included in a completed training cycle.

        Should only be called after a TrainResult is successfully returned
        from TrainCycle.run() — never on ingest or trigger. This ensures
        trained_in_cycle reflects reality: entries are only marked trained
        when training has actually completed.

        Args:
            entry_ids: IDs of entries that were trained on.
                       If None, marks all untrained entries.
            cycle_id: Identifier for the growth cycle.
                      If None, uses current timestamp.
        """
        cycle_id = cycle_id or datetime.now(timezone.utc).isoformat()

        if entry_ids is None:
            entry_ids = [e.entry_id for e in self._entries if e.trained_in_cycle is None]

        for eid in entry_ids:
            buf_entry = self._by_id.get(eid)
            if buf_entry:
                buf_entry.trained_in_cycle = cycle_id

        # Rewrite buffer.jsonl so trained_in_cycle persists across restarts
        self._rewrite_buffer()

        # Update manifest
        self._trained_manifest.setdefault("cycles", []).append({
            "cycle_id": cycle_id,
            "entry_ids": entry_ids,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        self._save_manifest()

    def reset_watermark(self) -> int:
        """Clear trained_in_cycle for all entries, recovering orphaned data.

        Useful when training failed silently and cycle_history.jsonl is
        empty despite entries showing trained_in_cycle=<some cycle id>.
        Returns the number of entries reset.
        """
        count = 0
        for entry in self._entries:
            if entry.trained_in_cycle is not None:
                entry.trained_in_cycle = None
                count += 1
        if count:
            self._rewrite_buffer()
        return count

    def stats(self) -> dict:
        """Buffer statistics.

        Returns:
            Dict with keys: total_entries, untrained_count,
            mean_surprise, mean_holonomy, oldest_entry,
            newest_entry, last_ingest_ts.
        """
        entries = list(self._entries)
        untrained = [e for e in entries if e.trained_in_cycle is None]
        surprises = [e.surprise_score for e in entries]
        holonomies = [e.holonomy_score for e in entries]

        return {
            "total_entries": len(entries),
            "untrained_count": len(untrained),
            "mean_surprise": round(sum(surprises) / len(surprises), 4) if surprises else 0.0,
            "mean_holonomy": round(sum(holonomies) / len(holonomies), 4) if holonomies else 0.0,
            "oldest_entry": entries[0].ingested_at if entries else None,
            "newest_entry": entries[-1].ingested_at if entries else None,
            "last_ingest_ts": self._last_ingest_ts or None,
        }

    # -- Persistence -------------------------------------------------------

    def _persist_entry(self, entry: BufferEntry) -> None:
        self._buffer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._buffer_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_buffer_entry_to_dict(entry), ensure_ascii=False) + "\n")

    def _rewrite_buffer(self) -> None:
        """Rewrite buffer.jsonl atomically to persist in-memory state."""
        tmp = self._buffer_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(_buffer_entry_to_dict(entry), ensure_ascii=False) + "\n")
        tmp.replace(self._buffer_path)

    def _load_persisted(self) -> None:
        if not self._buffer_path.exists():
            return
        with open(self._buffer_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = _dict_to_buffer_entry(data)
                    self._entries.append(entry)
                    self._by_id[entry.entry_id] = entry
                    self._seen_ids.add(entry.entry_id)
                except (json.JSONDecodeError, KeyError):
                    continue

    def _load_manifest(self) -> dict:
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_manifest(self) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(self._trained_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
