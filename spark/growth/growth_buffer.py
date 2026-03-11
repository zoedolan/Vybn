"""spark.growth.growth_buffer — Experience buffer for the recursive growth engine.

Phase 3 (REMEMBER) of the growth cycle described in issue #2483.

The buffer subscribes to MEDIUM-tier promotions from nested_memory.py,
filters through self_model.py verification, and maintains a rolling
buffer with surprise-weighted sampling for training.

Status: SCAFFOLD — interfaces defined, bodies not yet implemented.

Integration points (all verified to exist in the codebase):
  - Reads from: NestedMemory.consolidate_fast_to_medium() outputs
  - Filters via: self_model.curate_for_training()
  - Surprise scores from: topology.compute_surprise_scores()
  - Persists to: GROWTH_DIR / "buffer.jsonl"
  - Tracks trained set in: GROWTH_DIR / "trained_manifest.json"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from spark.nested_memory import NestedEntry, NestedMemory
from spark.paths import GROWTH_DIR
from spark.self_model import curate_for_training
from spark.topology import compute_surprise_scores


@dataclass(slots=True)
class BufferEntry:
    """A single entry in the growth buffer.

    Wraps a NestedEntry with growth-cycle metadata: which cycle (if any)
    trained on it, when it was ingested, and its current surprise score.
    """

    entry_id: str
    content: str
    source: str
    surprise_score: float
    ingested_at: str  # ISO-8601
    trained_in_cycle: Optional[str] = None
    nested_entry_scale: str = "MEDIUM"
    metadata: dict = field(default_factory=dict)


class GrowthBuffer:
    """Experience buffer for the recursive growth engine.

    Subscribes to MEDIUM-tier promotions from nested_memory.py,
    filters through self_model.py verification, and maintains a
    rolling buffer with surprise-weighted sampling for training.

    This is Phase 3 (REMEMBER) of the growth cycle described in #2483.

    Integration points:
      - Reads from: NestedMemory.consolidate_fast_to_medium() outputs
      - Filters via: self_model.curate_for_training()
      - Surprise scores from: topology.compute_surprise_scores()
      - Persists to: GROWTH_DIR / "buffer.jsonl"
      - Tracks trained set in: GROWTH_DIR / "trained_manifest.json"

    NOT YET IMPLEMENTED. All methods raise NotImplementedError.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the growth buffer.

        Args:
            config_path: Path to growth_config.yaml. If None, uses the
                default at GROWTH_DIR / "growth_config.yaml".
        """
        raise NotImplementedError("Phase 3 not yet implemented")

    def ingest(self, entry: NestedEntry) -> bool:
        """Ingest a promoted MEDIUM-tier entry into the buffer.

        Runs the entry through self_model.curate_for_training() to decide
        whether it should be deposited. Computes surprise score via
        topology.compute_surprise_scores(). Appends to buffer.jsonl.

        Args:
            entry: A NestedEntry that has been promoted to MEDIUM tier.

        Returns:
            True if accepted (passed self_model curation), False otherwise.
        """
        raise NotImplementedError("Phase 3 not yet implemented")

    def sample(self, n: int, strategy: str = "surprise") -> list[BufferEntry]:
        """Sample n entries from the buffer for replay during training.

        Args:
            n: Number of entries to sample.
            strategy: "surprise" (NLL-weighted, higher surprise = more
                likely to be sampled) or "uniform" (equal probability).

        Returns:
            List of BufferEntry objects sampled from the buffer.
        """
        raise NotImplementedError("Phase 3 not yet implemented")

    def delta_since_last_cycle(self) -> list[BufferEntry]:
        """Return entries added since the last completed growth cycle.

        These are the entries that have NOT been included in any training
        cycle yet — the growth delta.

        Returns:
            List of untrained BufferEntry objects.
        """
        raise NotImplementedError("Phase 3 not yet implemented")

    def mark_trained(self, entry_ids: list[str], cycle_id: str) -> None:
        """Mark entries as included in a completed training cycle.

        Updates the trained_manifest.json and sets trained_in_cycle on
        each buffer entry.

        Args:
            entry_ids: IDs of entries that were trained on.
            cycle_id: Identifier for the growth cycle.
        """
        raise NotImplementedError("Phase 3 not yet implemented")

    def stats(self) -> dict:
        """Buffer statistics: size, untrained count, mean surprise, etc.

        Returns:
            Dict with keys: total_entries, untrained_count,
            mean_surprise, oldest_entry, newest_entry.
        """
        raise NotImplementedError("Phase 3 not yet implemented")
