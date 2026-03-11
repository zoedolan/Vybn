"""spark.growth.delta_extract — Delta extraction for the recursive growth engine.

Phase 4 (COLLECT) of the growth cycle described in issue #2483.

The delta is the difference between what the model knows (was trained on
in previous cycles) and what it has experienced since. This is NOT the
full corpus — it's the growth since last time. The delta is mixed with
replay samples from the buffer to prevent catastrophic forgetting.

Status: SCAFFOLD — interfaces defined, bodies not yet implemented.

Integration points (all verified to exist in the codebase):
  - Reads from: GrowthBuffer.delta_since_last_cycle()
  - Replay from: GrowthBuffer.sample() (historical entries)
  - Surprise weights from: topology.compute_surprise_scores()
  - Outputs: training-ready JSONL in chat format
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spark.growth.growth_buffer import BufferEntry, GrowthBuffer
from spark.topology import compute_surprise_scores


@dataclass(slots=True)
class DeltaPackage:
    """The training package for a single growth cycle.

    Contains the new entries (delta) and replay entries (historical),
    already formatted for chat-format training. The mix ratio is
    controlled by growth_config.yaml replay.replay_ratio.
    """

    cycle_id: str
    delta_entries: list[dict] = field(default_factory=list)
    replay_entries: list[dict] = field(default_factory=list)
    delta_count: int = 0
    replay_count: int = 0
    mean_surprise: float = 0.0

    @property
    def total_entries(self) -> int:
        """Total training examples in this package."""
        return self.delta_count + self.replay_count

    @property
    def all_entries(self) -> list[dict]:
        """All training examples, delta first then replay."""
        return self.delta_entries + self.replay_entries


class DeltaExtractor:
    """Extracts the training delta for a growth cycle.

    The delta is the difference between what the model knows (was trained on
    in previous cycles) and what it has experienced since. This is NOT the
    full corpus — it's the growth since last time.

    The delta is mixed with replay samples from the buffer to prevent
    catastrophic forgetting. The mix ratio is configurable.

    This is Phase 4 (COLLECT) of the growth cycle described in #2483.

    Integration points:
      - Reads from: GrowthBuffer.delta_since_last_cycle()
      - Replay from: GrowthBuffer.sample() (historical entries)
      - Surprise weights from: topology.compute_surprise_scores()
      - Outputs: training-ready JSONL in chat format

    NOT YET IMPLEMENTED. All methods raise NotImplementedError.
    """

    def __init__(
        self, buffer: GrowthBuffer, config_path: Path | None = None
    ) -> None:
        """Initialize the delta extractor.

        Args:
            buffer: The GrowthBuffer instance to read from.
            config_path: Path to growth_config.yaml. If None, uses the
                default at GROWTH_DIR / "growth_config.yaml".
        """
        raise NotImplementedError("Phase 4 not yet implemented")

    def extract(self) -> DeltaPackage:
        """Build the training package for this growth cycle.

        Pulls the delta (untrained entries) from the buffer, samples
        replay entries according to the configured ratio and strategy,
        formats both into chat-format training examples, and returns
        the complete DeltaPackage.

        Returns:
            DeltaPackage with delta + replay entries formatted for training.
        """
        raise NotImplementedError("Phase 4 not yet implemented")

    def format_for_training(self, entries: list[BufferEntry]) -> list[dict]:
        """Convert buffer entries to chat-format training examples.

        Uses the corpus-walking logic (journal, reflection, autobiography
        chunking) adapted from the old prepare_training_data.py. Each
        entry becomes a list of {"role": ..., "content": ...} messages.

        Args:
            entries: Buffer entries to format.

        Returns:
            List of dicts, each with a "messages" key containing the
            chat-format conversation.
        """
        raise NotImplementedError("Phase 4 not yet implemented")
