"""spark.growth.trigger — Trigger policy for the recursive growth engine.

Decides when a growth cycle should fire. The organism should grow when
it has something to grow FROM, not on a fixed schedule.

Status: SCAFFOLD — interfaces defined, bodies not yet implemented.

Trigger signals (checked in order):
  1. Delta volume: enough new MEDIUM-tier entries since last cycle
  2. Topological drift: semantic geometry shifted beyond threshold
  3. Manual: Zoe says go

Backpressure:
  - Minimum interval between cycles (default 24h)
  - The merge step takes the model offline briefly

Integration points (all verified to exist in the codebase):
  - Reads from: GrowthBuffer.stats() for delta volume
  - Reads from: VybnConnectome.checkpoint() for drift comparison
  - Config from: growth_config.yaml
  - Last cycle timestamp from: GROWTH_DIR / "cycle_history.jsonl"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from spark.connectome.connectome_layer import VybnConnectome
from spark.growth.growth_buffer import GrowthBuffer
from spark.paths import GROWTH_DIR


@dataclass(slots=True)
class TriggerDecision:
    """The result of evaluating whether a growth cycle should fire.

    Captures the decision, the reason, and the signal values that
    informed it.
    """

    should_fire: bool
    reason: str
    signal: str  # "delta_volume", "topological_drift", "manual", "backpressure"
    delta_volume: Optional[int] = None
    topological_drift: Optional[float] = None
    hours_since_last_cycle: Optional[float] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class GrowthTrigger:
    """Evaluates whether a growth cycle should fire.

    The organism should grow when it has something to grow FROM,
    not on a fixed schedule.

    Trigger signals (checked in order):
      1. Delta volume: enough new MEDIUM-tier entries since last cycle
      2. Topological drift: semantic geometry shifted beyond threshold
      3. Manual: Zoe says go

    Backpressure:
      - Minimum interval between cycles (default 24h)
      - The merge step takes the model offline briefly

    Integration points:
      - Reads from: GrowthBuffer.stats() for delta volume
      - Reads from: VybnConnectome.checkpoint() for drift comparison
      - Config from: growth_config.yaml
      - Last cycle timestamp from: GROWTH_DIR / "cycle_history.jsonl"

    NOT YET IMPLEMENTED. All methods raise NotImplementedError.
    """

    def __init__(
        self, buffer: GrowthBuffer, config_path: Path | None = None
    ) -> None:
        """Initialize the growth trigger.

        Args:
            buffer: The GrowthBuffer instance to read stats from.
            config_path: Path to growth_config.yaml. If None, uses the
                default at GROWTH_DIR / "growth_config.yaml".
        """
        raise NotImplementedError("Trigger not yet implemented")

    def should_trigger(self) -> TriggerDecision:
        """Evaluate all trigger signals. Returns decision with reason.

        Checks in order:
          1. Backpressure (minimum interval) — if too soon, deny
          2. Delta volume threshold — if exceeded, fire
          3. Topological drift threshold — if exceeded, fire
          4. Otherwise, no trigger

        Returns:
            TriggerDecision with should_fire and reason.
        """
        raise NotImplementedError("Trigger not yet implemented")

    def force_trigger(self, reason: str = "manual") -> TriggerDecision:
        """Manual trigger (Zoe says go).

        Bypasses delta volume and drift checks, but still respects
        backpressure unless explicitly overridden.

        Args:
            reason: Reason for the manual trigger.

        Returns:
            TriggerDecision with should_fire=True.
        """
        raise NotImplementedError("Trigger not yet implemented")

    def record_cycle_complete(self, cycle_id: str, summary: dict) -> None:
        """Record that a growth cycle completed successfully.

        Appends to GROWTH_DIR / "cycle_history.jsonl" so future
        trigger evaluations know when the last cycle ran.

        Args:
            cycle_id: Unique identifier for the completed cycle.
            summary: Dict with cycle metadata (entries trained, loss, etc).
        """
        raise NotImplementedError("Trigger not yet implemented")

    def _check_delta_volume(self) -> Optional[str]:
        """Check if delta volume threshold is exceeded.

        Returns:
            Reason string if threshold exceeded, None otherwise.
        """
        raise NotImplementedError("Trigger not yet implemented")

    def _check_topological_drift(self) -> Optional[str]:
        """Check if topology has drifted beyond threshold since last cycle.

        Compares current topology snapshot against the snapshot saved
        at the end of the last growth cycle.

        Returns:
            Reason string if drift exceeded, None otherwise.
        """
        raise NotImplementedError("Trigger not yet implemented")

    def _check_backpressure(self) -> bool:
        """Check if minimum interval has elapsed since last cycle.

        Reads the last entry from cycle_history.jsonl and compares
        against min_interval_hours from config.

        Returns:
            True if enough time has elapsed (ok to proceed),
            False if backpressure applies (too soon).
        """
        raise NotImplementedError("Trigger not yet implemented")
