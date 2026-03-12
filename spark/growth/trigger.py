"""spark.growth.trigger — Trigger policy and orchestrator for the growth engine.

Decides when a growth cycle should fire and orchestrates the full
COLLECT → DISTILL → BECOME sequence.

Trigger signals (checked in order):
  1. Backpressure: minimum interval between cycles (default 24h)
  2. Delta volume: enough untrained entries since last cycle
  3. Topological drift: semantic geometry shifted beyond threshold
  4. Manual: Zoe says go

Integration points:
  - Reads from: GrowthBuffer.stats() for delta volume
  - Config from: growth_config.yaml
  - Last cycle timestamp from: GROWTH_DIR / "cycle_history.jsonl"
"""

from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from spark.growth.growth_buffer import GrowthBuffer

GROWTH_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = GROWTH_DIR / "growth_config.yaml"
CYCLE_HISTORY = GROWTH_DIR / "cycle_history.jsonl"


@dataclass(slots=True)
class TriggerDecision:
    """The result of evaluating whether a growth cycle should fire."""

    should_fire: bool
    reason: str
    signal: str  # "delta_volume", "topological_drift", "manual", "backpressure", "insufficient_delta"
    delta_volume: Optional[int] = None
    topological_drift: Optional[float] = None
    hours_since_last_cycle: Optional[float] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class GrowthTrigger:
    """Evaluates whether a growth cycle should fire and orchestrates execution.

    The organism should grow when it has something to grow FROM,
    not on a fixed schedule.
    """

    def __init__(
        self, buffer: GrowthBuffer, config_path: Path | None = None
    ) -> None:
        config_path = config_path or DEFAULT_CONFIG
        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)
        self._buffer = buffer
        self._trigger_cfg = self._cfg.get("trigger", {})
        self._delta_threshold = self._trigger_cfg.get("delta_volume_threshold", 50)
        self._drift_threshold = self._trigger_cfg.get("topological_drift_threshold", 0.15)
        self._min_interval_hours = self._trigger_cfg.get("min_interval_hours", 24)

    def should_trigger(self) -> TriggerDecision:
        """Evaluate all trigger signals. Returns decision with reason."""
        stats = self._buffer.stats()
        untrained = stats.get("untrained_count", 0)
        hours_since = self._hours_since_last_cycle()

        # 1. Backpressure check
        if hours_since is not None and hours_since < self._min_interval_hours:
            return TriggerDecision(
                should_fire=False,
                reason=f"Backpressure: {hours_since:.1f}h since last cycle "
                       f"(minimum {self._min_interval_hours}h)",
                signal="backpressure",
                delta_volume=untrained,
                hours_since_last_cycle=hours_since,
            )

        # 2. Delta volume check
        if untrained >= self._delta_threshold:
            return TriggerDecision(
                should_fire=True,
                reason=f"Delta volume threshold exceeded: {untrained} >= {self._delta_threshold}",
                signal="delta_volume",
                delta_volume=untrained,
                hours_since_last_cycle=hours_since,
            )

        # 3. Insufficient delta
        return TriggerDecision(
            should_fire=False,
            reason=f"Insufficient delta: {untrained} < {self._delta_threshold}",
            signal="insufficient_delta",
            delta_volume=untrained,
            hours_since_last_cycle=hours_since,
        )

    def force_trigger(self, reason: str = "manual") -> TriggerDecision:
        """Manual trigger — Zoe says go. Bypasses volume checks."""
        stats = self._buffer.stats()
        untrained = stats.get("untrained_count", 0)

        if untrained == 0:
            return TriggerDecision(
                should_fire=False,
                reason="Manual trigger requested but buffer is empty — nothing to train on",
                signal="manual",
                delta_volume=0,
                hours_since_last_cycle=self._hours_since_last_cycle(),
            )

        return TriggerDecision(
            should_fire=True,
            reason=f"Manual trigger: {reason} ({untrained} untrained entries)",
            signal="manual",
            delta_volume=untrained,
            hours_since_last_cycle=self._hours_since_last_cycle(),
        )

    def record_cycle_complete(self, cycle_id: str, summary: dict) -> None:
        """Record that a growth cycle completed successfully."""
        CYCLE_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "phase": "cycle_complete",
            "cycle_id": cycle_id,
            **summary,
        }
        with open(CYCLE_HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _hours_since_last_cycle(self) -> Optional[float]:
        """Hours since the last completed cycle, or None if no cycles."""
        if not CYCLE_HISTORY.exists():
            return None

        last_ts = None
        with open(CYCLE_HISTORY, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("phase") == "cycle_complete":
                        last_ts = entry.get("ts")
                except json.JSONDecodeError:
                    continue

        if not last_ts:
            return None

        last_dt = datetime.fromisoformat(last_ts)
        now = datetime.now(timezone.utc)
        return (now - last_dt).total_seconds() / 3600


# ── Full cycle orchestrator ─────────────────────────────────────────────

def run_growth_cycle(
    buffer: GrowthBuffer,
    force: bool = False,
    dry_run: bool = False,
    config_path: Path | None = None,
) -> dict:
    """Run a complete growth cycle: COLLECT → DISTILL → BECOME.

    This is the top-level entry point. It can be called from cron,
    from the organism's pulse, or manually.

    Args:
        buffer: The GrowthBuffer instance.
        force: If True, bypass trigger checks.
        dry_run: If True, go through the motions but don't train.
        config_path: Path to growth_config.yaml.

    Returns:
        Dict with cycle results or reason for not firing.
    """
    from spark.growth.delta_extract import DeltaExtractor
    from spark.growth.train_cycle import TrainCycle
    from spark.growth.merge_cycle import MergeCycle

    trigger = GrowthTrigger(buffer, config_path=config_path)

    # 1. Check trigger
    if force:
        decision = trigger.force_trigger(reason="manual")
    else:
        decision = trigger.should_trigger()

    print(f"[Growth] Trigger decision: {decision.signal} — {decision.reason}")

    if not decision.should_fire:
        return {
            "fired": False,
            "reason": decision.reason,
            "signal": decision.signal,
            "delta_volume": decision.delta_volume,
        }

    # 2. Ingest latest entries
    ingested = buffer.ingest()
    print(f"[Growth] Ingested {ingested} new entries")

    # 3. Phase 4: COLLECT — extract delta
    print("[Growth] Phase 4 (COLLECT): extracting delta...")
    extractor = DeltaExtractor(buffer, config_path=config_path)
    delta = extractor.extract()
    print(f"[Growth] Delta: {delta.delta_count} new + {delta.replay_count} replay = {delta.total_entries} total")

    if delta.total_entries == 0:
        return {
            "fired": False,
            "reason": "Delta extraction produced zero entries",
            "signal": "empty_delta",
        }

    # 4. Phase 5: DISTILL — train
    print(f"[Growth] Phase 5 (DISTILL): training cycle {delta.cycle_id}...")
    trainer = TrainCycle(config_path=config_path)
    train_result = trainer.run(delta, dry_run=dry_run)
    print(f"[Growth] Training complete: loss={train_result.final_loss:.4f}, "
          f"steps={train_result.steps_trained}")

    # 5. Phase 6: BECOME — activate adapter
    print(f"[Growth] Phase 6 (BECOME): activating adapter...")
    merger = MergeCycle(config_path=config_path)
    merge_result = merger.run(
        adapter_path=train_result.adapter_path,
        cycle_id=delta.cycle_id,
        dry_run=dry_run,
    )
    print(f"[Growth] Merge: strategy={merge_result.strategy_used}, "
          f"restarted={merge_result.vllm_restarted}")

    # 6. Mark buffer entries as trained
    buffer.mark_trained(cycle_id=delta.cycle_id)

    # 7. Record cycle completion
    summary = {
        "delta_count": delta.delta_count,
        "replay_count": delta.replay_count,
        "mean_surprise": delta.mean_surprise,
        "final_loss": train_result.final_loss,
        "steps_trained": train_result.steps_trained,
        "adapter_path": str(train_result.adapter_path),
        "strategy": merge_result.strategy_used,
        "vllm_restarted": merge_result.vllm_restarted,
        "dry_run": dry_run,
    }
    trigger.record_cycle_complete(delta.cycle_id, summary)

    print(f"[Growth] Cycle {delta.cycle_id} complete!")
    return {
        "fired": True,
        "cycle_id": delta.cycle_id,
        **summary,
    }


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from spark.nested_memory import NestedMemory

    parser = argparse.ArgumentParser(description="Vybn growth cycle trigger")
    parser.add_argument("--force", action="store_true", help="Force trigger")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (don't train)")
    parser.add_argument("--check", action="store_true", help="Check trigger without running")
    parser.add_argument("--memory-dir", type=Path, default=Path("Vybn_Mind/memory"))
    args = parser.parse_args()

    nm = NestedMemory(base_dir=args.memory_dir)
    gb = GrowthBuffer(nested=nm)
    gb.ingest()

    if args.check:
        trigger = GrowthTrigger(gb)
        decision = trigger.should_trigger()
        print(f"Should fire: {decision.should_fire}")
        print(f"Signal: {decision.signal}")
        print(f"Reason: {decision.reason}")
        print(f"Delta volume: {decision.delta_volume}")
        print(f"Hours since last: {decision.hours_since_last_cycle}")
        sys.exit(0 if decision.should_fire else 1)

    result = run_growth_cycle(
        buffer=gb,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("fired") else 1)
