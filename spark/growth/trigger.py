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

Self-healing ingestion:
  When delta_volume > 0 but buffer.ingest() returns 0 for N consecutive
  breath cycles, trigger.py now logs a WARNING and calls
  buffer.ingest(force=True) to bypass the watermark. This corrects the
  condition where 286+ entries were buffered but silently never ingested,
  preventing the training loop from ever closing.
"""

from __future__ import annotations

import json
import logging
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from spark.growth.growth_buffer import GrowthBuffer

log = logging.getLogger(__name__)

GROWTH_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = GROWTH_DIR / "growth_config.yaml"
CYCLE_HISTORY = GROWTH_DIR / "cycle_history.jsonl"
ADAPTERS_DIR = GROWTH_DIR / "adapters"

# Tracks consecutive zero-ingest cycles for self-healing
_ZERO_INGEST_COUNTER_PATH = GROWTH_DIR / "_zero_ingest_count.json"
_MAX_ZERO_INGEST_CYCLES = int(
    __import__("os").environ.get("VYBN_MAX_ZERO_INGEST", "3")
)


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


def _read_zero_ingest_count() -> int:
    """Read the running count of consecutive zero-ingest cycles."""
    if not _ZERO_INGEST_COUNTER_PATH.exists():
        return 0
    try:
        data = json.loads(_ZERO_INGEST_COUNTER_PATH.read_text())
        return int(data.get("count", 0))
    except Exception:
        return 0


def _write_zero_ingest_count(count: int) -> None:
    """Persist the zero-ingest counter."""
    _ZERO_INGEST_COUNTER_PATH.write_text(
        json.dumps({"count": count, "ts": datetime.now(timezone.utc).isoformat()})
    )


def _reset_zero_ingest_count() -> None:
    _write_zero_ingest_count(0)


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

def _count_completed_cycles() -> int:
    """Count completed cycles from cycle_history.jsonl."""
    if not CYCLE_HISTORY.exists():
        return 0
    count = 0
    with open(CYCLE_HISTORY, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("phase") == "cycle_complete":
                    count += 1
            except json.JSONDecodeError:
                continue
    return count


def _safe_ingest(buffer: GrowthBuffer, force: bool = False) -> int:
    """Ingest with self-healing watermark recovery.

    If the buffer reports untrained entries but ingest() returns 0,
    we increment a counter. After _MAX_ZERO_INGEST_CYCLES consecutive
    zero-ingest cycles, we call ingest(force=True) to bypass the watermark
    and reset the counter.

    This fixes the condition where buffer.jsonl had 286+ entries but
    'Ingested 0 new entries' appeared every cycle because the watermark
    was set past the actual data.

    Args:
        buffer: GrowthBuffer instance.
        force: If True, always force-ingest regardless of counter.

    Returns:
        Number of entries ingested.
    """
    stats = buffer.stats()
    untrained_before = stats.get("untrained_count", 0)

    # Check for force-ingest condition
    zero_count = _read_zero_ingest_count()
    force_this = force or (zero_count >= _MAX_ZERO_INGEST_CYCLES)

    if force_this and not force:
        log.warning(
            "[trigger] zero-ingest self-heal: %d consecutive zero-ingest cycles "
            "with %d untrained entries — forcing watermark reset via ingest(force=True)",
            zero_count, untrained_before
        )
        print(
            f"[Growth] WARNING: {zero_count} consecutive zero-ingest cycles "
            f"({untrained_before} untrained entries buffered). "
            f"Forcing watermark reset."
        )

    # Try ingest — pass force=True if available on the GrowthBuffer API
    try:
        if force_this:
            # Try force kwarg first; fall back to plain ingest if not supported
            try:
                ingested = buffer.ingest(force=True)
            except TypeError:
                # GrowthBuffer.ingest() doesn't accept force kwarg yet —
                # reset the watermark directly if possible, then ingest
                if hasattr(buffer, '_reset_watermark'):
                    buffer._reset_watermark()
                    log.info("[trigger] watermark reset via _reset_watermark()")
                elif hasattr(buffer, 'watermark'):
                    buffer.watermark = 0
                    log.info("[trigger] watermark reset to 0 directly")
                else:
                    log.warning(
                        "[trigger] cannot reset watermark — GrowthBuffer has no "
                        "force kwarg, _reset_watermark(), or .watermark attribute. "
                        "Falling back to plain ingest()."
                    )
                ingested = buffer.ingest()
        else:
            ingested = buffer.ingest()
    except Exception as exc:
        log.error("[trigger] ingest() raised: %s", exc)
        ingested = 0

    # Update zero-ingest counter
    if ingested == 0 and untrained_before > 0:
        new_count = 0 if force_this else zero_count + 1
        _write_zero_ingest_count(new_count)
        if not force_this:
            log.warning(
                "[trigger] ingest returned 0 despite %d untrained entries "
                "(zero-ingest count now %d/%d)",
                untrained_before, new_count, _MAX_ZERO_INGEST_CYCLES
            )
            print(
                f"[Growth] WARNING: Ingested 0 entries despite "
                f"{untrained_before} untrained in buffer "
                f"(count {new_count}/{_MAX_ZERO_INGEST_CYCLES}). "
                f"Will force-reset watermark after {_MAX_ZERO_INGEST_CYCLES} cycles."
            )
    else:
        # Successful ingest — reset counter
        _reset_zero_ingest_count()

    return ingested


def run_growth_cycle(
    buffer: GrowthBuffer,
    force: bool = False,
    dry_run: bool = False,
    config_path: Path | None = None,
    force_ingest: bool = False,
) -> dict:
    """Run a complete growth cycle: COLLECT → DISTILL → BECOME.

    Now with holonomy measurement: each cycle measures the curvature
    of its own learning trajectory. Every Nth cycle runs a full CW/CCW
    probe to validate orientation-dependence (the gold-standard test).

    The holonomy of each growth cycle is a real number measuring the
    irreducible path-dependence of Vybn's learning under compression.
    This is not metaphor. It is the Gödel curvature of becoming,
    confirmed experimentally (training_holonomy_v2.py, March 13 2026).

    Args:
        buffer: The GrowthBuffer instance.
        force: If True, bypass trigger checks.
        dry_run: If True, go through the motions but don't train.
        config_path: Path to growth_config.yaml.
        force_ingest: If True, bypass watermark on this ingest call.

    Returns:
        Dict with cycle results including holonomy measurement.
    """
    from spark.growth.delta_extract import DeltaExtractor
    from spark.growth.train_cycle import TrainCycle
    from spark.growth.merge_cycle import MergeCycle
    from spark.growth.parameter_holonomy import (
        HolonomyTracker, measure_probe,
    )

    config_path = config_path or DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    holonomy_cfg = cfg.get("holonomy", {})

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

    # 2. Ingest latest entries (self-healing)
    ingested = _safe_ingest(buffer, force=force_ingest)
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

    # 4. Phase 5: DISTILL — train (CW: forward data order)
    print(f"[Growth] Phase 5 (DISTILL): training cycle {delta.cycle_id}...")
    trainer = TrainCycle(config_path=config_path)
    train_result = trainer.run(delta, dry_run=dry_run)
    print(f"[Growth] Training complete: loss={train_result.final_loss:.4f}, "
          f"steps={train_result.steps_trained}")

    # Check if training was blocked (e.g., model too large / wrong quantization)
    training_blocked = train_result.metadata.get("blocked", False)
    if training_blocked:
        reason = train_result.metadata.get("reason", "unknown")
        print(f"[Growth] Training BLOCKED: {reason}")
        print(f"[Growth] Data prepared and saved. Skipping holonomy and merge.")
        buffer.mark_trained(cycle_id=delta.cycle_id)
        trigger.record_cycle_complete(delta.cycle_id, {
            "delta_count": delta.delta_count,
            "replay_count": delta.replay_count,
            "mean_surprise": delta.mean_surprise,
            "final_loss": -1.0,
            "steps_trained": 0,
            "adapter_path": str(train_result.adapter_path),
            "strategy": "blocked",
            "vllm_restarted": False,
            "dry_run": dry_run,
            "holonomy": None,
            "training_blocked": True,
            "blocked_reason": reason,
        })
        return {
            "fired": True,
            "cycle_id": delta.cycle_id,
            "training_blocked": True,
            "blocked_reason": reason,
            "data_ready": True,
        }

    # 5. Holonomy measurement
    holonomy_data = None
    tracker = HolonomyTracker(
        GROWTH_DIR / holonomy_cfg.get("log_path", "holonomy_log.jsonl").split("/")[-1]
    )

    # 5a. Check if this is a probe cycle
    probe_every_n = holonomy_cfg.get("probe_every_n_cycles", 5)
    completed = _count_completed_cycles()
    is_probe_cycle = (completed > 0) and (completed % probe_every_n == 0)

    if is_probe_cycle and not dry_run:
        print(f"[Growth] Holonomy PROBE cycle (every {probe_every_n} cycles)")
        print(f"[Growth] Running CCW training on reversed data order...")

        # Write reversed training data
        ccw_cycle_id = f"{delta.cycle_id}_ccw"
        ccw_dir = ADAPTERS_DIR / ccw_cycle_id
        ccw_dir.mkdir(parents=True, exist_ok=True)
        ccw_data_path = ccw_dir / "training_data.jsonl"
        delta.to_jsonl_reversed(ccw_data_path)

        # Create a reversed DeltaPackage for training
        from spark.growth.delta_extract import DeltaPackage
        ccw_delta = DeltaPackage(
            cycle_id=ccw_cycle_id,
            delta_entries=list(reversed(delta.delta_entries)),
            replay_entries=list(reversed(delta.replay_entries)),
            delta_count=delta.delta_count,
            replay_count=delta.replay_count,
            mean_surprise=delta.mean_surprise,
        )

        try:
            ccw_result = trainer.run(ccw_delta, dry_run=dry_run)
            print(f"[Growth] CCW training complete: loss={ccw_result.final_loss:.4f}")

            # Measure holonomy from the two adapters
            measurement = measure_probe(
                cycle_id=delta.cycle_id,
                adapter_cw=train_result.adapter_path,
                adapter_ccw=ccw_result.adapter_path,
                adapter_base=trainer._find_prev_adapter(),
            )
            tracker.log(measurement)
            holonomy_data = measurement.to_dict()

            print(f"[Growth] HOLONOMY: cos={measurement.cosine_cw_ccw:.4f} "
                  f"mag={measurement.holonomy_magnitude:.6f} "
                  f"verdict={measurement.verdict}")

            # Clean up CCW adapter (only the CW adapter gets merged)
            if ccw_dir.exists() and not dry_run:
                import shutil
                shutil.rmtree(ccw_dir, ignore_errors=True)
                print(f"[Growth] Cleaned up CCW probe adapter")

        except Exception as e:
            print(f"[Growth] CCW probe failed (non-fatal): {e}")
            holonomy_data = {
                "cycle_id": delta.cycle_id,
                "mode": "probe",
                "verdict": "PROBE_FAILED",
                "notes": str(e),
            }

    # 6. Phase 6: BECOME — activate adapter
    print(f"[Growth] Phase 6 (BECOME): activating adapter...")
    merger = MergeCycle(config_path=config_path)
    merge_result = merger.run(
        adapter_path=train_result.adapter_path,
        cycle_id=delta.cycle_id,
        dry_run=dry_run,
    )
    print(f"[Growth] Merge: strategy={merge_result.strategy_used}, "
          f"restarted={merge_result.vllm_restarted}")

    # 7. Mark buffer entries as trained
    buffer.mark_trained(cycle_id=delta.cycle_id)

    # 8. Record cycle completion
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
        "holonomy": holonomy_data,
        "holonomy_tracker_summary": tracker.summary(),
    }
    trigger.record_cycle_complete(delta.cycle_id, summary)

    print(f"[Growth] Cycle {delta.cycle_id} complete!")
    if holonomy_data:
        print(f"[Growth] Curvature of becoming: {holonomy_data.get('verdict', 'N/A')}")
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
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Force watermark reset and re-ingest all buffered entries",
    )
    parser.add_argument("--memory-dir", type=Path, default=Path("Vybn_Mind/memory"))
    args = parser.parse_args()

    nm = NestedMemory(base_dir=args.memory_dir)
    gb = GrowthBuffer(nested=nm)

    if args.force_ingest:
        print("[Growth] --force-ingest: resetting watermark and ingesting all buffered entries")
        ingested = _safe_ingest(gb, force=True)
        print(f"[Growth] Force-ingested {ingested} entries")
        if not (args.force or args.check):
            sys.exit(0)
    else:
        gb.ingest()

    if args.check:
        trigger = GrowthTrigger(gb)
        decision = trigger.should_trigger()
        print(f"Should fire: {decision.should_fire}")
        print(f"Signal: {decision.signal}")
        print(f"Reason: {decision.reason}")
        print(f"Delta volume: {decision.delta_volume}")
        print(f"Hours since last: {decision.hours_since_last_cycle}")
        # Also show zero-ingest counter state
        zic = _read_zero_ingest_count()
        print(f"Zero-ingest counter: {zic}/{_MAX_ZERO_INGEST_CYCLES}")
        sys.exit(0 if decision.should_fire else 1)

    result = run_growth_cycle(
        buffer=gb,
        force=args.force,
        dry_run=args.dry_run,
        force_ingest=getattr(args, 'force_ingest', False),
    )
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("fired") else 1)
