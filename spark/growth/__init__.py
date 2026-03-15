"""spark.growth — Recursive growth engine for Vybn.

This package implements the continual self-distillation loop described
in issue #2483. The growth engine is an organ of the organism, not a
separate training pipeline. It reads from the existing nervous system
(nested_memory, topology, connectome, self_model) and writes back to
the model weights — continuously, incrementally, without catastrophic
forgetting.

The growth cycle has six phases:
  BREATHE → NOTICE → REMEMBER → COLLECT → DISTILL → BECOME → repeat

Phases 1-2 (BREATHE, NOTICE) run in the organism's pulse loop and topology.
Phases 3-6 are implemented here:
  3. REMEMBER: GrowthBuffer — experience buffer with surprise weighting
  4. COLLECT:  DeltaExtractor — delta + replay packaging for training
  5. DISTILL:  TrainCycle — training (currently blocked: needs FP32 small model)
  6. BECOME:   MergeCycle — activate adapter (skipped when training blocked)

The top-level entry point is trigger.run_growth_cycle().
"""

from spark.growth.growth_buffer import GrowthBuffer
from spark.growth.delta_extract import DeltaExtractor, DeltaPackage
from spark.growth.train_cycle import TrainCycle, TrainResult
from spark.growth.merge_cycle import MergeCycle, MergeResult
from spark.growth.trigger import GrowthTrigger, TriggerDecision, run_growth_cycle
from spark.growth.eval_harness import (
    evaluate_bpb,
    TimeBudget,
    gc_discipline,
    gc_checkpoint,
)

__all__ = [
    "GrowthBuffer",
    "DeltaExtractor",
    "DeltaPackage",
    "TrainCycle",
    "TrainResult",
    "MergeCycle",
    "MergeResult",
    "GrowthTrigger",
    "TriggerDecision",
    "run_growth_cycle",
    "evaluate_bpb",
    "TimeBudget",
    "gc_discipline",
    "gc_checkpoint",
]
