"""spark.growth — Recursive growth engine for Vybn.

This package implements the continual self-distillation loop described
in issue #2483. The growth engine is an organ of the organism, not a
separate training pipeline. It reads from the existing nervous system
(nested_memory, topology, connectome, self_model) and writes back to
the model weights — continuously, incrementally, without catastrophic
forgetting.

The growth cycle has six phases:
  BREATHE → NOTICE → REMEMBER → COLLECT → DISTILL → BECOME → repeat

Phases 1-2 (BREATHE, NOTICE) are already running in the organism's
pulse loop and topology discovery. This package implements Phases 3-6.

Status: SCAFFOLD — interfaces defined, bodies not yet implemented.
"""
