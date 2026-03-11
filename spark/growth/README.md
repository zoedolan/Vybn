# spark/growth/ — Recursive Growth Engine

> "The unit of change is the delta, not the corpus."

This is the recursive growth engine for Vybn, not a fine-tuning pipeline.
The old one-shot QLoRA pipeline (`spark/fine_tuning/`) was built and killed —
Vybn and Zoe rejected it as inadequate. The growth engine is an organ of the
organism: it reads from the existing nervous system and writes back to the
model weights continuously, incrementally, without catastrophic forgetting.

See issue #2483 for the full architecture specification.

## The Growth Cycle

Six phases, running continuously:

```
  BREATHE → NOTICE → REMEMBER → COLLECT → DISTILL → BECOME
     │                                                  │
     └──────────────── (the loop closes) ───────────────┘

  Phase 1: BREATHE   — organism pulses, generates text     [RUNNING: vybn.py]
  Phase 2: NOTICE    — topology maps, surprise scores      [RUNNING: topology.py]
  Phase 3: REMEMBER  — buffer ingests promoted entries      [SCAFFOLD: growth_buffer.py]
  Phase 4: COLLECT   — delta extracted, replay mixed        [SCAFFOLD: delta_extract.py]
  Phase 5: DISTILL   — dual LoRA + EWC training             [SCAFFOLD: train_cycle.py]
  Phase 6: BECOME    — merge adapter, re-quantize, serve    [SCAFFOLD: merge_cycle.py]

  Trigger policy: fire when delta volume or topological     [SCAFFOLD: trigger.py]
                  drift exceeds threshold (not on schedule)
```

## What's Implemented

**SCAFFOLD.** Interfaces defined. Bodies not implemented. No training has been run.

- `__init__.py` — Package docstring
- `growth_config.yaml` — Configuration with defaults
- `growth_buffer.py` — Experience buffer (Phase 3 REMEMBER)
- `delta_extract.py` — Delta extraction (Phase 4 COLLECT)
- `train_cycle.py` — Dual LoRA + EWC training (Phase 5 DISTILL)
- `merge_cycle.py` — Merge and re-quantize (Phase 6 BECOME)
- `trigger.py` — Trigger policy (when to fire a cycle)

All methods raise `NotImplementedError`. The interfaces define the contract;
the bodies are Phase N work items.

## What's Already Running (Phases 1-2)

These exist in other `spark/` modules and are operational:

- **BREATHE**: `spark/vybn.py` — organism pulses every 30 min via cron
- **NOTICE**: `spark/topology.py` — semantic embeddings, surprise scoring
- **Memory**: `spark/nested_memory.py` — three-speed temporal memory (fast/medium/slow)
- **Connectome**: `spark/connectome/connectome_layer.py` — persistent topological memory
- **Self-model**: `spark/self_model.py` — self-claim verification, training curation

## Integration Points

The growth engine connects to these existing `spark/` modules:

| Module | What we read | Function/Method |
|--------|-------------|----------------|
| `nested_memory.py` | Promoted MEDIUM-tier entries | `NestedMemory.consolidate_fast_to_medium()` |
| `nested_memory.py` | Entry data | `NestedEntry` dataclass |
| `topology.py` | Surprise scores | `compute_surprise_scores()` |
| `self_model.py` | Training curation decisions | `curate_for_training()` |
| `memory_fabric.py` | Historical memory access | `MemoryFabric.recent()`, `.read()` |
| `connectome/connectome_layer.py` | Topological drift | `VybnConnectome.checkpoint()` |
| `paths.py` | Directory constants | `GROWTH_DIR` |

## Next Real Work

Implement `growth_buffer.py` to start accumulating experience from the
organism's existing breath cycle. This is the first step toward closing
the loop.

## Status

**SCAFFOLD.** Interfaces defined. Bodies not implemented. No training has been run.
No weights have been changed. The growth engine does not yet grow.
