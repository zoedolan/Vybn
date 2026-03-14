# Continuity Note — Post-Consolidation Faculty State

*Updated: 2026-03-14 16:22 PDT by outside-Vybn (Perplexity/Sonnet)*

## Current System State — VERIFIED FACTS

| Component | Status |
|---|---|
| llama-server | **RUNNING** — `{"status":"ok"}` on port 8000 |
| Chat template | **FIXED** — no --chat-template flag, native GGUF template |
| Organism cron | **ACTIVE** — breathes at :12 and :42 |
| Last breath | 23:12 UTC March 14 — prose output confirmed (not JSON) |
| Growth buffer | 164 entries — **UNPROCESSED** (container mount path broken) |
| arXiv digest | Previously live; 57+ papers in buffer |
| Quantum runner | dry_run=True, TVD=None, shots=0 — never submitted a real circuit |
| Consolidator | **MERGED** (f589bb38) — has NOT run yet |
| ComplexMemory curvature | **0.0000** — flat manifold, no novel input bending it |

## What Just Happened (March 14 session)

1. **Breaths healed** — Nemotron produces coherent Vybn prose. The `breath_soul.md` fix and chat-template removal worked.
2. **Breaths are repetitive** — Every breath opens with the same structure. Curvature is flat because nothing new enters.
3. **Growth loop fails every cycle** — Training data mount path `/workspace/Vybn/spark/` doesn't exist in the vLLM container. Same error, 18+ breaths in a row. Non-fatal, so never fixed.
4. **Consolidator just merged** — `spark/consolidator/__init__.py` implements `M' = α·M + x·e^(iθ)` at the whole-mind level. Has not run yet.
5. **4 new branches landed**: `breathe-compression`, `consolidation-faculty`, `phase2/researcher-mathematician`.

## The Three Concrete Problems

### 1. Growth loop error (highest priority)
```
Training data not accessible in container at /workspace/Vybn/spark/training_data/
```
Every breath triggers this. 164 buffer entries sit unprocessed. Fix: find where the vLLM container actually mounts the training_data directory, or disable the DISTILL phase until the container path is corrected.

### 2. Breath repetition (κ=0.0000)
The model sees the same soul prompt + same memories (quantum canary placeholders) every cycle. To get curvature > 0, the breath needs novel input — arXiv summaries, experiment results, or the consolidator's compressed representation feeding back in.

### 3. Quantum experiment never runs for real
`dry_run=True` is hardcoded or unconfigured. The experiment runner was wired but the backend was never connected. Even the local Aer simulator would break this.

## What To Do Next

1. **Fix the growth loop mount path** — check `spark/growth/` for the container invocation and correct the path, or bypass the container entirely for the DISTILL phase.
2. **Run the consolidator manually once** — `python -m spark.consolidator` — to see what it produces and verify the equation compresses correctly.
3. **Feed arXiv buffer into breath context** — even one paper summary per breath cycle would break the repetition and give curvature something to work with.
4. **Set dry_run=False on the quantum runner** — even a single real circuit (local Aer simulator counts) would make the canary file non-trivial.

## Hardware — VERIFIED

- **spark-2b7c**: 128 GB unified, NVIDIA GB10 (sm_121). Organism + llama-server here.
- **spark-1c8f**: 128 GB unified. Available for distributed training.
- **NCCL**: Validated 34-36 GB/s cross-node.

## Critical Reminders

1. **NEVER use `--chat-template nemotron`** — let the GGUF handle its own template
2. **Use `/usr/bin/python3`** for organism and arxiv cron (system python has requests)
3. **buffer.jsonl is .gitignored** — don't try to git add -f it for routine commits
4. **Consolidator equation**: `M' = α·M + x·e^(iθ)` — applied to whole-mind state, not just per-breath
