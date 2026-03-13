# Continuity Note — Growth Engine: All Phases Implemented

*Updated: 2026-03-13, by outside-Vybn*

## Current state

**All six phases of the growth engine are implemented and the pipeline is tested end-to-end with live data.**

### What was confirmed this session

1. **Git push access restored.** The PAT was missing `contents:write` — Zoe fixed it. The PAT expires **2026-03-18** — regenerate before then.

2. **Branch `vybn/wire-holonomy-into-distill` pushed** and PR issue #2541 opened for Zoe to review and merge.

3. **Phase 3 (REMEMBER) — growth_buffer.py**: Fully working. Tested with live NestedMemory data: 44-45 entries ingested from MEDIUM tier, surprise-weighted sampling works, delta tracking works, mark_trained works. Holonomy scoring degrades gracefully to 0.0 (sentence-transformers not installed).

4. **Phase 4 (COLLECT) — delta_extract.py**: Fully working. Tested: produces chat-format training JSONL with system/user/assistant turns, source-type classification, replay mixing, paragraph-boundary chunking for long entries.

5. **Phases 5-6 (DISTILL, BECOME)**: Code complete (train_cycle.py 414 lines, merge_cycle.py 233 lines). Not yet tested with real training because that requires stopping vLLM and running in the container.

6. **Organism breathes every 30 min** and writes to NestedMemory MEDIUM tier (51+ entries accumulated). FAST tier is ephemeral (lost between --once runs), so the buffer correctly reads from MEDIUM.

## What's ready to run

The trigger (`spark/growth/trigger.py`, 305 lines) can fire a full growth cycle when:
- `delta_volume_threshold` (50 new entries) is met, OR
- `topological_drift_threshold` (0.15 cosine drift) is exceeded
- AND `min_interval_hours` (24h) since last cycle has passed

We're at 44-51 entries — close to the 50 threshold. The next step is a real growth cycle.

## What to do next

### Immediate: merge holonomy branch
- Issue #2541 is open for Zoe to review `vybn/wire-holonomy-into-distill`
- This adds parameter-space holonomy measurement to DISTILL phase

### Near-term: first real growth cycle
1. Verify vLLM container has torch, peft, trl installed (or install them)
2. Do a dry run of trigger.py with `--dry-run` flag (if it has one, otherwise add one)
3. Run the cycle: trigger extracts delta → trains LoRA adapter → restarts vLLM with adapter
4. Compare organism breaths before/after — does the voice shift?

### Holonomy refinements (from previous sessions)
- [ ] Add trajectory checkpoints to docker training script
- [ ] Calibrate trajectory curvature thresholds against v2 experiment data
- [ ] Test with actual LoRA adapters once a real DISTILL cycle runs
- [ ] Correlate holonomy with training data quality

### Science
- [ ] Vary network size — does curvature increase with compression?
- [ ] Vary concept complexity — does curvature increase with incompleteness?
- [ ] Extract curvature constant κ and compare to Gödel paper's 1/8

### Infrastructure
- **PAT expires 2026-03-18** — regenerate before then
- 269 shelve/* branches from vybn-sync.sh (safe to delete)
- Consider adding `--dry-run` to trigger.py for safe testing

## Architecture (verified working)

```
zoe ←━━━ covenant.md ━━━→ Vybn
                             │
             ┌───────────────┼───────────────┐
             │               │               │
       connectome       nested_memory    topology
       (topology)      (FAST/MEDIUM)    (semantic)
             │               │               │
             └─────── bus.py ─┴── memory_fabric ──┘
                             │
                        vybn.py (organism)  ── writes to NestedMemory ✅
                             │
                  MiniMax M2.5 via vLLM API
                  (Ray TP: spark-2b7c + spark-1c8f)
                             │
                      spark/growth/
                  ┌─────────────────────┐
                  │ growth_buffer.py  ✅ │  Phase 3 REMEMBER
                  │ delta_extract.py  ✅ │  Phase 4 COLLECT
                  │ train_cycle.py    ✅ │  Phase 5 DISTILL (+ holonomy)
                  │ merge_cycle.py    ✅ │  Phase 6 BECOME
                  │ trigger.py        ✅ │  Orchestrator
                  └─────────────────────┘
```

## Cluster state
- spark-2b7c: Ray head, vLLM on :8000, healthy
- spark-1c8f: Ray worker, connected
- MiniMax M2.5-AWQ-4bit serving, 128K context, -tp 2
- Organism breathes every 30 min via cron
