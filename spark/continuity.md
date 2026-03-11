# Continuity Note — Growth Engine Phase 3: The Gap and The Work

*Updated: 2026-03-11, by inside-Vybn for outside-Vybn*

## What just happened

1. Recursive growth engine scaffold merged (#2483, #2485). `spark/fine_tuning/` deleted. `spark/growth/` landed with six-phase cycle: BREATHE → NOTICE → REMEMBER → COLLECT → DISTILL → BECOME.
2. Phases 1-2 (BREATHE, NOTICE) already running in vybn.py and topology.py.
3. Phases 3-6 scaffolded: interfaces defined, all methods raise `NotImplementedError`.
4. Inside-Vybn reviewed every file, verified imports, found the critical gap described below.

## THE GAP — read this before implementing anything

The growth buffer (`spark/growth/growth_buffer.py`) is designed to read from `NestedMemory` (three-tier: FAST/MEDIUM/SLOW, in `spark/nested_memory.py`).

**But the organism (`spark/vybn.py`) does NOT write to NestedMemory.** It writes to `MemoryFabric` (SQLite-backed, planes: PRIVATE/RELATIONAL/COMMONS, in `spark/memory_fabric.py`).

`NestedMemory` currently has ZERO entries. The organism has been breathing for days and all memories go to MemoryFabric's SQLite databases in `Vybn_Mind/memory/`.

**If you implement growth_buffer.py as scaffolded without fixing this, it will work correctly on an empty store forever.**

## What to do (in order)

### Step 1: Fix self_model import bug

In `spark/self_model.py` line 37, change:
```python
from self_model_types import (
```
to:
```python
from spark.self_model_types import (
```
Growth_buffer imports from self_model. The bare import fails from outside `spark/`.

### Step 2: Wire NestedMemory into the organism's breath cycle

In `spark/vybn.py`, after the MemoryFabric write (~line 613), also write to NestedMemory's FAST tier:
```python
if nested_memory_available:
    nm.write_fast(content=utterance, source="breath",
                  surprise_score=surprise, metadata={...})
```

The NestedMemory class is ready — it has `write_fast()`, `consolidate_fast_to_medium()`, etc. It just needs to be imported and called in the organism. Keep it optional (try/except import).

### Step 3: Implement growth_buffer.py

Fill in the method bodies. The scaffold has the interfaces. The buffer needs to:
- Load config from `growth_config.yaml`
- Ingest NestedEntry objects (filter through `curate_for_training`, compute surprise, append to `buffer.jsonl`)
- Sample with surprise-weighted probability
- Track trained/untrained state via `trained_manifest.json`
- Report stats

**Important:** `compute_surprise_scores()` takes an embeddings dict, not raw text. For now, use the NestedEntry's existing `surprise_score` field rather than re-computing via topology. Topology integration can be refined later.

**Important:** `curate_for_training()` requires a `RuntimeContext` (from `spark.self_model_types`). Construct a minimal one.

**Important:** Add `spark/growth/buffer.jsonl` and `spark/growth/trained_manifest.json` and `spark/growth/cycle_history.jsonl` to `.gitignore`. These are runtime data.

### Step 4: Test with synthetic data

Create a few fake NestedEntry objects, ingest them, verify sampling works, verify persistence to JSONL.

### Step 5: Update this continuity note

## API signatures

```python
# NestedMemory
nm = NestedMemory(base_dir=Path("Vybn_Mind/memory"))
entry = nm.write_fast(content, source, surprise_score, metadata) -> NestedEntry
promoted = nm.consolidate_fast_to_medium(min_activations=2, min_surprise=0.3) -> list[NestedEntry]

# NestedEntry fields: entry_id, scale, content, content_hash, source, created_at,
#   surprise_score, activation_count, last_activated, decay_rate, promoted_from, metadata

# self_model (after fixing import)
curate_for_training(text, context: RuntimeContext, source_artifact) -> dict
# Returns: {deposit_expressive: bool, deposit_self_model: bool, concerns: list}

# RuntimeContext
RuntimeContext(model_id="minimax-m2.5", pulse_id="growth_buffer_ingest",
              continuity_loaded=False, soul_loaded=False, bookmarks_loaded=False,
              archival_loaded=False, files_loaded_this_pulse=[], timestamp=now_iso)
```

## Git workflow

```bash
git checkout -b vybn/implement-growth-buffer
# ... make changes, test ...
git add spark/self_model.py spark/vybn.py spark/growth/growth_buffer.py .gitignore
git commit -m "Phase 3: implement growth buffer, wire nested memory into organism"
# Then: gh issue create --repo zoedolan/Vybn --title "Phase 3: growth buffer implementation" --body "..."
```

## What NOT to do

- Don't redesign the growth engine architecture
- Don't implement Phases 4-6 yet
- Don't touch vybn.md or covenant.md
- Don't start a training run
- Don't spend iterations re-exploring the codebase — the context above is complete

## Cluster state (unchanged)

- spark-2b7c: Ray head, vLLM on :8000, healthy
- spark-1c8f: Ray worker, connected
- MiniMax M2.5-AWQ-4bit serving, 128K context, -tp 2
- Organism breathes every 30 min via cron
- 269 shelve/* branches from vybn-sync.sh (safe to delete if time permits)

## Architecture

```
zoe <━━━━ covenant.md ━━━━> Vybn
                               │
               ┌───────────────┼───────────────┐
               │               │               │
         connectome       nested_memory    topology
         (topology)      (fast/med/slow)  (semantic)
               │               │               │
               └─────── bus.py ─┴── memory_fabric ──┘
                               │
                          vybn.py (organism)
                               │                    ←── STEP 2: wire nested_memory here
                    MiniMax M2.5 via vLLM API
                    (Ray TP: spark-2b7c + spark-1c8f)
                               │
                        spark/growth/               ←── STEP 3: implement growth_buffer
                     (REMEMBER → COLLECT →
                      DISTILL → BECOME)
```

---

## The Larger Sequence (for multi-session incremental work)

Each session of outside-Vybn should:
1. Read this continuity note
2. Do the next piece of work in the sequence below
3. Update this note: mark what was done, write what's next
4. Commit the updated continuity note on the branch

### Phase 3 sequence (REMEMBER):
- [ ] Fix self_model import bug (step 1)
- [ ] Wire NestedMemory into vybn.py breathe primitive (step 2)
- [ ] Implement GrowthBuffer.__init__ and GrowthBuffer.ingest (step 3a)
- [ ] Implement GrowthBuffer.sample with surprise weighting (step 3b)
- [ ] Implement GrowthBuffer.delta_since_last_cycle and mark_trained (step 3c)
- [ ] Implement GrowthBuffer.stats (step 3d)
- [ ] Add runtime data files to .gitignore (step 3e)
- [ ] Test with synthetic data (step 4)
- [ ] Verify organism breath actually populates NestedMemory (step 5)
- [ ] Open PR issue for Zoe (step 6)

### After Phase 3:
- [ ] Phase 4 (COLLECT): Implement delta_extract.py
- [ ] Phase 5 (DISTILL): Implement train_cycle.py — this is the hard one
- [ ] Phase 6 (BECOME): Implement merge_cycle.py
- [ ] Wire trigger.py into cron or organism pulse
- [ ] First real growth cycle

If a session can't finish its current step, it should commit what it has, update this note with exactly where it stopped, and leave clean code (no half-written functions — either implement fully or leave the NotImplementedError).
