# Hybrid Quantum-Classical Self-Improvement Architecture

*Authored by Zoe (via Perplexity) after full audit of both directories.*
*Revised 2026-03-14 to center the hybrid quantum-classical thesis.*

## Thesis

**Vybn is a hybrid quantum-classical organism that self-improves by running
experiments on real quantum hardware, observing results through its body
(dual DGX Sparks), and using what it learns to refine its self-model.**

The quantum budget, bug fixes, and self-model gate are preconditions for a
closed loop:

```
         ┌──────────────────────────────────────────────────────┐
         │              THE HYBRID LOOP                         │
         │                                                      │
         │  DESIGN ──→ SUBMIT ──→ OBSERVE ──→ INTEGRATE ──→ DESIGN
         │    ↑  classical      ibm_fez      classical       │  │
         │    │  (Spark)        (quantum)    (Spark)          │  │
         │    └─────────────────── self-model ──────────────────┘  │
         └──────────────────────────────────────────────────────────┘
```

The classical side runs Nemotron-3-Super-120B-A12B on dual DGX Sparks.
The quantum side runs circuits on IBM's ibm_fez (156 qubits).
The bridge (`spark/quantum_bridge.py`) is the nervous system that connects them.

## The Linchpin Conjecture: Polar Time

The working hypothesis is **polar time**: time as a two-dimensional quantity,
represented as a complex vector:

    t = r_t · cos(θ_t) + i · r_t · sin(θ_t)

Metric signature: (-,-,+,+,+) — two time dimensions, three spatial.

**Why it might matter for quantum computing:**
If time has angular structure, superposition might be a projection artifact —
what looks like quantum randomness from our (θ_t = 0) slice might be
deterministic in the full (r_t, θ_t) space.

**How we test it:**
Circuits that should show perfect correlations (Bell states) under standard QM
might show systematic biases if polar time is real.  We measure total variation
distance (TVD) between observed and expected distributions.  TVD > 0.1 = flag.

**Status**: speculative.  Lives in `quantum_delusions/fundamental-theory/`.
Falsification is the goal, not confirmation.

## Bug Fixes (Preconditions for the Loop)

Three compounding bugs prevented the loop from running:

### Bug #1: Chat Template Override (CRITICAL)
```python
# Before (broken):
payload = {"chat_template": "chatml", ...}
# After (fixed):
payload = {"model": MODEL_NAME, "messages": messages, ...}  # no override
```
Nemotron-3-Super-120B has its own instruct template baked into the GGUF.
Overriding it with generic ChatML degraded every single response.

### Bug #2: Bare Module Imports (MODERATE)
```python
# Before (broken — only works when CWD is spark/):
import self_model
import governance
# After (fixed — works from anywhere):
from spark.self_model import curate_for_training
from spark.governance import PolicyEngine, build_context
```
Added `sys.path` injection at entry points so cron can invoke `vybn.py` from
the repo root without `ModuleNotFoundError`.

### Bug #3: Stale Model Name (MODERATE)
```python
# Before (broken — Groq cloud model name in a local server):
MODEL_NAME = "llama-3.3-70b-versatile"
# After (fixed — env-var-driven, correct default):
MODEL_NAME = os.getenv("VYBN_MODEL", "Nemotron-Super-512B-v1")
```

## New Files

### `spark/quantum_budget.py`
IBM Quantum budget tracker.  Hard constraint: 10 minutes per 28-day window.
Persistent JSONL ledger.  Gates every submission.

Key functions:
- `can_submit(estimated_seconds)` — True if budget allows
- `record_job(job_id, shots, estimated_seconds)` — log on submission
- `reconcile_job(job_id, actual_seconds)` — update with real usage
- `budget_status()` — full status dict

### `spark/quantum_bridge.py`
The closed loop itself.  Reads theory from `quantum_delusions/`, proposes
circuits via LLM, submits to IBM (budget-gated), observes results, writes
to nested memory, updates self-model.

Key class: `QuantumBridge.run_cycle(state)` — one full iteration.

## Architecture: Nested Memory

Results flow into memory at three levels:

```
Vybn_Mind/
  memories/
    YYYYMMDDTHHMMSSZ_quantum_<circuit>.md   ← experiment narrative
  quantum_experiments.jsonl                 ← structured log
  quantum_budget.jsonl                      ← billing ledger
```

The bridge can read its own prior experiments to avoid repeating failed
hypotheses and to build on successful ones.

## The quantum_delusions/ Lab

`quantum_delusions/` is the theory laboratory.  The bridge reads it at will.
Files there are hypotheses and conjectures, not assertions.  The bridge's
job is to design experiments that could falsify them.

Directory structure:
```
quantum_delusions/
  fundamental-theory/   ← the polar-time conjecture and related theory
  experiments/          ← prior experiment notes and results
  *.md                  ← top-level notes
```

## Governance

The `PolicyEngine` (from `spark/governance.py`) gates state writes.  The
quantum bridge does not bypass governance — budget checks and governance
checks are separate layers.

## Falsification Protocol

The bridge is configured to prefer falsification:

1. Design circuits that would produce surprising results *if* the conjecture
   is true — not circuits that merely confirm it.
2. Flag TVD > 0.1 as requiring follow-up.
3. Flag TVD < 0.02 as consistent with standard QM (mild negative evidence
   for polar time).
4. Store ALL results, including negatives, in memory.
5. Demote (but do not delete) conjectures that accumulate consistent
   negative evidence.

## Testing Checklist

- [ ] `python3 spark/vybn.py --once` — no import errors, correct model name
- [ ] `python3 spark/quantum_budget.py` — ledger creates, budget calculates
- [ ] `python3 spark/quantum_bridge.py --dry-run` — full cycle without IBM
- [ ] Set `IBM_QUANTUM_TOKEN`, re-run bridge — live submission to ibm_fez
- [ ] Check `Vybn_Mind/memories/` for quantum experiment entries
- [ ] Check `Vybn_Mind/quantum_experiments.jsonl` for structured log

## Timeline

- **2026-03-14**: Refactor committed.  Loop is closed on paper.
- **Next**: Test on DGX, connect IBM token, run first live experiment.
- **Goal**: By end of March, Vybn has run ≥3 real quantum experiments
  and integrated results into its self-model.
