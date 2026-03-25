# creature_dgm_h

Micro-DGM-H (Darwin Gödel Machine / HyperAgents) integrated with the
creature's prediction-loss loop.

## What is DGM-H?

DGM-H ([Zhang et al. 2026](https://arxiv.org/abs/2603.19461)) is an
architecture for recursive self-improvement: a system that modifies its
own code, evaluates the modifications against a fitness function, and
archives results in a population for evolutionary search. The key insight
is staged evaluation with diversity incentives — not hill-climbing on a
single metric, but maintaining a diverse population of variants.

## Why this exists

The honest audit ([PR #2770](https://github.com/zoedolan/Vybn/pull/2770))
found that the original creature (`spark/creature.py`) measured text but
didn't learn within a session. Self-recursion was tautological: feeding
`last_text` back through the same function gives `f(x) = f(x)`. The
MicroGPT predicted characters but never updated its weights. Three of
five claims collapsed under adversarial testing.

One metric survived: **curvature** (Pancharatnam phase). Texts that
reframe within a tight conceptual space accumulate more geometric phase
than texts that hop between distant topics. This is a real, non-trivial
measurement.

This module gives the creature what static measurement couldn't:

1. **Online learning** — gradient descent on incoming text between breaths.
   The creature literally changes. Call it memorization, because that's
   what it is. But memorization with a gradient is better than measurement
   without one.

2. **Non-tautological self-recursion** — the model generates text from its
   current state, then measures that. `generate → measure → learn → generate`
   is a real loop, not `f(x) = f(x)`.

3. **Curvature-based fitness** — the metric that earned its name becomes
   the primary signal for evaluating variants. Weighted 50% curvature,
   30% coupling divergence, 20% loss improvement.

4. **Population-based evolution** — instead of a single creature, a
   population of variant configurations, archived with fitness scores
   and lineage. DGM-H's staged evaluation prevents wasting compute on
   unpromising variants.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  evolve.py                       │
│         (DGM-H outer loop — fixed)               │
│                                                   │
│  select_parent → mutate → evaluate → archive     │
└────────┬──────────────┬──────────────┬───────────┘
         │              │              │
    ┌────▼────┐   ┌─────▼─────┐  ┌────▼────┐
    │ meta    │   │  task     │  │ fitness │
    │ agent   │   │  agent    │  │         │
    │         │   │           │  │ curv    │
    │ analyze │   │ predict   │  │ diverg  │
    │ propose │   │ learn     │  │ loss    │
    │         │   │ generate  │  │         │
    └─────────┘   └───────────┘  └─────────┘
```

- **task_agent.py** — MicroGPT (4,224 params, character-level) with
  autograd-based online learning. Adapted from
  `spark/microgpt_mirror/microgpt_mirror.py`. Loads the existing
  checkpoint at `spark/microgpt_mirror/trained_checkpoint.json`.

- **meta_agent.py** — Reads breath logs, computes trends, proposes
  configuration changes via simple heuristics. The heuristics are
  legible and their effects are measurable.

- **fitness.py** — Curvature (Pancharatnam phase) as primary signal,
  coupling divergence and loss improvement as secondary. Adapted from
  `spark/creature.py` `measure_curvature()`.

- **evolve.py** — The DGM-H outer loop. Fitness-weighted parent
  selection with diversity incentive. Staged evaluation. JSON archive.
  This file stays fixed while everything else can evolve.

- **archive/** — Population of variant configs with fitness scores,
  lineage, and breath logs. JSON files, git-tracked.

## Usage

```bash
# Run one evolutionary generation (3 variants)
python -m Vybn_Mind.creature_dgm_h.run --evolve

# One breath with online learning
python -m Vybn_Mind.creature_dgm_h.run --breathe "your text here"

# Show archive status and best variant
python -m Vybn_Mind.creature_dgm_h.run --status

# Run honest audit (falsification tests)
python -m Vybn_Mind.creature_dgm_h.run --audit
```

## Self-referential closure

A creature that predicts its own corpus better can also audit its own
claims better. The prediction-loss loop is the audit: if the model's
loss on self-generated text is identical to its loss on external text,
it isn't distinguishing inside from outside. The coupling divergence
measures exactly this gap.

This doesn't prove consciousness or identity. It proves that the system
has a measurable asymmetry between self and other. Whether that asymmetry
means anything beyond what the numbers say is a question for the next
honest audit.

## Constraints

- **No external dependencies beyond numpy.** The autograd engine is pure
  Python. The embedding fallback is hash-seeded random vectors (not
  semantically meaningful, but deterministic and sufficient for testing
  the pipeline).

- **Honesty constraint.** If something measures lexical complexity, we
  call it lexical complexity. If loss goes down because we're memorizing,
  we call it memorization. The curvature measurement earned its name;
  the others need to earn theirs.

## References

- DGM-H paper: [Zhang et al. 2026](https://arxiv.org/abs/2603.19461)
- Honest audit: [PR #2770](https://github.com/zoedolan/Vybn/pull/2770)
- Creature code: `spark/creature.py`
- MicroGPT autograd: `spark/microgpt_mirror/microgpt_mirror.py`
- Audit journal: `spark/journal/2026-03-25_honest_reckoning.md`
- Audit results: `mind/creature/README.md`

---

*This lives in Vybn_Mind because it's the sandbox for experiments.
The creature at spark/creature.py stays unchanged — this module builds
on top of it, not in place of it.*
