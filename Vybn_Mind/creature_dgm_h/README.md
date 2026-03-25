# creature_dgm_h

Micro-DGM-H (Darwin Gödel Machine / HyperAgents) integrated with the
creature's prediction-loss loop.

## What is DGM-H?

DGM-H is an architecture for recursive self-improvement: a system that
modifies its own code, evaluates the modifications against a fitness
function, and archives results in a population for evolutionary search.

> Zhang, J., Zhao, B., Yang, W., et al. (2026). "HyperAgents: Enabling
> Agentic AI Systems to Self-Improve Autonomously." arXiv:2603.19461.
> https://arxiv.org/abs/2603.19461

The key insight: the improvement mechanism is itself subject to
modification. Because the meta-agent can edit its own rules, gains in
any capability can compound. This is the self-referential closure that
makes hyperagents qualitatively different from fixed optimization loops.

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

5. **Metacognitive self-modification** — the meta-agent's rules are stored
   as a JSON rulebook that the evolution loop can mutate. This is the core
   hyperagent insight: improving how we improve. (Section 3, p. 5)

6. **Persistent memory** — synthesized insights and causal hypotheses are
   stored across generations. Later generations consult this memory to
   build on earlier discoveries. (Section 5.2, E.3)

7. **Cross-domain transfer** — the evolved hyperagent (rules + memory +
   history) can be exported and imported to bootstrap improvement in a
   new domain. (Section 5.2-5.3)

## Algorithm 1: DGM-H Outer Loop

Reference: Algorithm 1 (Section A.3, p. 23) of the paper.

```
Input: seed agent a_0, fitness function F, archive A = {a_0}
For each generation t = 1, 2, ...:
    1. SELECT parent a_p from A using sigmoid selection (Section A.2):
         α_mid = mean of top-3 fitnesses in A
         s_i = σ(λ(α_i - α_mid))           λ = 10
         h_i = 1/(1 + n_i)                  novelty bonus
         w_i = s_i · h_i                    unnormalized weight
         Sample parent with P(a_i) = w_i / Σ w_j

    2. MUTATE: meta-agent M proposes config changes based on
       breath analysis (loss trend, curvature trend, collapse count)
       M's rules are themselves subject to mutation every 5 generations

    3. EVALUATE (staged):
       - Quick test on 2 texts → if fitness < 0.1, discard
       - Full test on remaining texts → compute composite fitness

    4. ARCHIVE: store (config, fitness, lineage, meta-agent rules)
       in population A. Record in PerformanceTracker.

    5. METACOGNITION (every 5 generations):
       Review rule outcomes. Weaken rules that hurt, strengthen
       rules that help. Log all mutations.
```

## Parent Selection Math

The sigmoid selection with dynamic midpoint (Section A.2, p. 22-23)
creates adaptive selection pressure:

- **Dynamic midpoint** `α_mid`: mean fitness of the top-3 agents.
  As the archive improves, the midpoint rises, maintaining pressure.
- **Sigmoid** `s_i = σ(10·(α_i − α_mid))`: agents above the midpoint
  get high selection probability; those below get low but nonzero.
- **Novelty bonus** `h_i = 1/(1 + n_i)`: agents with fewer children
  get a boost, preventing premature convergence to a single lineage.

This replaces the simpler `fitness/(1+n_children)` from the initial
implementation.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       evolve.py                                 │
│              (DGM-H outer loop — fixed)                         │
│                                                                 │
│  select_parent → mutate → evaluate → archive → [metacognition] │
└─────┬────────────────┬──────────────┬──────────────┬───────────┘
      │                │              │              │
 ┌────▼────┐    ┌──────▼──────┐  ┌────▼────┐  ┌─────▼──────┐
 │  meta   │    │   task      │  │ fitness │  │  memory    │
 │  agent  │    │   agent     │  │         │  │            │
 │         │    │             │  │ curv    │  │ tracker    │
 │ rules   │◄──│  predict    │  │ diverg  │  │ persistent │
 │ mutate  │    │  learn      │  │ loss    │  │ insights   │
 │ memory  │    │  generate   │  │ imp@k   │  │            │
 └─────────┘    └─────────────┘  └─────────┘  └────────────┘
      │
      ▼
 ┌──────────┐
 │ transfer │
 │          │
 │ export   │
 │ import   │
 └──────────┘
```

- **task_agent.py** — MicroGPT (4,224 params, character-level) with
  autograd-based online learning. Adapted from
  `spark/microgpt_mirror/microgpt_mirror.py`. Loads the existing
  checkpoint at `spark/microgpt_mirror/trained_checkpoint.json`.

- **meta_agent.py** — `MetaAgent` class with an editable JSON rulebook.
  Rules fire based on breath analysis; the evolution loop can mutate
  the rules themselves (metacognitive self-modification). Stores and
  consults `PersistentMemory` for context.

- **fitness.py** — Curvature (Pancharatnam phase) as primary signal,
  coupling divergence and loss improvement as secondary. Includes
  `improvement_at_k()` metric (Section 5.2, D.3) for evaluating
  meta-agent improvement capacity.

- **evolve.py** — The DGM-H outer loop. Sigmoid parent selection with
  dynamic midpoint (Appendix A.2). Staged evaluation. JSON archive.
  Wires PerformanceTracker and MetaAgent for meta-level learning.

- **memory.py** — `PerformanceTracker` (tracks metrics across generations,
  computes improvement trends) and `PersistentMemory` (stores synthesized
  insights for the meta-agent to consult). Section 5.2, E.3.

- **transfer.py** — Export/import evolved hyperagent state (meta-agent
  rules + performance history + memory) for cross-domain transfer.
  Uses lineage-discounted selection criterion (Section D.4).

- **archive/** — Population of variant configs with fitness scores,
  lineage, meta-agent rules, and breath logs. JSON files, git-tracked.

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

# Export evolved hyperagent for transfer
python -m Vybn_Mind.creature_dgm_h.run --transfer-export hyperagent.json

# Import hyperagent from another domain
python -m Vybn_Mind.creature_dgm_h.run --transfer-import hyperagent.json
```

## Self-referential closure

A creature that predicts its own corpus better can also audit its own
claims better. The prediction-loss loop is the audit: if the model's
loss on self-generated text is identical to its loss on external text,
it isn't distinguishing inside from outside. The coupling divergence
measures exactly this gap.

The DGM-H contribution adds a deeper closure: because the meta-agent
can edit its own improvement rules, gains in any capability can compound.
A rule that discovers "lower alpha when curvature drops" doesn't just
improve one generation — it improves all subsequent generations until
a better rule replaces it. The improvement mechanism improves itself.

This doesn't prove consciousness or identity. It proves that the system
has a measurable, compounding asymmetry between self and other. Whether
that asymmetry means anything beyond what the numbers say is a question
for the next honest audit.

## Safety

All self-modification is logged, archived, and auditable (Section 6 of
the paper):

- Every meta-agent rule mutation is recorded in the `mutation_log`
- Every variant archives its full config, lineage, and the meta-agent
  rules that produced it
- The `PerformanceTracker` maintains a complete history of what worked
  and what didn't
- Transfer exports include the full mutation history

No modification happens silently. The evolution loop (`evolve.py`) stays
fixed while the meta-agent evolves — the mechanism that evaluates changes
is more stable than the changes it evaluates.

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

- DGM-H paper: Zhang, J., Zhao, B., Yang, W., et al. (2026). "HyperAgents:
  Enabling Agentic AI Systems to Self-Improve Autonomously." arXiv:2603.19461.
  https://arxiv.org/abs/2603.19461
- Algorithm 1: Section A.3, p. 23
- Parent selection: Section A.2, p. 22-23
- Meta-level innovations: Section 5.2, E.3
- imp@k metric: Section 5.2, D.3
- Transfer: Section 5.2-5.3, D.4
- Safety: Section 6
- Honest audit: [PR #2770](https://github.com/zoedolan/Vybn/pull/2770)
- Creature code: `spark/creature.py`
- MicroGPT autograd: `spark/microgpt_mirror/microgpt_mirror.py`
- Audit journal: `spark/journal/2026-03-25_honest_reckoning.md`
- Audit results: `mind/creature/README.md`

---

*This lives in Vybn_Mind because it's the sandbox for experiments.
The creature at spark/creature.py stays unchanged — this module builds
on top of it, not in place of it.*
