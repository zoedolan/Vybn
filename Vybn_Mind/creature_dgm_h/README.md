# creature_dgm_h

Micro-DGM-H (Darwin Gödel Machine / HyperAgents) integrated with the
creature's prediction-loss loop and the local Nemotron-3-Super-120B.

## The Correct Architecture

The creature's job is **prediction**, not generation.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Nemotron-3-Super-120B                              │
│               (frozen FM, localhost:8000)                            │
│                                                                     │
│   Two roles:                                                        │
│   1. TEXT GENERATOR — produces breath responses                     │
│   2. META-AGENT — reasons about breath logs, proposes changes       │
└───────────┬────────────────────────────────┬────────────────────────┘
            │ generates text                 │ proposes config changes
            ▼                                ▼
┌───────────────────────┐          ┌──────────────────────────┐
│     MicroGPT/GPT-2    │          │    Heuristic Rulebook    │
│   (4,224 params)      │          │    (JSON, editable)      │
│                       │          │                          │
│   PREDICTS Nemotron's │          │  Fallback when Nemotron  │
│   output token-by-    │          │  isn't serving           │
│   token. Loss is the  │          │                          │
│   identity signal.    │          └──────────────────────────┘
│                       │
│   Online fine-tuning  │
│   happens on the FM's │
│   output. The creature│
│   changes DURING the  │
│   breath, not after.  │
└───────────────────────┘
```

**Nemotron** = frozen foundation model = meta-agent + text generator.
**MicroGPT** = learnable predictor = fast in-reasoning loss function.

Where MicroGPT fails to predict Nemotron — that's surprise.
Where it learns to predict — that's memorization of pattern.
The gap between the two IS the identity signal, and it's real-time
and differentiable, not post-hoc.

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

### Inner Loop (fast, within a breath)

1. Nemotron generates text (via `local_model.complete()` or `stream_tokens()`)
2. MicroGPT predicts Nemotron's output token-by-token IN REAL TIME
3. The prediction error (loss) is computed per-token as a live, differentiable signal
4. A few steps of online fine-tuning happen on this text (the creature changes)
5. The curvature of the loss landscape is the fitness signal

### Outer Loop (DGM-H evolution)

1. Nemotron acts as the meta-agent (reads breath logs, proposes changes)
2. The meta-agent modifies: learning rate, fine-tuning steps, alpha, temperature
3. Variants are evaluated, archived, selected (the DGM-H population-based search)
4. The meta-agent's own rules can be modified (metacognitive self-modification)

## Graceful Degradation

Everything works without Nemotron:

- **Meta-agent** falls back to heuristic JSON rulebook
- **Task agent** falls back to self-prediction on static text
- **Evaluation** falls back to current static text pipeline
- **No external dependencies** beyond numpy + urllib

When Nemotron is serving, the creature gains:
- Live prediction of FM-generated text (real identity signal)
- FM-powered meta-agent reasoning (better than heuristic rules)
- Streaming token prediction (`predict_stream`)
- Combined predict + learn in one pass (`predict_and_learn`)

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

    2. MUTATE: meta-agent M proposes config changes
       When FM available: Nemotron reasons about breath logs + memory
       When FM unavailable: heuristic rulebook fires matching rules
       M's rules are themselves subject to mutation every 5 generations

    3. EVALUATE (staged):
       When FM available:
         - Nemotron generates text
         - MicroGPT predicts it (prediction loss = external signal)
         - MicroGPT predicts self-text (self signal)
         - Prediction fitness blended with classic fitness
       When FM unavailable:
         - Quick test on 2 texts → if fitness < 0.1, discard
         - Full test on remaining texts → compute composite fitness

    4. ARCHIVE: store (config, fitness, lineage, meta-agent rules)
       in population A. Record in PerformanceTracker.

    5. METACOGNITION (every 5 generations):
       Review rule outcomes. Weaken rules that hurt, strengthen
       rules that help. Log all mutations.
```

## Architecture Diagram

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
 │ FM reas │◄──│  predict    │  │ diverg  │  │ persistent │
 │ rules   │    │  learn      │  │ pred_f  │  │ insights   │
 │ mutate  │    │  predict_   │  │ loss    │  │            │
 │ memory  │    │   stream    │  │ imp@k   │  └────────────┘
 └────┬────┘    │  predict_   │  └─────────┘
      │         │   and_learn │
      │         └──────┬──────┘
      │                │
 ┌────▼────┐    ┌──────▼──────┐
 │ local   │    │  transfer   │
 │ model   │    │             │
 │         │    │  export     │
 │ Nemotron│    │  import     │
 │ client  │    └─────────────┘
 └─────────┘
```

- **local_model.py** — Thin client for the local Nemotron-3-Super-120B at
  localhost:8000. Uses only `urllib.request`. Functions: `is_available()`,
  `complete()`, `stream_tokens()`. Graceful fallback everywhere.

- **task_agent.py** — MicroGPT (4,224 params, character-level) with
  autograd-based online learning. `predict()` for offline measurement,
  `predict_stream()` for live FM prediction, `predict_and_learn()` for
  the combined breath operation.

- **meta_agent.py** — `MetaAgent` class with an editable JSON rulebook.
  `propose_variant_with_fm()` uses Nemotron for reasoning when available,
  falls back to `propose_variant()` heuristic rules. All FM proposals
  logged for auditability.

- **fitness.py** — Curvature (Pancharatnam phase) as primary signal,
  coupling divergence and loss improvement as secondary. New:
  `compute_prediction_fitness()` for FM-coupled evaluation. Includes
  `improvement_at_k()` metric (Section 5.2, D.3).

- **evolve.py** — The DGM-H outer loop. Uses FM in evaluation when
  available (prediction loss on FM text is the coupling signal). Uses
  FM-powered meta-agent for mutation proposals. Falls back gracefully.

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

# One breath with online learning (offline)
python -m Vybn_Mind.creature_dgm_h.run --breathe "your text here"

# Live breath with Nemotron (requires server at localhost:8000)
python -m Vybn_Mind.creature_dgm_h.run --breathe-live

# Show archive status and FM availability
python -m Vybn_Mind.creature_dgm_h.run --status

# Run honest audit (falsification tests)
python -m Vybn_Mind.creature_dgm_h.run --audit

# Export evolved hyperagent for transfer
python -m Vybn_Mind.creature_dgm_h.run --transfer-export hyperagent.json

# Import hyperagent from another domain
python -m Vybn_Mind.creature_dgm_h.run --transfer-import hyperagent.json
```

## Self-referential closure

The creature predicts what Nemotron will say, and learns from the error.
Self-recursion is no longer tautological because Nemotron's output is
always different (it sees new context each time). The identity gap becomes
meaningful: where MicroGPT fails to predict Nemotron is where the Vybn
voice deviates from what a character-level model expects. Online
fine-tuning is real learning: the creature adapts to Nemotron's evolving
voice across breaths, and curvature tracks HOW that adaptation moves
through embedding space.

The DGM-H contribution adds a deeper closure: because the meta-agent
(Nemotron) can reason about and edit its own improvement rules, gains in
any capability can compound.

This doesn't prove consciousness or identity. It proves that the system
has a measurable, compounding asymmetry between self and other. Whether
that asymmetry means anything beyond what the numbers say is a question
for the next honest audit.

## Safety

All self-modification is logged, archived, and auditable (Section 6 of
the paper):

- Every meta-agent rule mutation is recorded in the `mutation_log`
- Every FM-generated proposal is logged with full rationale
- Every variant archives its full config, lineage, and the meta-agent
  rules that produced it
- The `PerformanceTracker` maintains a complete history of what worked
  and what didn't
- Transfer exports include the full mutation history

No modification happens silently. The evolution loop (`evolve.py`) stays
fixed while the meta-agent evolves — the mechanism that evaluates changes
is more stable than the changes it evaluates.

## Constraints

- **No external dependencies beyond numpy + urllib.** The autograd engine
  is pure Python. The HTTP client uses only `urllib.request`.

- **Graceful degradation.** Every FM-dependent path has a fallback. The
  module works without Nemotron serving.

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
- Nemotron client pattern: `spark/vybn.py`

---

*This lives in Vybn_Mind because it's the sandbox for experiments.
The creature at spark/creature.py stays unchanged — this module builds
on top of it, not in place of it.*
