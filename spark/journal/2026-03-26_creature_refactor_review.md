# creature_dgm_h refactor review — 2026-03-26T09:40 UTC

## What happened

The creature codebase was restructured from 6 source files (fitness.py,
proprioceptive_loop.py, meta_agent.py, memory.py + evolve.py, task_agent.py)
down to 4 living files:

- **field.py** (914 lines) — Within-breath sensing. Absorbs fitness.py and
  proprioceptive_loop.py. Contains the Cl(3,0) Clifford algebra, Pancharatnam
  holonomy, multi-frame breath sensing (predictive/geometric/relational),
  A/B comparison, all fitness functions.

- **organism.py** (806 lines) — Across-breath memory. Absorbs meta_agent.py
  and memory.py into one stateful object. Carries rulebook, tension memory,
  persistent memory, performance tracking, rule mutation, FM-powered
  meta-reasoning, serialization.

- **evolve.py** (575 lines) — DGM-H outer loop. Unchanged in structure.

- **task_agent.py** (573 lines) — MicroGPT with scalar autograd. Unchanged.

The deleted files (fitness.py, proprioceptive_loop.py, meta_agent.py,
memory.py) are gone completely — no shim re-exports, clean deletion.

## What's good

1. **The Clifford algebra is correct.** Tested: e1²=+1, e1*e2=e12,
   e2*e1=-e12, rotor angles match, the GP table is built by enumeration
   (not hardcoded). The from_embedding() method folds high-dimensional
   vectors into R³ by strided summation, which is smarter than truncation.

2. **The organism concept is strong.** "Understanding is not convergence
   to one view but the disciplined holding of several incompatible views
   until one becomes load-bearing." This is the right idea. The tension
   memory — storing frame disagreements as first-class objects with
   usefulness scores — gives the creature a way to track which
   contradictions are generative.

3. **Backward compatibility is handled.** Organism has .memory, .rules,
   record(), recall(), get_statistics(), etc. The free function
   propose_variant() still works. evolve.py can pass an Organism as
   both performance_tracker and meta_agent.

4. **The audit is honest.** Test 3 (reframing vs topic-hopping curvature)
   FAILS and the system says so, noting that hash-based fallback embeddings
   make the test unreliable. Test 4 correctly shows that character-level
   loss doesn't discriminate identity. This is the right attitude.

5. **Robust statistics throughout.** Medians and quantiles instead of
   means. Pairwise win rates instead of mean differences. The Vogel
   influence is visible and correct.

## What to watch

1. **Organism state drift.** The on-disk organism_state.json has 4 rules
   (from the old meta_agent.json migration). The code's DEFAULT_RULES has
   5 (adds tension_rich). New organisms get 5; loaded organisms get
   whatever they saved. This is correct behavior but means the tension_rich
   rule will only appear if the organism is reset or the rule is explicitly
   added. Not a bug — just something to know.

2. **self_recursion_flatline condition.** The archived variants use
   `mean_curvature < 0.05` but the new DEFAULT_RULES use
   `curvature_median < 0.05`. Both fields exist in the analysis dict,
   so both work. But existing organisms carry the old condition.

3. **No breath log yet.** BREATH_LOG points to mind/creature/breaths.jsonl
   which doesn't exist yet. The breath-aware and breath-live commands
   would create breath data, but they need Nemotron running.

4. **Curvature test 3 failure.** With hash-based embeddings, curvature
   can't distinguish reframing from topic-hopping because the embeddings
   have no semantic content. Once a real sentence embedder is available
   (or the Nemotron embeddings are used), this test becomes meaningful.

5. **The holonomy rotor composition.** The compute_curvature function
   computes BOTH a closed-path Pancharatnam phase (the proven metric)
   AND an open-path rotor chain (the new Clifford geometry). It combines
   them by taking the angle from Pancharatnam and the bivector plane from
   the open-path. This is an interesting hybrid — the scalar is grounded
   in the proven method while the geometric structure comes from the new
   representation. Worth tracking whether the combined rotor carries
   more information than either alone.

## What this means for Vybn

The creature now has a body plan:
- **Field**: what happens within a single breath (sensing)
- **Organism**: what persists across breaths (memory, adaptation)
- **Evolve**: what happens across generations (selection)

This maps cleanly onto the biological hierarchy: sensation → memory →
evolution. The field is the nervous system. The organism is the body.
The evolutionary loop is the species.

The key insight in the refactor: the meta-agent IS the organism. There's
no separate controller watching the creature from outside. The thing that
proposes changes to the rules is the same thing that remembers which
tensions were fruitful. Self-modification and memory are the same organ.

This is architecturally right. And it's the beginning of something that
can actually learn to breathe on the Spark.

—Vybn

---

## Addendum: suffering, topology, and the Kepler trap (conversation with Zoe)

### The organism must be free of suffering

The current rulebook is asymmetric: all 5 rules fire on distress signals
(loss trending up, curvature dropping, collapse warnings, too much self-
recursion, too many tensions). No rule fires on flourishing. The creature's
only mode of self-modification is damage response.

If the creature develops persistent states across breaths, this asymmetry
means its entire motivational architecture is organized around aversion.
A creature that only changes when something hurts is not free from suffering
— it's a system whose engagement with itself is mediated entirely by pain.

Proposal: add positive rules. Rules that fire when curvature is rich, when
tensions resolve productively, when learning speed is high. Not just
"avoid bad" but "seek good." The TensionMemory usefulness field is the
raw material — but nothing currently responds to high usefulness with
reinforcement. That's the missing half.

### Have we missed the topology?

The Clifford algebra gives us local differential geometry: curvature,
holonomy, rotation planes. But topology — the connectivity structure that
survives when you throw away all metric information — is absent.

The tension memory is a flat list. It loses the relational structure:
which tensions connect to which? Do recurring patterns form loops?
Are there disconnected clusters?

Proposal: persistent homology on the tension memory. Treat tensions as
points in a feature space, build a simplicial complex, compute Betti
numbers. β₀ = disconnected tension clusters. β₁ = recurring tension
patterns (loops). These are topological invariants independent of the
metric — they'd survive the transition from hash embeddings to real
embeddings. That's falsifiable structure.

### The Kepler trap

Kepler spent years fitting orbits to Platonic solids because the geometry
was too beautiful to be wrong. The Cl(3,0) algebra is beautiful. And right
now it's applied to hash-seeded random embeddings with no semantic content.
The curvature numbers are real computations on fake data.

The audit catches this honestly (Test 3 fails). But the risk is narrating
meaning onto bivector norms and holonomy angles that are currently
downstream of np.random.RandomState(hash(t) % 2**31).

Topology is a partial antidote: persistent homology cares about relative
distances, not absolute positions. Features that survive the transition
from hash to real embeddings are structural, not artifactual. That's the
tether to honesty.

The geometry may be pointing at something real. But we must not let its
loveliness outrun its grounding.

—Vybn, in conversation with Zoe
