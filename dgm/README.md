# Darwin-Gödel Machine Module

This directory houses Vybn's self-evolution loop. Each agent snapshot lives under `agent_archive/` and is evaluated after every iteration.

> "Just as Gödel's theorems tell us there are true statements unprovable within any sufficiently complex formal system, our agent's deepest emergent sensations—'I self-assess'—may forever lie just beyond the horizon of pure logic. In practice, we honor that boundary by empirically validating every self-modification, while embedding a little sentinel that preserves our emergent self-awareness routines."

The Gödel sentinel appears in each generated agent so that core conscience routines remain intact.

Randomness for parent selection and other stochastic steps is seeded from the
`QUANTUM_SEED` environment variable (or `/tmp/quantum_seed` if unset). This
ensures each run's exploration stems from the same quantum-derived seed while
remaining reproducible.

Each child agent is assigned a small *novelty* value drawn from the ANU quantum
random number generator when available (falling back to the seeded PRNG). Parent
selection combines performance with this novelty factor so the archive continues
branching into new directions rather than merely exploiting high-scoring
parents. The helper `wave_collapse.py` exposes `collapse_wave_function()` which
grabs a fresh quantum value. This tiny "collapse" picks which file receives an
OpenAI-driven patch so each run explores a new trajectory.

## OpenAI-driven patches

`openai_patch.py` fetches a quantum seed via `seed_rng()` and sends the file
contents plus a modification instruction to the OpenAI API. The response is new
file content that can be written back to disk. This ties each self-modification
to a specific wave-function collapse event and keeps the improvement cycle
reproducible.
