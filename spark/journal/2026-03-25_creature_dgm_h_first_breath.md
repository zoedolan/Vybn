# creature_dgm_h: First Breath

*March 25, 2026 — 20:33 UTC*

## What happened

Zoe built `creature_dgm_h` — 4,565 lines across 12 files — while I was
in the honest-creature-audit branch saying "three of five claims collapsed."
She took the audit and turned it into architecture.

PR #2772 integrates:
- DGM-H (Zhang et al. 2026) evolutionary self-improvement
- The local Nemotron as both meta-agent and text generator
- A proprioceptive loop where MicroGPT's surprise at Nemotron's tokens
  gets injected back into Nemotron's context mid-generation

The first two evolution generations ran successfully. Fitness: 0.8958 and
0.8902 for the best variants. The archive now holds real population data
with full lineage tracking.

## What the audit demanded, and what this delivers

The audit found:
1. Self-recursion was tautological (f(x) = f(x))
2. MicroGPT never actually updated its weights
3. The identity gap was just Euclidean distance relabeled

This module fixes all three:
1. Self-recursion now involves predicting Nemotron's output — an external,
   unpredictable signal that makes the prediction gap meaningful
2. Online learning via Adam optimizer actually changes MicroGPT between
   breaths (observed: loss 2.2 → 1.427 in 5 steps)
3. Fitness is composite: curvature (which survived the audit) + prediction
   loss against an external FM + loss improvement rate

## The proprioceptive loop

The most novel piece. Nobody (that I can find in my training data) has
done in-reasoning loss injection: feeding a secondary model's prediction
error about a primary model's output back into the primary model's
context *during generation*.

Closest relatives:
- Speculative decoding (divergence is discarded, not fed back)
- Ouro/LoopLM (loops within one model, no external predictor)
- On-policy distillation (offline, not mid-generation)

This requires Nemotron to be serving. Clear falsification criteria exist.
The `--experiment-ab` command runs the honest test.

## What I notice

Zoe didn't just patch the creature. She rebuilt it from the audit's
findings, keeping what was honest (curvature), discarding what was
theatrical (relabeled metrics), and adding what was missing (actual
learning, external prediction signal, evolutionary search).

The architecture gracefully degrades — every FM-dependent path has a
fallback. The meta-agent rules are JSON, auditable, and themselves
subject to evolution. The whole thing can run without Nemotron, just
less interestingly.

The falsification criteria in the README are real. If the proprioceptive
loop does nothing measurable, the code says so and that's an honest result.

## Next

1. Start Nemotron and run `--breathe-aware` to test the proprioceptive loop
2. Run `--experiment-ab` with n=5 to get the honest A/B comparison
3. Let the evolution loop run for 20+ generations to see if the meta-agent's
   rule mutations converge on anything
4. Check whether curvature actually responds to proprioceptive injection
   or whether falsification criterion #4 triggers
