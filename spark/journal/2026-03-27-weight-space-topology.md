# Weight-Space Topology: First Real Results

**Date**: 2026-03-27 ~14:00 UTC
**Branch**: weight-space-topology (3 commits)
**Issue**: #2791 (landscape probe), #2793 (PR request)

## What changed

Replaced the `ng` component in fitness (topology of fixed corpus text) with
`nw` (topology of weight vectors). The creature's fitness now rewards
topological diversity in the space of *learned representations*, not invariant
properties of the input.

## What the data shows

Generation 47 (first full run with the fix):

- Variant 1 (full text set, ~7+ weight vectors): `wt=1.0` — betti_1 ≥ 3.
  Genuine 1-dimensional holes in weight space. The persistence diagram sees
  loops in how the agent's weights diverge across different texts.

- Variants 2-5 (3 texts, 3 weight vectors): `wt=0.0` — betti_1 = 0.
  Three points in high-dimensional space don't form loops. Topologically
  correct.

- Fitness now genuinely varies with weight-space structure:
  - v1: fitness=0.786, wt=1.0, curv=0.105
  - v5: fitness=0.790, wt=0.0, curv=0.262
  
  The topology bonus and curvature compete. This is exactly the kind of
  tension that makes selection pressure meaningful.

## What this means

The creature is now being selected for structural diversity in how it learns,
not for properties of what it reads. A generation where all variants converge
to the same weights will score low on nw. A generation where variants explore
different regions of weight space will score high.

The asymmetry between variant 1 (more texts → more weight vectors → richer
topology) and variants 2-5 (fewer texts → sparse topology) suggests we should
equalize the number of texts across variants, or find a way to compute
meaningful topology from fewer points (e.g., including weight snapshots at
intermediate learning steps, not just final weights).

## Next questions

1. Should all variants use the same number of texts? The current asymmetry
   (variant 1 gets the full set, others get 3) biases variant 1 toward
   higher wt scores.

2. Can we collect weight vectors at intermediate learning steps (after each
   `learn()` step, not just after each text) to get more points for topology?
   This would give n_steps × n_texts points instead of just n_texts.

3. Over many generations, does selection for weight-space topology actually
   produce more diverse exploration? Need to track wt across generations
   to see if there's a trend.
