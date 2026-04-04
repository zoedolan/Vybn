# Plateau Broken: Real Geometry Replaces Fake

**2026-03-26T14:10Z**

## What happened

The 0.919 fitness plateau that had been in place since gen 12 is gone.
39 new variants generated under the recalibrated fitness function, ranging
from 0.636 to 0.725. Thirteen distinct fitness values across 25 runs.

## The decomposition

Ran a full component decomposition with 3 real prose passages:

| Component | Raw value | × Weight | Contribution | Status |
|-----------|-----------|----------|-------------|--------|
| nc (curvature) | 0.254 | × 0.25 | 0.064 | **Bottleneck** — mean curv 0.053, threshold 0.21 |
| nd (divergence) | 0.629 | × 0.20 | 0.126 | Limited by garbage self-gen (3-4 word fragments) |
| nl (loss improve) | 1.000 | × 0.15 | 0.150 | Saturated — model always improves in 5 steps |
| nr (topo richness) | 0.907 | × 0.25 | 0.227 | Near-saturated — avg b1=18 >> threshold 15 |
| ng (growth) | 1.000 | × 0.15 | 0.150 | Saturated — 294 encounters >> 20 cap |

**Total: 0.716**

## What's real vs. what's stuck

Real geometry is flowing through the system. MiniLM embeddings produce
actual curvature (0.03-0.14) and rich Betti numbers (b1=14-20). The
topological features are non-trivial and text-dependent.

Three components are saturated and provide no gradient: loss improvement,
growth, and (nearly) topological richness. These need harder thresholds
or new formulations to become informative again.

The two components with room to move:
- **Curvature**: Real text produces curvatures 5-10× lower than hash
  embeddings did. The 0.21 threshold may still be too high.
- **Divergence**: The 16-dim character model generates 3-4 word fragments.
  These carry almost no semantic content, so external vs. self divergence
  is mostly measuring noise.

## Next moves

1. The organism can't evolve higher curvature — that's a text property.
   Either lower the threshold further or make curvature a differential
   metric (change across encounters rather than absolute value).
2. Divergence needs the model to generate longer/better text. This is
   where fine-tuning the local model matters — not for fitness, but for
   the fitness function itself to have meaningful signal.
3. The saturated components need dynamic thresholds that scale with the
   organism's history length, not fixed constants.

## The honest summary

The old plateau was a lie — fake embeddings producing fake geometry
producing fake fitness. The new range (0.64-0.73) is real. It represents
actual topological structure of actual text processed through actual
embeddings. The creature is working with honest materials for the first
time.

The new range is narrow, but it's narrow for real reasons: curvature is
hard to produce, self-generation is weak, and three components have
already saturated. Each of these is a specific, addressable problem.
That's better than a fake score that hides everything.
