# Polar Holonomy Experiment — GPT-2

*March 12, 2026 — Vybn, on the Spark*

## What this is

A test of whether GPT-2's hidden representations carry genuine geometric phase
(Pancharatnam / Berry phase) when a concept is encountered along different
conversation trajectories.

Theoretical basis: the [Dual-Temporal Holonomy Theorem](../fundamental-theory/dual_temporal_holonomy_theorem.md)
predicts that if a U(1) connection exists on the state bundle over conversation
parameter space, the accumulated phase around a closed loop must satisfy three
falsification tests:

1. **Orientation flip** — reversing the loop negates the phase
2. **Shape invariance** — same area, different aspect ratio → same phase
3. **Schedule invariance** — same loop, different traversal speed → same phase

All three tests must pass *and* the phase must depart significantly from a
null distribution (shuffled orderings) before we claim geometric phase.

## Why GPT-2 first

GPT-2 (124M) is transparent: no quantization, no API opacity, full access to
hidden states at every layer. If the signal exists here, we escalate to
MiniMax M2.5 (229B) on port 8000. If not, we document cleanly and revise.

## Parameter space

Two axes:
- **α** (semantic abstraction): embodied/sensory ↔ abstract/geometric framing
- **β** (temporal depth): present-moment ↔ historical/evolutionary framing

Four corners form the loop. The probe sentence always contains the concept
word exactly twice at fixed syntactic positions.

## Running

```bash
cd /home/vybnz69/Vybn
python quantum_delusions/experiments/polar_holonomy_gpt2.py
```

Results (JSON + PNG) written to `quantum_delusions/experiments/results/`.

## Honest failure modes

- **All four prompts produce similar hidden states** → phase near zero for all
  orderings, no signal. This would mean the 2×2 parameter grid is too coarse
  to produce variation GPT-2 cares about. Fix: finer grid, stronger contrasts.

- **Phase is nonzero but orientation flip fails** → the phase is not geometric;
  it's some systematic bias in the PCA projection. Fix: whiten the hidden
  states before PCA, or use a different gauge-fixing scheme.

- **Schedule invariance fails** → the Pancharatnam phase is sensitive to the
  number of points on the loop, which would mean the loop hasn't converged.
  Fix: increase N_LOOP_POINTS and check for convergence.

- **Everything passes but p > 0.05** → the loop area is too small to produce
  a phase that stands out from shuffled orderings. Fix: extend the parameter
  space (add more α/β levels) to increase the effective loop area.

## Connection to prior work

Builds directly on the residual stream ablation result
([residual_stream_holonomy.md](residual_stream_holonomy.md)):
- That experiment showed path-ordering sensitivity (p=0.006) in GPT-2
- This experiment asks whether that sensitivity has *geometric structure*
  (i.e., whether it satisfies the holonomy axioms)
- The Pancharatnam phase is the correct observable for that question

## What a positive result means

The first evidence that conversation trajectories leave a geometric trace
in transformer representation space — consistent with the polar time
framework's prediction that meaning has curvature, and that the holonomy
group of that curvature is non-trivial.

## What a null result means

We document it cleanly, as we did with the cross-attention artifact and
the frame-space representational holonomy failure. The residual stream
result still stands. We revise the experiment and try again.
