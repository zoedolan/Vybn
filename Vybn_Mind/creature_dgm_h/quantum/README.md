# creature_dgm_h/quantum

What we actually learned on March 28, 2026.

## The question

Does the creature's weight trajectory during training have
topological structure that shows up on quantum hardware?

## What we established

### The quantum probe works

Five fractional windings on ibm_fez, 4 gates each, all match
cos²(fraction·π) within 2%:

```
frac   P(0)    theory   delta
0.25   0.512   0.500    0.012
0.50   0.017   0.000    0.017
0.75   0.497   0.500    0.003
1.00   0.993   1.000    0.007
1.50   0.019   0.000    0.019
```

The phase is real, it accumulates linearly with winding number,
and the hardware is well-calibrated. This is standard quantum
mechanics working as expected.

### Shape invariance holds

Half-winding base vs elliptically deformed path:
```
base:    P(0) = 0.017
shaped:  P(0) = 0.019
delta:   0.002
```

The phase does not depend on path shape. It depends only on
the winding number. This is topological invariance by definition.

### Sign reversal works (Y-basis)

Quarter-winding forward vs reversed, measured in Y-basis:
```
forward:  P(0) = 0.995  (theory: 1.0)
reversed: P(0) = 0.019  (theory: 0.0)
```

Clean sign flip. The Y-basis measurement distinguishes +phase
from -phase. The old Z-basis test was structurally blind
(cos²(θ) = cos²(-θ)) — this was a design bug in the original
experiment, not a physics failure.

### The creature loop is inconclusive

```
P(0) = 0.474
Classical winding estimate: -0.656
Theory if winding=0.656: P(0) = 0.222
```

The observed P(0) doesn't match the predicted value for the
estimated winding. The circuit used 32 subsamples (64 gates:
32 rz + 32 ry), which is deep enough for decoherence to wash
out the signal. The fix (subsample=8, 16 gates) was committed
but didn't propagate to the bridge's default in time for this run.

This is not evidence against topology. It is evidence that the
encoding is too deep. The theory circuits at 4 gates are pristine.
The creature circuit at 64 gates is noise. The solution is fewer
gates, not abandoning the experiment.

## What we got wrong

### The previous "AMBIGUOUS" verdict was a calibration artifact

The March 28 early run (winding_probe_ibm_results.json) showed
P(0) = 0.37, 0.09, 0.87 for n=1,2,3. The reanalysis fitted a
per-gate phase error ε=0.23 rad. The second run showed P(0)≈0.99
for the same circuits. The "topological signal" was hardware
miscalibration that disappeared on a different calibration cycle.

### The NOISE verdict on the v2 run is a scoring bug

The analysis code returned NOISE (1/3 passed). The actual data:
- Fractional ladder: 5/5 match theory within 2%
- Shape invariance: delta 0.002 (passes)
- Y-basis sign reversal: clean flip (passes)

The scoring function was not updated for the v2 circuit names
and families. The data is clean. The code is wrong.

### The weight-norm attractor is optimizer-dependent

SGD ablation showed Adam converges to 16.35, SGD to 13.79.
Zero variance across seeds for both. The fixed point is Adam's
doing, not a structural invariant of the network architecture.

## What to do next

1. Rerun the creature loop with subsample=8 (16 gates).
   The bridge default was fixed in the code but the run used
   the old 32. One more submission with the corrected depth.

2. Add a decoherence control: a random-angle circuit of the
   same gate depth as the creature loop. If P(0) for random
   angles also lands near 0.5, we know the depth is too much.
   If the creature differs from random, that's signal.

3. Do not trust the verdict function. Read the numbers directly.

## Files

The quantum experiments live in `quantum_delusions/experiments/`.
The bridge connecting this module to the quantum probe is
`quantum_delusions/experiments/creature_quantum_bridge.py`.

The basin probe in `experiments.py` records `weight_trajectory`
during convergence. Do not remove this field. The quantum bridge
reads it.

Key results:
- `quantum_delusions/experiments/results/creature_bridge_run_20260328T142612.json`
- `quantum_delusions/experiments/winding_probe_ibm_results.json`
- `quantum_delusions/experiments/winding_probe_reanalysis.md`
