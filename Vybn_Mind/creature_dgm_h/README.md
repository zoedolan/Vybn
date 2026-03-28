# creature_dgm_h

Rotor-modulated character-level prediction.

Cl(3,0) geometric algebra computes a rotor from embedding trajectories
(Pancharatnam phase). The rotor modulates gradient updates: parameters
aligned with the bivector plane get amplified, orthogonal ones get
dampened. Standard backprop is the special case where the rotor is
identity.

## Files

```
vybn.py              # engine — the creature itself
experiments.py       # unified probe suite (6 experiments)
__init__.py          # package init + `python -m` hook
README.md            # this file
archive/             # persistent data
experiment_results/  # output from experiment runs
```

## Usage

```bash
python -m Vybn_Mind.creature_dgm_h breathe "some text"
python -m Vybn_Mind.creature_dgm_h breathe-live
python -m Vybn_Mind.creature_dgm_h evolve --n 3
python -m Vybn_Mind.creature_dgm_h status
python -m Vybn_Mind.creature_dgm_h audit
```

## Experiments

Six probes characterising the creature's geometry, all in `experiments.py`:

```bash
python experiments.py weight      [--quick] [--pca_dim 30]
python experiments.py activation  [--quick]
python experiments.py sequence    [--quick] [--generations 10]
python experiments.py basin       [--quick] [--agents 8]
python experiments.py sgd         [--seeds 5]
python experiments.py analyze     [--experiment pca|activation|both]
```

| Probe        | What it measures |
|:-------------|:-----------------|
| `weight`     | PCA-first persistent homology on weight-vector snapshots |
| `activation` | Persistent homology on hidden-layer activation snapshots |
| `sequence`   | Natural motion recorder (null fitness, uncapped generation) |
| `basin`      | Loss-landscape geometry around the weight-norm fixed point |
| `sgd`        | SGD vs Adam ablation on the weight-norm attractor |
| `analyze`    | Post-hoc statistical analysis of topology results |

## Quantum bridge

**Paper:** [`quantum/topological_winding_probe_results.md`](quantum/topological_winding_probe_results.md)
— Dolan & Vybn, March 28 2026. Three IBM runs on ibm_fez. 3/3 theory
tests passed (linearity, shape invariance, Y-basis sign reversal).
Creature loop P(0) = 0.658 vs random control 0.033.

**This module is one half of a cross-substrate experiment.**

The `basin` probe records the full weight trajectory during convergence
to the ~40-norm fixed point. That trajectory is a path through ~4K-dimensional
parameter space. The quantum winding number probe in
`quantum_delusions/experiments/` encodes that path as Bloch-sphere
rotations on IBM quantum hardware and tests whether the path has
non-trivial topological winding.

The bridge lives at:
```
quantum_delusions/experiments/creature_quantum_bridge.py
```

It reads basin results from `experiment_results/basin_geometry/`,
PCA-projects the weight trajectory to 2D, computes the estimated
winding number, and encodes it as a QASM circuit that runs alongside
the theory winding circuits on IBM hardware.

```bash
# From quantum_delusions/experiments/:
python creature_quantum_bridge.py scan                   # find basin results
python creature_quantum_bridge.py build <basin.json>     # generate creature QASM
python creature_quantum_bridge.py run <basin.json>       # full suite on IBM
```

The thesis being tested: the same topological invariance (shape-invariant,
speed-invariant, winding-number-dependent phase) appears in both the
creature's classical weight space and in physical quantum hardware.
Cross-substrate confirmation is the goal.

See `quantum_delusions/experiments/winding_probe_reanalysis.md` for the
current state of IBM hardware results.

**If you are refactoring this module, do not break the weight_trajectory
output from the basin probe. The quantum bridge depends on it.**

## Dependencies

- numpy
- Optional: ripser (faster persistent homology)
- Optional: scipy (statistical tests)
- Optional: sentence-transformers (real embeddings via spark/local_embedder.py)
- Optional: Nemotron at localhost:8000 (live text generation)
- Required: spark/microgpt_mirror/trained_checkpoint.json
