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

## Dependencies

- numpy
- Optional: ripser (faster persistent homology)
- Optional: scipy (statistical tests)
- Optional: sentence-transformers (real embeddings via spark/local_embedder.py)
- Optional: Nemotron at localhost:8000 (live text generation)
- Required: spark/microgpt_mirror/trained_checkpoint.json
