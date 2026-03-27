# creature_dgm_h

Rotor-modulated character-level prediction. Single file: `vybn.py`.

Cl(3,0) geometric algebra computes a rotor from embedding trajectories
(Pancharatnam phase). The rotor modulates gradient updates: parameters
aligned with the bivector plane get amplified, orthogonal ones get
dampened. Standard backprop is the special case where the rotor is
identity.

## Usage

```bash
python -m Vybn_Mind.creature_dgm_h breathe "some text"
python -m Vybn_Mind.creature_dgm_h breathe-live
python -m Vybn_Mind.creature_dgm_h evolve --n 3
python -m Vybn_Mind.creature_dgm_h status
python -m Vybn_Mind.creature_dgm_h audit
```

## Topology experiments

Two approaches for measuring whether text selection affects the topological
structure of learning (superseding the original raw weight-space experiment,
which produced a uniform null result due to the curse of dimensionality):

### PCA-first persistence

Projects weight-vector snapshots to ~20 dimensions via PCA before computing
persistent homology. Concentrates variance along the learning trajectory.

```bash
python experiment_weight_topology.py              # full experiment
python experiment_weight_topology.py --quick      # smoke test
python experiment_weight_topology.py --pca_dim 30 # custom PCA dim
```

### Activation-space persistence

Captures hidden-layer activations (16-dim) during training using a fixed probe
sentence. Lower-dimensional and semantically tied to model behaviour.

```bash
python experiment_activation_topology.py          # full experiment
python experiment_activation_topology.py --quick  # smoke test
```

### Analysis (unified)

```bash
python experiment_analysis.py                         # both experiments
python experiment_analysis.py --experiment pca        # PCA only
python experiment_analysis.py --experiment activation # activation only
```

## Dependencies

- numpy
- Optional: ripser (faster persistent homology)
- Optional: scipy (statistical tests)
- Optional: sentence-transformers (real embeddings via spark/local_embedder.py)
- Optional: Nemotron at localhost:8000 (live text generation)
- Required: spark/microgpt_mirror/trained_checkpoint.json
