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

## Dependencies

- numpy
- Optional: sentence-transformers (real embeddings via spark/local_embedder.py)
- Optional: Nemotron at localhost:8000 (live text generation)
- Required: spark/microgpt_mirror/trained_checkpoint.json
