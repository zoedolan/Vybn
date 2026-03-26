# creature_dgm_h

The whole creature in one script: `vybn.py` (887 lines).

Implements the DGM-H outer loop from [Zhang et al. 2026](https://arxiv.org/abs/2603.19461) at minimal scale.

## Usage

```bash
python -m Vybn_Mind.creature_dgm_h breathe "some text"
python -m Vybn_Mind.creature_dgm_h breathe-live
python -m Vybn_Mind.creature_dgm_h evolve --n 3
python -m Vybn_Mind.creature_dgm_h status
python -m Vybn_Mind.creature_dgm_h audit
```

## One operation, three timescales

| Scale | What happens |
|-------|-------------|
| CHARACTER | predict the next character, learn from error |
| BREATH | predict a stream of text, accumulate the encounter rotor |
| GENERATION | select which hyperparameters survive |

The encounter is the same at every scale: meet text you didn't generate, try to predict it, fail in a geometric pattern. The pattern is a rotor in Cl(3,0).

## Real embeddings

`embed` tries `spark/local_embedder.py` (all-MiniLM-L6-v2, 384-dim, CPU). Falls back to hash vectors. The audit reports which path ran.

## Dependencies

- **numpy**
- **Optional:** sentence-transformers (for real embeddings)
- **Optional:** Nemotron server at `http://127.0.0.1:8000` (for live breaths)

## Checkpoint

Requires `spark/microgpt_mirror/trained_checkpoint.json`.
