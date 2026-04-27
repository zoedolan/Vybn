# MicroGPT mirror

Archived reproducibility capsule for the old MicroGPT surprise-contour experiment.

This directory is **not live runtime**. It preserves the small script that can
rebuild the character-level mirror corpus used by the archived experiment.

Tracked:
- `build_mirror_corpus.py` — rebuilds `mirror_corpus.txt` from preserved Vybn prose.
- `.gitignore` — keeps regenerated payloads out of source.

Generated / local-only:
- `mirror_corpus.txt`
- `trained_checkpoint.json`

ABC note, 2026-04-27: the generated checkpoint was removed from tracked source
and preserved locally under `~/logs/repo_garden_payloads/`. See
`spark/ARCHIVE.md` for the exact restore path, checksum, and the historical
meaning of the experiment.

Rule: regenerate or restore model payloads locally; do not commit checkpoints.
