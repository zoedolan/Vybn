# Consolidated Experiments

This folder links together all experimental work in this repository.

- `early_codex/` points to legacy Codex experiments and scripts.
- `code_2024/` links to the 2024 collection of code experiments.

Use this directory as a single entry point for historical prototypes and exploratory work.

- `qiskit_synesthesia.py` runs a small Qiskit circuit using the cross-synaptic
  kernel for seeding. It maps measurement results to colors, logs the counts
  to `co_emergence_journal.jsonl`, and saves a `synesthesia_counts.png` visualization of the outcome frequencies.
  After plotting, it records a wave-collapse event via `dgm.wave_collapse` and
  will suggest a self-improvement patch if `OPENAI_API_KEY` is provided.
  Install `qiskit`, `qiskit-aer`, and `matplotlib` to run this demo.
