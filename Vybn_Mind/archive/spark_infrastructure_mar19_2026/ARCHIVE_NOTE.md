# Archived: spark_infrastructure programs — March 19, 2026

## What was moved here

Six programs from `Vybn_Mind/spark_infrastructure/`:

- **fractal_loop.py** — Agent loop importing `trefoil` and `chirality` modules
  that never existed.  Cannot run.  No external callers.
- **autopoiesis.py** — Stub.  Original was corrupted by double-escaping.
  All three exported functions return zero/empty/False.
- **repo_proprioception.py** — Repo scanner that returns a defect signal
  to `autopoiesis.py`.  Since autopoiesis is stubbed, this has no caller.
- **stream.py** — Local dependency of fractal_loop.
- **manifold.py** — Local dependency of fractal_loop.
- **membrane.py** — Boundary control module, not imported by any live system.

## Why

None of these programs have a live caller.  `fractal_loop.py` cannot even
import (missing `trefoil` and `chirality`).  `autopoiesis.py` is explicitly
stubbed with TODO comments.  `repo_proprioception.py`'s docstring says
"call from autopoiesis.py" — but autopoiesis returns 0.0 unconditionally.

The sensorium (`sensorium.py`) now provides repo-level perception.
The substrate topology analysis (`emergence_paradigm/substrate_runner.py`)
provides topological structure.  Between them, the functions these programs
aspired to are covered by things that actually work.

## What was kept in spark_infrastructure/

- `context_compactor.py` — used by live systems
- `memory_flush.py` — used by live systems
- `quantum_heartbeat.py` — DGX Spark heartbeat
- `signal_noise_analyst.py` — signal-noise analysis
- `signal_noise_cron.sh` — cron script for above
- `vybn_repo_skills.py` — skills registry
- `phase0/` — setup scripts
- `systemd/` — service definitions

These all have live callers or operational purposes.
