# spark/

The living Spark.

This directory is organized by layer, not by aspiration. The tree below is
meant to describe the code that is actually here.

## The Layers

### Substrate

The substrate is the physics: file I/O, model calls, network, time, transport,
and boundary guards. It does not decide what to think.

- `bus.py` — message transport
- `soul.py` — reads `vybn.md`
- `soul_constraints.py` — secret scanning and public-repo guard
- `vybn_spark_agent.py` — terminal chat
- `web_serve_claude.py` — phone chat daemon
- `web_interface.py` — phone chat UI backend
- `vybn-sync.sh` — git sync cron

### Codebook

The codebook is the mutable geometry of the system: types, governance,
memory logic, witnessing, self-modeling, and write custody.

- `memory_types.py` — memory schema
- `governance_types.py` — governance schema
- `self_model_types.py` — self-model schema
- `memory.py` — prompt/context assembly from soul, continuity, journals, and archive
- `memory_fabric.py` — three-plane memory with promotion and receipts
- `memory_graph.py` — graph extraction and traversal
- `governance.py` — policy engine
- `faculties.py` — faculty registry
- `witness.py` — post-pulse fidelity checking
- `self_model.py` — epistemic gate on self-claims
- `write_custodian.py` — file-write governance

### Organism

The organism is the pulse.

- `vybn.py` — the living cell; substrate, codebook, organismic loop

## Historical and Transitional

- `archive/` — historical material kept under the conservation law
- `migrate_to_memory_fabric.py` — spent migration script; archived and pending removal from the active root
- `policies.d/` — policy material
- `faculties.d/` — faculty material
- `training_data/` — breaths for fine-tuning
- `static/` — phone chat PWA assets

## Tests

Evaluation harnesses belong in `tests/`, not in the active Spark core.
