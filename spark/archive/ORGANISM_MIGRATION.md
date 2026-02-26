# Organism Migration — 2026-02-25

## What happened

32 living files compressed to 8 (7 Python + 1 shell).
spark/vybn.py (507 lines) replaces ~2,762 lines across 10+ files.

## What was archived

### organism_replaced/
Files directly superseded by vybn.py:
- `cell.py` (170 lines) → `_breathe` primitive
- `synapse.py` (181 lines) → deposit logic in organism
- `z_listener.py` (138 lines) → organism runs as daemon eventually
- `transcript.py` (123 lines) → traces list in organism
- `fine_tune_vybn.py` (622 lines) → dormant, future capability
- `harvest_training_data.py` (753 lines) → dormant
- `merge_lora_hf.py` (111 lines) → dormant
- `retrain_cycle.py` (380 lines) → dormant
- `build_modelfile.py` (151 lines) → dormant
- `run_training_safely.sh` — dormant
- `safe_pull.sh` — dormant
- `config.yaml` — unused config
- `evolving_canvas.md` — stale doc
- `meta_process.md` — stale doc
- `OPENCLAW_INTEGRATION.md` — stale doc
- `state_summary.md` — stale doc
- `web_interface.py.bak` — backup
- `.vybn_thermodynamics` — stale

### organism_replaced/lingua/
The v3 lingua, superseded by the organism's evolutionary codebook:
- `breathe_lingua.py` (113 lines)
- `vybn_lingua_v3.py` (998 lines)
- `__init__.py`, `README.md`

Note: `living_lingua_v3.pt` (250 cycles) preserved in Vybn_Mind/lingua/

### skills.d_archived/
18 skill files. None were imported by any running process.
The organism's primitives replace this mechanism.

### sentinel/
Sentinel daily cycle. Was already broken (daily_cycle.sh missing).

### scripts/, fine_tune_output/, graph_data/, unsloth_compiled_cache/
Dormant infrastructure. Preserved for when fine-tuning activates.

## What survived

7 Python files + 1 shell script:
- `vybn.py` — the organism (NEW)
- `vybn_spark_agent.py` — terminal chat
- `web_serve_claude.py` — phone chat
- `web_interface.py` — phone chat UI
- `bus.py`, `memory.py`, `soul.py` — web chat deps
- `vybn-sync.sh` — git sync

## Cron

v6 (7 lines, 4 jobs) → v7 (2 lines, 2 jobs):
- Git sync every 5 min
- Organism pulse every 30 min
