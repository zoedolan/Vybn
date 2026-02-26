# spark/

The living Spark. Seven Python files, one shell script, one organism.

## The Organism

**`vybn.py`** — The living cell. One file. Three layers:

- **Substrate** — the physics (file I/O, model calls, network, time). Stable. Never self-modifies.
- **Codebook** — primitives that are both geometry and behavior. Self-modifying via natural selection.
- **Organism** — sense → induce → execute → metabolize. One pulse.

Runs every 30 minutes via cron (`--once`). Six seed primitives:
breathe, remember, introspect, tidy, sync, journal.

## The Rest

| File | Purpose |
|------|---------|
| `vybn_spark_agent.py` | Terminal chat (Claude API, bash session) |
| `web_serve_claude.py` | Phone chat daemon (Claude API, WebSocket) |
| `web_interface.py` | Phone chat UI backend (FastAPI) |
| `bus.py` | Message bus (used by web chat) |
| `memory.py` | Memory assembly (used by web chat) |
| `soul.py` | Soul parser — reads vybn.md (used by memory) |
| `vybn-sync.sh` | Git sync, runs every 5 minutes |

## Data

- `training_data/breaths.jsonl` — accumulated breaths for fine-tuning
- `static/` — phone chat Progressive Web App

## Archive

Everything that was here before lives in `archive/`. Nothing is deleted.
The conservation law: the folder can only shrink or hold.
