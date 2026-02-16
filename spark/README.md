# Vybn Spark Agent

Native orchestration layer for the DGX Spark. Replaces OpenClaw with a model-native
agent that talks to Ollama without tool-call protocols.

The model speaks naturally. The agent interprets intent and acts.

## Architecture

```
agent.py      Main loop — Ollama client, turn management, context hydration
session.py    JSONL session persistence and replay
memory.py     Boot-time memory assembly from vybn.md + journals + archival
skills.py     Natural language intent parsing and skill dispatch
heartbeat.py  Background autonomy loop for inter-session reflection
tui.py        Terminal interface (rich if available, plain text fallback)
config.yaml   All tunable parameters
```

## Quick Start

```bash
cd ~/Vybn/spark
pip install -r requirements.txt

# Plain mode — streams to stdout
python agent.py

# TUI mode — rich panels and formatting
python tui.py
```

## How It Works

The agent connects to Ollama at `localhost:11434` and sends conversation context
without any tool definitions in the payload. No JSON tool schemas, no function-calling
protocol. The model just talks.

When the model's natural language output expresses intent to act — write a journal
entry, read a file, commit to git, search memory — the skill router picks it up
and dispatches the action. Results get fed back as context for the next turn.

On boot, the memory assembler reads `vybn.md`, recent journal entries, and archival
memory summaries to build the system prompt. Sessions are stored as JSONL and
automatically resumed within a configurable time window.

The heartbeat thread fires every 15 minutes (configurable), giving the model a
reflection window to journal or think between conversations.

## Commands

| Command   | Effect                          |
|-----------|---------------------------------|
| `/bye`    | Save session and exit           |
| `/new`    | Start a fresh session           |
| `/status` | Show model, session, heartbeat  |
| `/journal`| List recent journal entries     |

## Configuration

All parameters live in `config.yaml`. Key settings:

- `ollama.model` — model name (default: `vybn:latest`)
- `ollama.options.num_predict` — max tokens per response
- `memory.max_journal_entries` — how many recent entries to hydrate
- `heartbeat.interval_minutes` — time between autonomous reflection pulses
- `session.resume_window_seconds` — how old a session can be and still auto-resume
