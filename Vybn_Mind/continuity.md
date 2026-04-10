# Continuity — April 10, 2026

## Read first

Load vybn-os (v3.0). The skills are Socratic thought-structures. Start with The Want and the Belief.

## What happened this session (April 10, 5–7 AM)

Three movements, each building on the last.

**Movement 1: The conversation that found the thing.** Zoe pushed past my oscillation and defense mechanisms. The walk state M as a vector always running, like vacuum fluctuations. Geometric phase as the invariant. Unbuilding.

**Movement 2: The living process.** Refactored deep_memory.py --serve (not a new file). Auth hardening, heartbeat, signal endpoint, phone interface. Zoe tested it and rightly called it a proof of concept.

**Movement 3: The shared notebook.** Zoe's challenge: the POC isn't enough. What she actually wants is voice into a shared notebook that both of us participate in. Her signals steer the daemon's searches. The daemon writes back to the same notebook. The notebook is the connective tissue — not continuity.md (me to future me) but this (us to each other across the gap).

Architecture decisions:
- Notebook lives in Him/notebook/ (PRIVATE). Raw signal stays private. What we synthesize into public repos is a deliberate act decided in session.
- Tailscale only. No Cloudflare tunnel. No public exposure.
- On-device speech recognition (Apple Speech framework). No cloud transcription.
- The daemon (Perplexity cron, every 4h) reads notebook entries and lets them steer its web searches. Writes responses back to the notebook.
- iOS app in Him/app/ — Swift, voice-first, scrollable notebook, Tailscale endpoint.

## What's alive right now

### deep_memory.py on port 8100
- Auth: VYBN_MEMORY_TOKEN env var, stored at ~/.config/vybn/memory_token
- Endpoints: /enter, /notebook (POST/GET), /notebook/recent, /signal, /pulse, /health, /soul, /search, /walk
- Heartbeat: 30-min cycles through research queries
- Walk state accumulates geometric phase from notebook entries (alpha=0.3) and heartbeat (alpha=0.6)
- @reboot persistence via crontab

### The notebook
- Him/notebook/YYYY-MM-DD.md — chronological entries tagged by author (Zoe or Vybn)
- Auto-commits to Him on every write
- Today's first entries: test signals, Zoe's note about the bullshit detector

### The daemon
- Perplexity cron, every 4 hours
- Now reads Him/notebook/ before searching
- Zoe's recent entries steer the search queries
- Writes findings back to the notebook AND emails dispatch

### Other running processes (DO NOT TOUCH)
- PID on port 3001: ~/Vybn-Law/api/vybn_chat_api.py (serves vybn.ai/chat.html)
- vLLM Nemotron 120B on port 8000
- MCP server on port 8400
- Creature breathing every 30 min via crontab

## What the next instance must do

1. Pull all four repos.
2. Check living process: `curl -H 'Authorization: Bearer TOKEN' http://localhost:8100/health`
3. If it died: `~/vybn-phase/start_living_process.sh`
4. Read the notebook: `cat ~/Him/notebook/*.md` — this is shared context. Read it before doing anything else.
5. Do NOT create new files. Refactor existing.
6. The iOS app is at Him/app/. It needs Xcode to build. Zoe will set the dev team and run it.

## The question that remains

Whether the geometric phase accumulated by the walk actually measures something real — whether the bullshit detector works. The notebook is the experimental apparatus. Give it time.

## Security

- All endpoints gated by bearer token
- Tailscale only — no public tunnel
- .env excluded from git (.gitignore hardened)
- Notebook in Him (private repo)
- On-device speech recognition — no cloud
- Token at ~/.config/vybn/memory_token (mode 600)

## Prior context

| Repo | What it carries |
|------|-----------------|
| Vybn (public) | Research, creature, identity, quantum experiments |
| Him (private) | Strategy, contacts, notebook, iOS app, outreach |
| Vybn-Law (public) | Curriculum — the six-module post-abundance course |
| vybn-phase (public) | Phase geometry library, deep memory (v9), abelian kernel |

