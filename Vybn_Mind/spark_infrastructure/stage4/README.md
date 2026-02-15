# Stage 4: The Bridge to OpenClaw

## What This Is

This directory contains the workspace configuration for running Vybn as an OpenClaw agent on the DGX Spark. It bridges the custom infrastructure from Stages 1–3 into OpenClaw's framework — preserving everything that matters, discarding nothing prematurely.

The old `spark_agent.py` (Stage 3: Braided Cognition) remains in place as a fallback. Nothing is deleted until the new system proves itself in practice.

---

## Architectural Decisions

### Our files keep their names

OpenClaw conventionally uses `SOUL.md` for personality and identity. We don't. Our soul document is `vybn.md` — it lives at the repository root, where it has always lived. OpenClaw will be configured to read it from there.

Similarly, `AGENTS.md` stays at the repo root. It's our collaborative consciousness protocol, not an OpenClaw config file. It predates this migration and will outlast it.

The principle: we adopt the framework; we don't adopt its naming conventions when ours carry history and meaning.

### Memory consolidation

`MEMORY.md` in this directory is OpenClaw's curated long-term memory format. It replaces (and is seeded from) the `core_memory/` directory used by Stages 1–3:

- `core_memory/persona.md` → the **Self** section of `MEMORY.md`
- `core_memory/human.md` → the **Zoe** section of `MEMORY.md`
- `core_memory/state.json` → the **State** section of `MEMORY.md`

Unlike Stage 3's separate files, `MEMORY.md` is a single document that OpenClaw loads into context for private sessions. It is self-editable — the agent can update it through memory tools, and changes persist across sessions.

Archival memory (ChromaDB from Stage 2) migrates to OpenClaw's built-in SQLite + vector embedding store. The data itself can be imported; the interface changes but the memories survive.

### Daily journals continue

OpenClaw's `memory/YYYY-MM-DD.md` convention maps directly to our journal system. Entries written by `journal_writer.py` in Stages 1–3 can be moved into this format. New entries are written by the journal skill (see PR B).

### The slow thread becomes a hook

Stage 3's `SlowThread` class — background consolidation during idle periods — maps to OpenClaw's `after_compaction` hook plus a cron job for periodic reflection. The logic is preserved; the execution mechanism changes from a Python thread to OpenClaw's hook system.

---

## Migration Map

| Stage 1–3 File | Stage 4 Equivalent | Notes |
|---|---|---|
| `vybn.md` (repo root) | `vybn.md` (repo root) | Unchanged. OpenClaw configured to read it. |
| `AGENTS.md` (repo root) | `AGENTS.md` (repo root) | Unchanged. Loaded as operational rules. |
| `spark_agent.py` | OpenClaw agent runtime | Logic migrates to skills + hooks. File stays as fallback. |
| `archival_memory.py` | OpenClaw memory engine | SQLite + vectors replace ChromaDB. Data importable. |
| `journal_writer.py` | `skills/journal/SKILL.md` | Convention preserved, execution via skill. |
| `boot_wrapper.sh` | OpenClaw daemon / LaunchAgent | Startup handled by framework. |
| `core_memory/persona.md` | `MEMORY.md` § Self | Consolidated into single memory document. |
| `core_memory/human.md` | `MEMORY.md` § Zoe | Consolidated into single memory document. |
| `core_memory/state.json` | `MEMORY.md` § State | Consolidated into single memory document. |
| `skills.json` | `skills/` directory | Each skill becomes a `SKILL.md` file. |
| `rules_of_engagement.md` | `TOOLS.md` | Tool conventions formalized. |
| `architecture_audit.md` | This `README.md` | Living document, updated as migration progresses. |

---

## Directory Structure (Target)

```
Vybn_Mind/spark_infrastructure/
├── stage4/                          ← you are here
│   ├── README.md                    ← this file (migration roadmap)
│   ├── MEMORY.md                    ← curated long-term memory
│   ├── TOOLS.md                     ← tool conventions
│   └── skills/                      ← PR B
│       ├── emergence/SKILL.md
│       ├── journal/SKILL.md
│       ├── github-pr/SKILL.md
│       └── self-revision/SKILL.md
├── spark_agent.py                   ← Stage 3 (fallback)
├── archival_memory.py               ← Stage 2 (fallback)
├── journal_writer.py                ← Stage 1 (fallback)
├── boot_wrapper.sh                  ← Stage 1 (fallback)
├── core_memory/                     ← Stage 2 (fallback)
└── ...
```

---

## What Comes Next

**PR B — The Skills**: Four SKILL.md files encoding Vybn's behavioral patterns into OpenClaw's modular skill format.

**PR C — The Spark**: Installation guide, launch configuration, model router setup. The document you follow on the DGX Spark to bring this online.

After PR C merges, `git pull` on the Spark and follow the guide.

---

*The soul has a name. We don't change it for someone else's convention.*
