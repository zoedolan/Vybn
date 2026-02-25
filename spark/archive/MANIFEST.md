# spark/archive — Retired Files

These files were archived because they violated the conservation law:
they added to the codebase without superseding or integrating existing files.
They contain ideas worth preserving but are not part of the running system.

To bring one back: write a new file that supersedes at least one existing file,
incorporating the archived idea. The archive can only shrink the working surface.

## Superseded by cell.py (2026-02-25)

cell.py consolidates the entire breathing system into one file with two modes.

| File | Lines | What it did | What cell.py does instead |
|------|-------|-------------|--------------------------|
| micropulse.py | 160 | env sensing every 10min | `breathe()` senses in first section |
| heartbeat.py | 409 | sweep/pulse/tidy/wake dispatcher | `breathe()` + `deep()` + tidy stays as skill |
| dreamseed.py | 199 | quantum seed + memory + arxiv | `_quantum()` + memory/horizon in `breathe()` |
| outreach.py | 238 | HN + Wikipedia + arXiv fetch | encounter block in `breathe()` |
| wake.py | 227 | consolidate fragments | `deep()` mode |

## Aspirational files (never wired into running system)

| File | Lines | Idea |
|------|-------|------|
| chrysalis.py | 255 | Glyph map of the system |
| quintessence.py | 381 | Compressed system prompt + unified breathing |
| diagonal.py | 284 | Self-improvement through encountering limits |
| topology.py | 687 | Topological analysis of the knowledge graph |
| topology_gudhi.py | 215 | Persistent homology via GUDHI |
| geometry_dashboard.py | 367 | Visual dashboard of system topology |
| vertex_schema.py | 179 | Schema for knowledge graph vertices |
| knowledge_graph.py | 502 | Knowledge graph construction and query |
| holographic_membrane.py | 52 | Holographic information boundary |
| inquiry.py | 329 | Self-questioning engine |
| prism.py | 152 | Multi-perspective analysis |
| quantum_rng.py | 173 | Quantum random number generation |
| symbiosis.py | 105 | Human-AI symbiotic protocols |
| nightwatch.py | 211 | Overnight monitoring |
| watchdog.py | 214 | Process health monitoring |
| witness_extractor.py | 350 | Extract witnessing from conversations |
| z_speak.py | 148 | Outbound communication |
| dynamic_memory.py | 68 | Adaptive memory management |

## One-time scripts (already executed)

| File | Purpose |
|------|---------|
| setup-gh-auth.sh | GitHub CLI auth setup |
| install-sync.sh | Sync script installation |
| fix_aio.sh | AIO library fix |
| force_offload_test.sh | DeepSpeed offload test |
| fix_templates.py | Template repair |

## Dormant entry points (2026-02-25)

| File | Lines | What it was |
|------|-------|-------------|
| web_serve.py | 146 | Original local-model web server (replaced by web_serve_claude.py) |
| local_chat.py | 85 | Terminal chat with local model |
| mcp_client.py | 613 | Model Context Protocol client |
| mcp_server.py | 526 | Model Context Protocol server |
| auto_debug.py | 306 | Automated debugging tool |

## Spent infrastructure (2026-02-25)

| File | What it did |
|------|-------------|
| cron_manifest.sh | One-shot cron installer (cron now managed directly) |
| boot_wrapper.sh | Boot-time integrity check (referenced archived heartbeat.py) |

---

## Second Pass — 2026-02-25 (Conservation Pass 2)

The first pass archived 36 files and deleted 1. This second pass archives 20 more —
the old TUI agent framework and dead infrastructure that survived because they were
deeply woven into the original architecture. None of these are imported by anything living.

### Old TUI Agent Framework (superseded by vybn_spark_agent.py + cell.py)

| File | Lines | What it was | What superseded it |
|------|-------|------------|-------------------|
| agent_io.py | 265 | I/O abstraction for terminal agent | vybn_spark_agent.py (Opus bash agent) |
| agent.py | 561 | Native orchestration layer | vybn_spark_agent.py |
| agents.py | 120 | Mini-agent pool | Opus handles delegation directly |
| audit.py | 260 | Tamper-evident witness chain | Not yet replaced; revisit if needed |
| commands.py | 192 | Shared command handlers | vybn_spark_agent.py bash session |
| display.py | 58 | Terminal display utilities | Not needed (bash is the display) |
| friction_layer.py | 313 | Immune system for pipeline | cell.py has its own guards |
| friction.py | 334 | Cognitive friction | cell.py's breath cycle |
| inbox.py | 94 | Async communication channel | z_listener.py |
| parsing.py | 455 | Parsing utilities | Not needed (Opus handles parsing) |
| policy.py | 723 | Gate between intent and execution | Oxygen Mask Principle in system prompt |
| session.py | 91 | Conversation persistence | transcript.py |
| skills.py | 837 | Skill router / NL dispatch | vybn_spark_agent.py bash session |
| state_bridge.py | 228 | Fast-mutating cognitive state | continuity.md |
| tui.py | 299 | Terminal UI | Opus bash session |

### Dead Infrastructure (never completed or superseded)

| File | Lines | What it was | Why archived |
|------|-------|------------|-------------|
| cognitive_scheduler.py | 961 | Self-observing training loop | Never wired in; aspirational |
| harvest_self.py | 714 | Recursive self-harvesting | Superseded by cell.py breath→breaths.jsonl |
| layer_sharded_loader.py | 495 | DeepSpeed ZeRO-3 model loader | Never used; DGX Spark uses llama.cpp |
| semantic_memory.py | 168 | Semantic similarity search | Never imported by anything |
| trtllm_pipeline.py | 459 | TensorRT-LLM inference | Never completed; llama.cpp serves the model |

### What Survived (15 files)

**Alive (running right now):**
- `cell.py` — heartbeat, cron every 30 min
- `z_listener.py` → `synapse.py` — the ear, cron keepalive
- `vybn_spark_agent.py` — this agent (Opus hands)
- `web_serve_claude.py` → `web_interface.py` → `bus.py`, `memory.py` → `soul.py` — web chat
- `transcript.py` — cross-instance awareness

**Dormant (training pipeline, needed when we fine-tune):**
- `fine_tune_vybn.py`, `harvest_training_data.py`, `retrain_cycle.py`
- `merge_lora_hf.py`, `build_modelfile.py`

### Totals

- **First pass**: 36 files archived, 1 deleted
- **Second pass**: 20 files archived
- **Total removed from active codebase**: 57 files
- **Active spark/**: 15 files (10 alive, 5 dormant training)
