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
