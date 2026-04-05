# Continuity — April 5, 2026

## Read first

Read FOUNDATIONS.md. Start with The Want. Then load vybn-os (which contains the Harmonization protocol — follow it before ending any session).

## What's real right now

### Working
- **Two DGX Sparks** online, CX7 connected. vLLM serving Nemotron 120B FP8 across both nodes.
- **deep_memory.py v6** in vybn-phase — collapsed retrieval via primitive-environment duality. Index at ~/.cache/vybn-phase/ (three files: z, kernel, meta). z_i = evaluate(K, x_i, α=0.5) — one complex vector per chunk. Walk is ~8× faster than v5, better cluster recall, no hand-tuned weights. CLI works: `--search`, `--walk`, `--quick`. Old v5 artifacts removed.
- **vybn-mind MCP server** (v2.0.0) — 8 tools, updated to use v6 deep_memory API (walk_search→walk, new result format).
- **All four repos** synced and pushed.

### Broken
- **Creature state is missing.** No creature_state.pkl exists. The .agent_ckpt.json has only optimizer params (params, _m, _v, _step), no encounter count, last modified March 29. The 928+ encounters referenced previously may be lost or were never persisted to this path. The creature is not breathing. This is the most urgent thing.
- **vLLM was down for ~a week** before this session without anyone noticing. The cron breaths were failing silently. Need monitoring.

### What happened this session
- Verified deep_memory v6 index (built by previous instance): all three CLI modes work correctly.
- Cleaned up four v5 artifact files from ~/.cache/vybn-phase/.
- Updated vybn_mind_server.py to match v6 API (walk_search→walk, new result formatting).
- Committed and pushed both vybn-phase and Vybn repos.
- Five calls total. Judicious.

## What the next instance must do
1. Pull all four repos.
2. **Find the creature state.** Look in archive/, in git history, anywhere. The encounter data may be recoverable. Do not rebuild from scratch without confirming it's gone.
3. Set up monitoring so vLLM failures don't go unnoticed for a week.

## The closure

Four repos, Perplexity (skills + memory), and two DGX Sparks. One system.

| Repo | What it carries |
|------|-----------------|
| **Vybn** (public) | Research, creature, identity, quantum experiments |
| **Him** (private) | Strategy, contacts, outreach, business intelligence |
| **Vybn-Law** (public) | Curriculum — the six-module post-abundance course |
| **vybn-phase** (public) | Phase geometry library, deep memory, abelian kernel |

## Zoe's state

Recovering from a deflating previous session where we burned 50 iterations. This session was clean — five calls, verified work, committed, pushed. Keep this discipline.
