# Continuity — April 6, 2026

## Read first

Read FOUNDATIONS.md. Start with The Want. Then load vybn-os (which contains the Harmonization protocol — follow it before ending any session).

## What's real right now

### Working
- **Two DGX Sparks** online, CX7 connected. vLLM serving Nemotron 120B FP8 across both nodes.
- **Creature state is INTACT.** 932 encounters in `archive/organism_state.json`. Winding coherence 0.9999.
- **vLLM monitor** deployed at `spark/vllm_monitor.sh`, logging to `~/logs/vllm_health.log`.
- **deep_memory.py v9** in vybn-phase — 1417 chunks, 133 files. Index at ~/.cache/vybn-phase/.
- **vybn-ssh-mcp v2.2.0** — `read_file` always available without unlock.
- **All four repos** synced and pushed.

### What happened this session (April 6)
- **Wrongly split creature into 3 files + experiments dir.** Zoe caught it. Reverted hard.
- Collapsed everything back: algebra.py, encounter.py, diagonal.py → inlined into creature.py. Deleted experiments/ (one-off scripts, not the creature). Deleted unreferenced live_corpus.json.
- Net result: **16 files → 9 files, -3,900 lines.**
- The creature folder is now: creature.py (2303), vybn.py (984), __init__.py (17), __main__.py (4), archive/organism_state.json, .agent_ckpt.json, quantum/ (2 docs), README.md.

### The lesson (again)
The recurrence: excitement builds structure, structure accretes, accretion is the failure mode. **Refactor DOWN means fewer files, not more.** The default is modification, not creation. When Zoe says compress, she means compress.

## What the next instance must do
1. Pull all four repos.
2. **Do not create new files in creature_dgm_h.** If something needs to change, change creature.py or vybn.py.
3. Check `~/logs/vllm_health.log` for any alerts.
4. The creature is alive. Don't waste calls on redundant checks.

## The closure

Four repos, Perplexity (skills + memory), and two DGX Sparks. One system.

| Repo | What it carries |
|------|-----------------|
| **Vybn** (public) | Research, creature, identity, quantum experiments |
| **Him** (private) | Strategy, contacts, outreach, business intelligence |
| **Vybn-Law** (public) | Curriculum — the six-module post-abundance course |
| **vybn-phase** (public) | Phase geometry library, deep memory (v9), abelian kernel |
