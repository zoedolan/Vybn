# _archive/ — superseded code kept for history

These files are no longer loaded by any live service and no longer imported
by any live module. They are preserved here because the repository is the
evidence of co-evolution and nothing is deleted silently. Anything moved in
from now on gets a short explanation: where it came from, what replaced it,
when it stopped being live, why keeping the history matters.

If a file in this directory ever needs to come back, do not edit it in
place — either copy it out or promote a specific section into the live
tree. Archive files are not a staging area.

## April 18, 2026 — collective-walk refactor

After Zoë's diagnosis — _“too many disparate scripts”_ — the single walk
(`vybn-phase/walk_daemon.py`, port 8101, 14,755+ steps of accumulated
geometry) became the sole source of truth. Every other portal, chat API,
and MCP adapter that had started maintaining its own walk state was
demoted to a rotation/retrieval proxy. The files below were left over from
before that unification and no longer serve the live system.

| File | Superseded by | Why kept |
|------|---------------|----------|
| `Vybn_Mind__origins_portal_api.py` (88 lines) | `origins_portal_api_v4.py` | The April 11 stub — the moment the portal first existed. Worth keeping as the seed form. |
| `Vybn_Mind__vybn-chat-api.service` | Systemd unit never enabled; chat now runs via `nohup` on port 3001 from `Vybn-Law/api/vybn_chat_api.py` | Documents an intended deployment path we abandoned for the current lighter footprint. |
| `vybn-phase/deep_memory_v6_backup.py` (in sibling repo) | `vybn-phase/deep_memory.py` current head | Pre-distillation form of deep_memory before the v6 rewrite shipped. Archived inside `vybn-phase` (private), not here. |

## April 18, 2026 — round 2: dead-MCP audit

A second pass after the first round 2 commit. Continuity claimed `Vybn_Mind/vybn_mind_server.py` had been “kept as a local/stdio variant” — but `pgrep`, `lsof`, the systemd units, and the cron table all agreed: nothing was actually using it. The live MCP surface is `spark/server.py` on port 8400, which already includes the walk-daemon endpoints (`/where`, `/enter`, `/arrive`) and the deep-memory routes. Keeping a parallel server.py that no caller knew about was a textbook case of dead wood masquerading as redundancy.

| File | Superseded by | Why kept |
|------|---------------|----------|

## Principle

Archiving is an act of respect for the code that got us here. Nothing in
this directory is alive; everything in this directory was once the best
thing we had. Do not modify. Do not import. Do not resurrect without
reading the commit that moved it in.

