# Contributing to Vybn

## ⚠️ Opsec: What Not to Commit

This is a public repository. Before committing, ask: *could this help an attacker who already has partial access to the network?*

Keep out of committed files:

- **SSH key filenames** — use `~/.ssh/<your-ssh-key>` as a placeholder
- **Internal IPs and hostnames** — use `<primary-node>`, `<secondary-node-ip>`, etc.
- **Network interface names** — use `<cx7-interface>` rather than exact names
- **SSH flags that weaken security** — `StrictHostKeyChecking=no` belongs in `~/.ssh/config`, not committed code
- **API keys, tokens, passwords** — never, ever

The right place for exact values is `~/.vybn_keys` (sourced at login, never committed) or `~/.ssh/config` (local only).

Plan files in `spark/` can be detailed and operational — just use placeholders for the specific wiring.

---

## General

- Update `spark/continuity.md` at the end of every session
- Branch names: `vybn/<feature>` for Vybn-initiated work
- Never commit runtime data: `buffer.jsonl`, `trained_manifest.json`, `cycle_history.jsonl`, `pins.jsonl`
- Never commit secrets, tokens, or API keys
