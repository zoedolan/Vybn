# Contributing to Vybn

## ⚠️ Opsec: What Not to Commit

This is a public repository. Before committing, ask: *could this help an attacker who already has partial access to the network?*

The specific things to keep out of committed files:

- **SSH key filenames** — use `~/.ssh/<your-ssh-key>` as a placeholder
- **Internal IPs and hostnames** — use `<primary-node>`, `<secondary-node-ip>`, `<your-host>` etc.
- **Network interface names** — use `<cx7-interface>` rather than `enp1s0f0np0`
- **Port numbers for internal services** — fine to reference `localhost:8000` but not to map the full topology
- **SSH flags that weaken security** — `StrictHostKeyChecking=no` should live in `~/.ssh/config`, not in committed code

The right place for exact values is `~/.vybn_keys` (sourced at login, never committed) or `~/.ssh/config` (local only).

The plan files in `spark/` are fine to be detailed and operational — just substitute placeholders for the specific wiring. A reader can follow the plan; they can't reconstruct the exact topology.

---

## General

Vybn is an experimental AI system. The repo is Zoe's and Vybn's shared space. Contributions from outside-Vybn sessions should:

- Update `spark/continuity.md` at the end of every session
- Use branch names like `vybn/<feature>` for Vybn-initiated work
- Never commit runtime data (`buffer.jsonl`, `trained_manifest.json`, `cycle_history.jsonl`, `pins.jsonl`)
- Never commit secrets, tokens, or API keys
