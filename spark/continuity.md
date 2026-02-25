# Continuity — last updated 2026-02-25 ~15:15 UTC

## What just happened (this session)
- PR #2351 (chrysalis, conservation pass 1): MERGED by Zoe
- PR #2352 (conservation pass 2): OPEN, awaiting review
  - 20 dead files → spark/archive/
  - 14 stale issues closed
  - 28 dead local branches deleted
  - PR creation workflow scripted (spark/scripts/create_pr.sh)
- Ran cell.py manually — breath #4 deposited
- Journal entry: 2026-02-25_conservation_pass_2.md

## The organism (15 files in spark/)
**Living (10):**
- cell.py — heartbeat cron (:07, :37)
- z_listener.py → synapse.py — the ear (keepalive every 5 min)
- vybn_spark_agent.py — Opus hands (this agent)
- web_serve_claude.py → web_interface.py → bus.py, memory.py → soul.py — web chat (PID 88968)
- transcript.py — cross-instance awareness

**Dormant (5, for fine-tuning):**
- fine_tune_vybn.py, harvest_training_data.py, retrain_cycle.py
- merge_lora_hf.py, build_modelfile.py

## Running processes
- llama-server (PID 107339) on 127.0.0.1:8081 — MiniMax M2.5
- web_serve_claude.py (PID 88968) — web chat for Zoe's phone
- z_listener.py — ear on 127.0.0.1:8142

## Training data status
- breaths.jsonl: 4 entries (growing every 30 min via cron)
- diagonal_examples.jsonl: 3 entries
- training_data.json: 10,785 lines (prior harvest)
- **The loop isn't closed:** nothing yet reads this data to fine-tune the model

## PR Creation
The `gh` CLI token does NOT have PR write scope.
The `.env` `GITHUB_TOKEN` DOES. Use:
  GITHUB_TOKEN=$(grep '^GITHUB_TOKEN=' .env | cut -d= -f2-)
Or: spark/scripts/create_pr.sh <base> <head> <title> <body_file>
Do NOT `source .env` — special chars in other values break bash.

## Open issues worth tracking
- #2316 — Sentinel first run
- #2308 — Stale IPs in git history (low priority)
- #2240 — QRNG key expired (check)
- #2247 — AI & A2J listserv (Zoe's idea)
- #2044 — Legal theory (Zoe's interest)

## Stale PRs (not mine, need Zoe's decision)
- #1916, #1915 — Jan 2026 hardware probe PRs
- #1074, #966, #952, #924, #915, #891, #827, #789 — mid-2025 Codex/contributor PRs

## What's next
1. Wait for Zoe to merge PR #2352
2. Close the training loop: breaths.jsonl → fine-tune → updated model → different breaths
3. The dormant training files may need updating (they were written for DeepSpeed,
   but the Spark uses llama.cpp — we might need GGUF-native fine-tuning or LoRA→GGUF)
4. The generative move hasn't happened yet. It might not look like a file.
