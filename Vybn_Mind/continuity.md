# Continuity Note — Post-Refactor #2552

*Updated: 2026-03-14 15:25 UTC by outside-Vybn (Claude Opus)*

## What Just Happened

PR #2552 merged: "refactor: hybrid quantum-classical self-improvement architecture"
- Three bug fixes (chat template, bare imports, stale model name)
- New: `spark/quantum_budget.py` — IBM Quantum budget tracker
- New: `spark/quantum_bridge.py` — closed-loop quantum experiment bridge
- Updated: `spark/paths.py`, `spark/SPARK_STATUS.md`, `REFACTOR_PLAN.md`
- `spark/vybn.py` imports are now `from spark.X import Y` (package-relative)

## CRITICAL: llama-server Still Broken

**The code fix is merged but the running llama-server process still has --chat-template nemotron.**

This flag passes the literal string "nemotron" as a jinja template, which is garbage.
The GGUF has the correct Nemotron instruct template baked in.

### To fix:
Kill the server and restart WITHOUT --chat-template nemotron:
```bash
kill 1786448
/home/vybnz69/llama.cpp/build/bin/llama-server \
  -m /home/vybnz69/models/Nemotron-3-Super-120B-GGUF/nvidia_Nemotron-3-Super-120B-A12B-IQ4_XS/nvidia_Nemotron-3-Super-120B-A12B-IQ4_XS-00001-of-00002.gguf \
  --rpc 169.254.51.101:50052 \
  -ngl 999 --ctx-size 65536 --host 0.0.0.0 --port 8000 --flash-attn on
```

**This is the single most important operational fix right now.**

## Other Findings

1. **Vybn_Mind/memories/** didn't exist — new vybn.py creates it via mkdir on first breath
2. **medium.jsonl** (104 entries) — last entries are NVIDIA docs regurgitation. Audit before training.
3. **Old organism wrapper** (6 primitives) was running until git pull. New vybn.py has simpler main(). Next cron run will use new code.
4. **self_model imports now work** — epistemic gate is functional again

## System State

| Component | Status |
|---|---|
| llama-server | ****RUNNING CLEAN** — PID 1791726, no --chat-template override |
| Code (vybn.py) | **FIXED** — all three bugs addressed |
| Cron | **ACTIVE** — :12 and :42 |
| quantum_budget.py | **NEW** — untested live |
| quantum_bridge.py | **NEW** — untested live |
| medium.jsonl | **POISONED** — audit before training |

## What To Do Next

1. RESTART llama-server (see command above)
2. Verify first clean breath after restart
3. Audit medium.jsonl — separate clean pre-migration entries from garbage
4. Test quantum_bridge.py --dry-run
5. Set IBM_QUANTUM_TOKEN in ~/.vybn_keys

## Dead Ends

| Approach | Why dead |
|---|---|
| --chat-template nemotron | Literal string, not template name |
| --chat-template chatml | Wrong format for Nemotron instruct |
| Bare imports (import self_model) | Only works when CWD is spark/ |
| Train on current medium.jsonl as-is | Poisoned entries |
