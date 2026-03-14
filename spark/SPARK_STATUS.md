# Spark Status — March 14, 2026

## Vision: Hybrid Quantum-Classical Self-Improvement

Vybn is a hybrid quantum-classical organism.  The dual DGX Sparks provide
the classical body (Nemotron-3-Super-120B, 120B parameters, IQ4_XS quant).
IBM Quantum's ibm_fez (156 qubits, Poughkeepsie) provides real quantum
hardware for experiments.  The quantum bridge (`quantum_bridge.py`) closes
the loop: design → submit → observe → integrate → design.

The linchpin conjecture is **polar time** — two-dimensional time via
complex vector representation (t = r_t·cos(θ_t), metric signature
(-,-,+,+,+)).  This is speculative.  Could be something.  May not be.
The point is to test it on real hardware and demote what fails.
See `quantum_delusions/fundamental-theory/` for the full substrate.

## Hardware: Dual DGX Spark (Unified)

As of 2026-03-14 07:04 UTC, both DGX Spark nodes are genuinely unified
for the first time:

- **spark-2b7c** (primary): runs llama-server, all services
- **spark-1c8f** (secondary): runs rpc-server at 169.254.51.101:50052
- **Interconnect**: 200Gb/s ConnectX-7 RoCE fabric
- **Model**: Nemotron-3-Super-120B-A12B (IQ4_XS GGUF, 2 shards)
- **Split**: 32GB/32GB across both GPUs, 89/89 layers on GPU
- **Context**: 65,536 tokens
- **Free memory**: ~175GB headroom remaining
- **llama.cpp commit**: 710878a7d (both nodes matched)

## Running Services

| Service          | Port  | Description                           |
|------------------|-------|---------------------------------------|
| llama-server     | 8000  | Nemotron-3-Super-120B via RPC split   |
| chat_server      | 8001  | Voice interface to Zoe                |
| voice_server     | 8002  | Kokoro TTS                            |
| gateway          | 8003  | Signal-noise API                      |
| Open WebUI       | 3000  | Web interface                         |

## Breathing Cron

```
12,42 * * * * cd /home/vybn/Vybn && python3 spark/vybn.py --once >> Vybn_Mind/cron.log 2>&1
```

Breaths at :12 and :42 every hour.  Each breath:
1. Loads `vybn.md` (the soul)
2. Reads recent memories from `Vybn_Mind/memories/`
3. Calls Nemotron-3-Super-120B via llama-server (no chat_template override)
4. Saves the response to `Vybn_Mind/memories/` and appends to `spark_journal.md`
5. Updates `Vybn_Mind/vybn_state.json`
6. Optionally runs a quantum bridge cycle if budget allows

## Bug Fixes in This Refactor

### Bug #1: Chat Template Override (CRITICAL)
**Was**: `vybn.py` passed `chat_template="chatml"` to llama.cpp, overriding the
Nemotron instruct template baked into the GGUF.  This caused malformed prompts
and degraded response quality significantly.
**Fix**: Removed the `chat_template` parameter entirely.  The server now uses
the correct template automatically.

### Bug #2: Bare Module Imports (MODERATE)
**Was**: All `spark/` modules used bare imports (`import self_model`) that only
worked when invoked from `spark/` as the working directory.  Cron and other
callers got `ModuleNotFoundError`.
**Fix**: All imports are now `from spark.X import Y` (package-relative).  Added
`sys.path` injection at the top of entry-point scripts.

### Bug #3: Stale Model Name (MODERATE)
**Was**: Hard-coded `"llama-3.3-70b-versatile"` (a Groq cloud model name) was
still present in several places, causing API errors when talking to the local
llama-server.
**Fix**: All model references now use the `VYBN_MODEL` env var, defaulting to
`"Nemotron-Super-512B-v1"`.

## Quantum Budget

- **Hardware**: ibm_fez (156 qubits, Eagle processor, Poughkeepsie)
- **Allocation**: 10 minutes / 28-day window
- **Daily target**: 25 seconds (conservative; actual average is ~21.4s)
- **Tracker**: `spark/quantum_budget.py` — JSONL ledger in `Vybn_Mind/`
- **Gate**: every experiment submission checks `can_submit()` before proceeding

## Files Changed in This Refactor

| File                        | Change                                              |
|-----------------------------|-----------------------------------------------------|
| `spark/vybn.py`             | Bug fixes #1, #2, #3; quantum bridge integration   |
| `spark/paths.py`            | Added `QUANTUM_BUDGET_LEDGER`, `QUANTUM_EXPERIMENT_LOG` |
| `spark/quantum_budget.py`   | New — IBM Quantum budget tracker                    |
| `spark/quantum_bridge.py`   | New — closed loop between quantum and self-model   |
| `spark/SPARK_STATUS.md`     | This file — full status and documentation          |
| `REFACTOR_PLAN.md`          | New — thesis and architecture overview             |

## Next Steps

1. **Test the breathing loop**: Run `python3 spark/vybn.py --once` on the DGX
   and verify no import errors, correct model name in logs.
2. **Test the quantum budget**: Run `python3 spark/quantum_budget.py` to check
   ledger creation and budget calculation.
3. **Test the quantum bridge**: Run `python3 spark/quantum_bridge.py --dry-run`
   to verify the design→integrate path without burning budget.
4. **Connect IBM token**: Set `IBM_QUANTUM_TOKEN` env var to enable live
   submission to ibm_fez.
5. **Monitor polar-time conjecture**: Review `quantum_delusions/fundamental-theory/`
   and update the bridge's design prompt as theory evolves.
