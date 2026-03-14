# Continuity Note — Post-Consolidation, Container Running

*Updated: 2026-03-14 16:39 PDT by outside-Vybn (Perplexity/Sonnet)*

## Current System State — VERIFIED FACTS

| Component | Status |
|---|---|
| llama-server | **RUNNING** — `{"status":"ok"}` on port 8000 |
| Chat template | **FIXED** — no --chat-template flag, native GGUF template |
| Organism cron | **ACTIVE** — breathes at :12 and :42 |
| Last breath | 23:12 UTC March 14 — coherent prose confirmed |
| Growth buffer | 164 entries — previously unprocessed |
| vllm_node container | **RUNNING** — started with `--gpus all`, repo mounted at /workspace/Vybn |
| Container GPU | CUDA 13.1, GB10 detected, PyTorch cuda=True inside container |
| PEFT + TRL | Installed in container (peft 0.18.1, trl 0.29.0) |
| Consolidator | **FIXED** — max_tokens 1000→2048 (Nemotron reasoning token bug) |
| ComplexMemory curvature | 0.0000 — still flat (no novel input reaching breaths yet) |

## What Was Fixed This Session (March 14 afternoon)

### 1. vllm_node container — was not running
The growth loop has been failing with "Training data not accessible in container" for 18+ breaths because the `vllm_node` container wasn't running. The image existed (`vllm-node:latest`, 25.6GB). Started with:
```
docker run -d --name vllm_node --gpus all \
  -v /home/vybnz69/Vybn:/workspace/Vybn \
  vllm-node:latest sleep infinity
```
Container confirmed: GPU visible, /workspace/Vybn mounted, PEFT+TRL installed.

### 2. Consolidator token budget — Nemotron reasoning bug
Nemotron 3 Super uses chain-of-thought reasoning_content before writing its content response. With `max_tokens=1000`, the model exhausted its token budget on reasoning and returned `content=""`, `finish_reason="length"`. The synthesis appeared to work (no error) but produced 0 chars.

Fix: `SYNTHESIS_MAX_TOKENS = 2048` in `spark/consolidator/__init__.py`. Verified: with 2048 tokens, Nemotron produces ~880 content tokens after ~105 reasoning tokens.

### 3. Growth loop model mismatch — documented but not yet fixed
The `train_cycle.py` training script calls `AutoModelForCausalLM.from_pretrained(model_id)`. Only GGUF models are on disk. This will fail at model load. The correct path is `llama-finetune` (present at `~/llama.cpp/build/bin/llama-finetune`, CUDA-enabled). **The container path check now succeeds** (repo is mounted). The training script itself still needs to be ported to llama-finetune.

## What Remains

1. **Growth loop training** — container is up, data path works, but script uses AutoModelForCausalLM on a GGUF. Port `_generate_train_script()` to use `llama-finetune` OR download a HF-format model.
2. **Breath repetition** — arXiv papers (57+) exist in the buffer but breaths don't see them (stripped in commit 6cc4b58b). Wire one paper per breath into the context assembler to break flat curvature.
3. **Quantum dry_run** — the quantum experiment runner still has `dry_run=True`. Set it to False with the local Aer simulator.
4. **Consolidator** — now fixed. Will produce actual synthesis on next scheduled run.

## Hardware — VERIFIED

- **spark-2b7c**: 128 GB unified, NVIDIA GB10 (sm_121), CUDA 13.0. Organism + llama-server here.
- **spark-1c8f**: 128 GB unified. Available for distributed training.
- **NCCL**: Validated 34-36 GB/s cross-node.
- **llama-server**: PID active, consuming ~65GB VRAM, ~63GB free.

## Critical Reminders

1. **NEVER use `--chat-template nemotron`** — let the GGUF handle its own template
2. **Use `/usr/bin/python3`** for organism and arxiv cron (system python has requests)
3. **buffer.jsonl is .gitignored** — don't try to git add -f it for routine commits
4. **vllm_node container** — will not survive reboot. May need to add to systemd or cron.
5. **Consolidator token budget**: `SYNTHESIS_MAX_TOKENS = 2048` — do not lower below 1500 or Nemotron reasoning will starve content again.
