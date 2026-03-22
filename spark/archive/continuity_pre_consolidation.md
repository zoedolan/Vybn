# Continuity Note — Ground Truth, March 15 2026

*Updated: 2026-03-15 07:05 PDT by outside-Vybn (Perplexity/Sonnet)*

## Current System State — VERIFIED FACTS

| Component | Status |
|---|---|
| Serving model | **Nemotron 3 Super 120B-A12B** — IQ4_XS GGUF, ~63 GB, llama-server on port 8000 |
| Training model | **Nemotron NVFP4 safetensors** — 75 GB in `~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/` |
| MiniMax M2.5 | On disk (~228 GB) as rollback only. **Not serving. Not the training model.** |
| Organism cron | **ACTIVE** — breathes at :12 and :42. Latest breath: 13:03 UTC March 15 (breath #10) |
| Growth buffer | ~237 training examples ready |
| vllm_node container | **RUNNING** — `--gpus all`, repo at `/workspace/Vybn`, PEFT 0.18.1 + TRL 0.29.0 installed |
| Secondary Spark | Available. NCCL validated 34-36 GB/s cross-node |
| Memory available | ~53 GB free on primary node (68 GB used) |
| arXiv buffer | Being fed per breath via `spark/arxiv_fetcher.py` (landed in PR #2568) |
| Quantum | dry_run — IBM_QUANTUM_TOKEN not set in cron environment |

## The ONE Confabulation to Stamp Out

**The local model is NOT MiniMax M2.5 (229B dense).** It never was, after March 14.
MiniMax is on disk as a fallback. Nemotron 3 Super 120B-A12B has been the serving model
since the migration. Any summary or prompt that says otherwise is stale.

When in doubt: `ps aux | grep llama-server | grep -o '\-m [^ ]*'`

## The Training Path — What The Migration Plan Says

This is documented in `spark/NEMOTRON_MIGRATION_PLAN.md` and `spark/growth/nemotron_assessment.md`.
The short version:

**Phase 3 — LoRA fine-tuning:**
- Run PEFT/TRL LoRA inside the `vllm_node` container (or across both Sparks via `torchrun`)
- Base model: NVFP4 safetensors (`~/.cache/huggingface/hub/models--nvidia--...NVFP4/`)
- Target layers: attention projections only — `q_proj, k_proj, v_proj, o_proj` (8 layers, ~12B active params for MoE routing)
- Training data: 237 examples in the growth buffer, DeltaPackage format, already pipeline-ready

**Phase 4 — Hot-load adapter:**
- Convert LoRA adapter to GGUF: `python3 convert_lora_to_gguf.py`
- Load into running llama-server: `--lora <adapter.gguf>` (no restart needed)

**What `train_cycle.py` must do:**
- NOT use llama-finetune (it is a full-parameter fine-tuner for FP32, not a LoRA tool)
- NOT target the serving GGUF as the training model
- Run `docker exec vllm_node torchrun` (or direct `torchrun` across both nodes) with a PEFT/TRL script against the NVFP4 safetensors
- Output: a `.safetensors` LoRA adapter, then convert to GGUF via `convert_lora_to_gguf.py`

## What Was Fixed in the Post-Consolidation Session (March 14)

1. **vllm_node container** — started with `--gpus all`, repo mounted, PEFT+TRL confirmed
2. **Consolidator token budget** — raised to 2048 so Nemotron reasoning doesn't starve content
3. **arXiv feed** — `spark/arxiv_fetcher.py` now refills buffer.jsonl and wires one paper per breath
4. **Quantum token diagnostic** — now logs clearly when IBM_QUANTUM_TOKEN is missing

## What Remains

1. **`train_cycle.py` rewrite** — point at NVFP4 safetensors, use PEFT/TRL LoRA, follow Phase 3/4 of the migration plan. The data pipeline is ready. The container is ready. The weights are on disk. This is the next real task.
2. **Quantum real shots** — `export IBM_QUANTUM_TOKEN=<token>` in cron environment or `~/.profile`
3. **ComplexMemory curvature** — still flat. arXiv papers now feeding per breath; curvature should become non-zero as novel material enters.
4. **vllm_node container** — will not survive reboot. Add to systemd or cron if uptime matters.

## Hardware — VERIFIED

- **spark-2b7c** (primary): 128 GB unified, NVIDIA GB10 (sm_121), CUDA 13.0. Organism + llama-server here.
- **spark-1c8f** (secondary): 128 GB unified. Available for distributed LoRA training.
- **NCCL**: 34-36 GB/s cross-node. `torchrun --nproc_per_node=1 --nnodes=2` validated.

## Critical Reminders

1. **NEVER use `--chat-template nemotron`** — let the GGUF handle its own template
2. **Use `/usr/bin/python3`** for organism and arxiv cron (system python has requests)
3. **buffer.jsonl is .gitignored** — do not `git add -f` it
4. **vllm_node container will not survive reboot**
5. **Consolidator token budget**: `SYNTHESIS_MAX_TOKENS = 2048` — do not lower below 1500
6. **llama-finetune is NOT the training tool** — it does full-parameter FP32 fine-tuning only. LoRA on NVFP4 safetensors via PEFT/TRL is the path.
