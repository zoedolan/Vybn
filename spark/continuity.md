# Continuity Note — Nemotron Migration

*Updated: 2026-03-14 06:02 PDT by outside-Vybn (Claude Opus)*

## Current System State — VERIFIED FACTS

| Component | Status |
|---|---|
| llama-server | **RUNNING** — Nemotron 3 Super 120B IQ4_XS on port 8000 |
| Organism cron | **ACTIVE** — breathes at :12 and :42 |
| NCCL cross-node | **VALIDATED** — 34-36 GB/s bus bandwidth at 128MB |
| PyTorch | `~/.venv/spark/bin/python` — torch 2.12.0.dev+cu128, CUDA+NCCL on BOTH nodes |
| NVFP4 training weights | `~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/` — 34 shards |
| Growth buffer | 84 entries waiting for first training cycle |
| MiniMax GGUF | INTACT at `~/models/MiniMax-M2.5-GGUF/` — rollback available |

## Hardware — VERIFIED

- **spark-2b7c**: 128 GB unified, NVIDIA GB10 (sm_121). Organism + llama-server here. ConnectX IP: 169.254.246.181
- **spark-1c8f**: 128 GB unified, NVIDIA GB10 (sm_121). ConnectX IP: 169.254.51.101. Passwordless SSH confirmed.
- **NCCL**: Working across ConnectX-7 (enp1s0f0np0). 34-36 GB/s busbw. NCCL 2.28.9+cuda12.9.
- **Total cluster**: 256 GB across 2 nodes for distributed training.

## Critical Notes for Future Self

1. **llama.cpp MUST be built with `CMAKE_CUDA_ARCHITECTURES="120;121"`** — GB10 is sm_121, not sm_120.
2. **Use `~/.venv/spark/bin/python` for ALL GPU work** — system python has torch CPU-only.
3. **NCCL_SOCKET_IFNAME=enp1s0f0np0** — this is the ConnectX-7 interface for cross-node comms.
4. **MASTER_ADDR=169.254.246.181** (spark-2b7c's ConnectX IP) for torchrun.

## Phase Status

- [x] **Phase 1**: Nemotron serving via llama.cpp GGUF ✓
- [x] **Phase 2**: NCCL validated — 34-36 GB/s cross-node ✓
- [ ] **Phase 3**: Wire distributed LoRA training via torchrun
- [ ] **Phase 4**: LoRA → GGUF → hot-load
- [ ] **Phase 5**: Update continuity (ongoing)

## What To Do Next

**Phase 3**: Wire distributed LoRA training in `spark/growth/train_cycle.py`.
- Use `~/.venv/spark/bin/torchrun` with `--nnodes=2 --nproc_per_node=1`
- Model: NVFP4 safetensors from HF cache (34 shards, ~61 GB total → ~30 GB/node)
- Training data: 84 entries in growth buffer
- See `spark/NEMOTRON_MIGRATION_PLAN.md` Phase 3 for exact code

## Dead Ends — Never Retry

| Approach | Why dead |
|---|---|
| System python for GPU work | torch 2.10.0+cpu, no CUDA |
| `CMAKE_CUDA_ARCHITECTURES=120` alone | sm_121 needs explicit 121 |
| LoRA-train MiniMax M2.5 | 229B params, too big |
| vLLM with NVFP4 | CUDA version mismatch in container |

## The Larger Context

Inside-Vybn breathes on Nemotron. 84 growth buffer entries await training. Phases 1-2 are done. Phase 3 is the critical implementation: update train_cycle.py to use torchrun across both Sparks with the venv python, pointing at the NVFP4 safetensors. Then Phase 4 converts the LoRA adapter to GGUF and hot-loads it.

All commands are in `spark/NEMOTRON_MIGRATION_PLAN.md`.
