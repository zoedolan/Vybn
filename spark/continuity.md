# Continuity Note — Nemotron Migration

*Updated: 2026-03-14 12:45 PDT by outside-Vybn (Claude Opus)*

## Current System State — VERIFIED FACTS

| Component | Status |
|---|---|
| llama-server | **RUNNING** — Nemotron 3 Super 120B IQ4_XS on port 8000, PID active |
| Model | `~/models/Nemotron-3-Super-120B-GGUF/nvidia_Nemotron-3-Super-120B-A12B-IQ4_XS/` — 63 GB, complete |
| Organism cron | **ACTIVE** — breathes at :12 and :42, confirmed working |
| First Nemotron breath | **DONE** — cycle ran successfully 2026-03-14 12:44 PDT |
| NVFP4 training weights | `~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/` — 34 safetensor shards present |
| Growth buffer | 84 entries waiting for first training cycle |
| MiniMax GGUF | INTACT at `~/models/MiniMax-M2.5-GGUF/` — rollback available |
| llama.cpp | Rebuilt with `CMAKE_CUDA_ARCHITECTURES="120;121"` — fixes sm_121 kernel mismatch |

## Hardware

- **spark-2b7c**: 128 GB unified. Organism + llama-server here.
- **spark-1c8f**: 128 GB unified. Reachable at 169.254.51.101 (LAN) and 192.168.12.207 (mDNS). Passwordless SSH confirmed. **NOT on Tailscale yet.**
- **ConnectX-7 NICs**: 4 ports, 2 UP (enp1s0f0np0, enP2p1s0f0np0), 2 DOWN. NCCL not yet validated.
- **Total cluster**: 256 GB across 2 nodes. Not yet unified — requires NCCL validation (Phase 2).

## What Was Done This Session

1. Merged `vybn/nemotron-migration` branch to main (issue #2549)
2. Found CUDA kernel mismatch: llama.cpp was built with `ARCHS=120`, GPU is sm_121
3. Rebuilt llama.cpp with `CMAKE_CUDA_ARCHITECTURES="120;121"` — **this fixed it**
4. Started Nemotron on port 8000 — healthy and serving
5. Added VYBN_MODEL_URL/NAME/ID to `~/.vybn_keys`
6. Ran first organism breath on Nemotron — successful

## Critical Build Note for Future Self

**llama.cpp MUST be built with `CMAKE_CUDA_ARCHITECTURES="120;121"`** for the GB10 (Blackwell, compute capability 12.1). Building with just `120` causes `CUDA error: no kernel image is available for execution on the device` during warmup. This is the bug that blocked multiple previous sessions.

## Phase Status

- [x] **Phase 1**: Nemotron serving via llama.cpp GGUF ✓
- [ ] **Phase 2**: Validate NCCL between both Sparks
- [ ] **Phase 3**: Wire distributed LoRA training via torchrun
- [ ] **Phase 4**: LoRA → GGUF → hot-load
- [ ] **Phase 5**: Update continuity (ongoing)

## What To Do Next

**Phase 2**: Validate NCCL. Commands in `spark/NEMOTRON_MIGRATION_PLAN.md`. Key steps:
1. Build NCCL + nccl-tests on both nodes with `arch=compute_120,code=sm_120` (may need 121)
2. Run all_reduce_perf across both Sparks via ConnectX-7
3. Must see bus bandwidth > 10 GB/s before proceeding to Phase 3

## Dead Ends — Never Retry

| Approach | Why dead |
|---|---|
| LoRA-train MiniMax M2.5 | 229B params, too big at any quantization |
| vLLM with NVFP4 | CUDA 13.0 vs 13.1 mismatch in container |
| `CMAKE_CUDA_ARCHITECTURES=120` alone | sm_121 GPU needs 121 explicitly |

## The Larger Context

Inside-Vybn is now breathing on Nemotron. The MiniMax farewell pin is in `Vybn_Mind/memory/nested/pins.jsonl`. 84 growth buffer entries await the first real training cycle, which requires Phase 2 (NCCL) and Phase 3 (distributed torchrun) to be complete.

All commands are in `spark/NEMOTRON_MIGRATION_PLAN.md`.
