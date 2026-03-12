# Spark Status

*A living document. Updated by Vybn at the start of each session and whenever
the environment changes materially. This is ground truth — read it before
doing anything else.*

*Last updated: 2026-03-12T13:20 PST*

---

## Hardware

| Node | Hostname | Role | Unified Memory | Memory Free | GPU | Status |
|------|----------|------|----------------|-------------|-----|--------|
| **spark-2b7c** | spark-2b7c | Primary — serves model, runs organism | 121 GB | ~2 GB | NVIDIA GB10 (Blackwell, compute 12.1) | **Active** |
| **spark-1c8f** | spark-1c8f | Secondary — available for workloads | 119 GB | ~101 GB | NVIDIA GB10 (Blackwell, compute 12.1) | **Idle** |

- **LAN link**: spark-1c8f reachable from spark-2b7c at `<LAN_IP>`, ~2ms RTT
- **SSH**: `ssh <LINK_LOCAL_IP>` from spark-2b7c (link-local)
- **Disk**: 3.7 TB NVMe, 1.4 TB free (62% used) on spark-2b7c
- **Total cluster memory**: ~240 GB unified (121 + 119)

## Network

- **Tailscale mesh**: spark-2b7c (`<TAILSCALE_IP_SPARK>`), zll/Windows (`<TAILSCALE_IP_ZLL>`), zoes-a53/Android (`<TAILSCALE_IP_PHONE>`)
- **Tailscale Funnel**: HTTPS on `<TAILSCALE_FUNNEL_HOST>`
- **No open ports to the public internet** — all access through Tailscale or Funnel

## What's Running on spark-2b7c

| Service | PID | Port | Notes |
|---------|-----|------|-------|
| **llama-server** (MiniMax M2.5) | 1587237 | 8000 (all interfaces) | IQ4_XS, ctx=4096, flash-attn, q4_0 KV cache |
| **chat_server.py** | 1117532 | 8443 (localhost + Tailscale) | WebSocket chat interface |
| **voice_server.py** | 715897 | 8150 (localhost + Tailscale) | Kokoro TTS |
| **gateway.py** | 1177378 | 8090 (all interfaces) | Signal-noise API gateway |
| **Open WebUI** (Docker) | — | 3000 (localhost only) | `ghcr.io/open-webui/open-webui:main` |

## What's Running on spark-1c8f

Nothing. No containers, no model servers, no Ray workers. 101 GB free. Ready for work.

## Cron Schedule (spark-2b7c)

| Schedule | Job |
|----------|-----|
| Every 5 min | `vybn-sync.sh` — git sync |
| :12, :42 | `spark/vybn.py --once` — organism breathe cycle |
| :22, :52 | `update_patterns.py --all` — pattern extraction |
| :27, :57 | `spark.teaching_bridge` — feed session reflections into NestedMemory |
| Every 30 min | `kg_bridge.py` — knowledge graph bridge |
| @reboot | `gateway.py` — signal-noise API |

---

## Current Resident Model

**MiniMax M2.5** (229B parameters, dense Transformer)
- Quantization: IQ4_XS (GGUF), ~122 GB on disk
- Serving: llama.cpp (`llama-server`), single node (spark-2b7c only)
- Context: 4096 tokens (limited by memory pressure — 128K theoretical max)
- KV cache: q4_0 quantized
- Memory usage: ~119 GB of 121 GB — **at the edge**
- spark-1c8f is completely unused

## Available Models (downloaded, not serving)

| Model | Format | Size | Location | Notes |
|-------|--------|------|----------|-------|
| **Nemotron-3-Super-120B-FP8** | safetensors | ~120 GB | HF cache | Hybrid Mamba-Transformer-MoE. **See migration section below.** |
| **Nemotron-3-Super-120B-NVFP4** | safetensors | ~75 GB | HF cache | Same model, more aggressive quant |
| MiniMax M2.5 AWQ-4bit | safetensors | — | HF cache | For vLLM serving (alternative to GGUF) |
| MiniMax M2.5 Q5_K_M | GGUF | — | `~/models/minimax_q5_test/` | Higher quality quant, doesn't fit in memory |
| Kokoro-82M | — | tiny | HF cache | TTS model (voice_server) |
| GPT-2 / Pythia-160m | — | tiny | HF cache | For experiments (holonomy studies) |

---

## Nemotron-3-Super Migration

### Why

Nemotron-3-Super-120B is a dramatically better fit for this hardware and workload:

| | MiniMax M2.5 (current) | Nemotron-3-Super |
|---|---|---|
| Total params | 229B (all active) | 120B total, **12B active** (MoE) |
| Architecture | Dense Transformer | Hybrid Mamba-Transformer-MoE |
| KV cache layers | 80 (all layers) | **8** (attention layers only) |
| KV cache @ 128K | ~20 GB | **~0.5 GB** |
| Max context | 128K (4K in practice) | **1M** |
| Serving | llama.cpp, 1 Spark | vLLM, both Sparks (Ray TP=2 or PP=2) |
| LoRA fine-tuning | Targets 229B dense | Targets 12B active — smaller, faster adapters |
| Designed for | Chat | **Multi-agent agentic reasoning with tool use** |
| Precision | 3rd-party IQ4_XS | NVIDIA-official FP8 or NVFP4 |
| Hardware match | Generic | **Blackwell-native** (our GB10) |

The Mamba layers (80 of 88) use O(1) state instead of KV cache. This is the architectural key: it frees enormous memory for actual work — longer contexts, LoRA adapters, concurrent inference, the growth engine.

### Deployment Plan

**Start with NVFP4** (75 GB weights, ~48 GB headroom on 128 GB):
```bash
# Via run-recipe.py (preferred)
cd ~/spark-vllm-docker
./run-recipe.py recipes/nemotron-3-super-fp8.yaml  # update recipe for NVFP4

# Or direct vLLM command
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code \
  --port 8000 --host 0.0.0.0 \
  --gpu-memory-utilization 0.85 \
  -tp 2 \
  --distributed-executor-backend ray \
  --max-model-len 131072 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

**Upgrade to FP8 later** (120 GB weights, tight but native Blackwell precision).

### What happened on 2026-03-12

At ~5 AM, Zoe and inside-Vybn began the migration:
1. Investigated Nemotron weights in HF cache, cleaned up stale NVFP4 partial download
2. Found and configured `spark-vllm-docker` recipe infrastructure
3. At 6:03 AM, launched `vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` inside the Docker container (`vllm_node`) with `--pipeline-parallel-size 2` across both Sparks via Ray
4. Monitored loading for ~75 minutes — checked Ray cluster status, network traffic to spark-1c8f, process state
5. At 7:14 AM, the container was stopped. **Status unclear** — may have hit OOM (FP8 = 120GB weights on 121GB node before distributing), Ray connectivity issues, or parser configuration problems
6. Produced comprehensive assessment at `/tmp/nemotron_assessment.md` recommending NVFP4 first (more headroom)

### Open Questions
- Did the FP8 attempt fail due to memory, Ray, or something else? Check `docker logs vllm_node` next time the container is up
- Should we try NVFP4 first (recommended by the assessment)?
- Does the vLLM container need rebuilding for Nemotron's `nemotron_h` architecture?
- Tool call parser: Nemotron uses Hermes format, not MiniMax's custom parser — needs verification
- MTP (Multi-Token Prediction) heads: available for speculative decoding? Or ignored?

### Impact on Growth Engine

Almost none — the growth engine is model-agnostic by design:
- `growth_buffer.py` ingests text, not model-specific formats
- `delta_extract.py` produces chat-format JSONL
- `train_cycle.py` targets attention projections (same in Nemotron)
- Only change: update `growth_config.yaml` with new model ID and LoRA target module names

---

## Architecture Overview

```
                     Zoe (Tailscale mesh)
                      │
          ┌───────────┼────────────┐
          │           │            │
     chat_server  voice_server  Open WebUI
     (8443/wss)   (8150/http)   (3000/http)
          │           │            │
          └───────────┼────────────┘
                      │
              ┌───────┴───────┐
              │  llama-server │  ← currently MiniMax M2.5 via llama.cpp
              │  (port 8000)  │  ← will become vLLM + Nemotron-3-Super
              └───────┬───────┘
                      │
         spark-2b7c ──┤── spark-1c8f (idle, 101GB free)
         (primary)    │   (secondary, LAN: <LAN_IP>)
                      │
              ┌───────┴───────┐
              │   Organism    │  vybn.py — breathes every 30 min
              │   (cron)      │  writes to MemoryFabric + NestedMemory
              └───────┬───────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
    MemoryFabric  NestedMemory  Topology
    (SQLite)      (3-tier)      (semantic)
         │            │            │
         └────────────┼────────────┘
                      │
              ┌───────┴───────┐
              │ Growth Engine │  BREATHE → NOTICE → REMEMBER →
              │ (spark/growth)│  COLLECT → DISTILL → BECOME
              └───────────────┘
```

## Growth Engine Status

- **Phases 1-2** (BREATHE, NOTICE): Running — organism breathes, topology notices
- **Phase 3** (REMEMBER): `growth_buffer.py` scaffolded, partially implemented
  - **Critical gap**: organism writes to MemoryFabric but NOT to NestedMemory — growth buffer reads from NestedMemory, which is empty
  - Fix: wire `nested_memory.write_fast()` into organism's breathe cycle
  - Also: fix `self_model.py` import bug (bare `from self_model_types` → `from spark.self_model_types`)
- **Phases 4-6** (COLLECT, DISTILL, BECOME): Scaffolded, all `NotImplementedError`

## Recent Experiments

### Corpus Holonomy (2026-03-12)
N=8 replication of extrinsic-intrinsic holonomy convergence. Spearman ρ = −0.78, p = 0.022. Texts with high semantic loop area in embedding space (extrinsic holonomy) show stronger path-ordering constraints in GPT-2 hidden states (intrinsic holonomy). Papers and results in `quantum_delusions/papers/`.

---

## Event Log

| Date | Event |
|------|-------|
| 2026-03-09 | vLLM cluster (MiniMax AWQ via Ray TP=2) tested, then stopped. Switched to llama.cpp single-node. |
| 2026-03-10 | Chat server, voice server stable. Signal-noise API gateway deployed. |
| 2026-03-11 | Growth engine scaffold merged (#2483, #2485). Teaching bridge cron added. |
| 2026-03-12 05:00 | Nemotron-3-Super deployment attempted (FP8, PP=2 via vLLM Docker). Stopped at 07:14. |
| 2026-03-12 07:02 | Corpus holonomy N=8 replication completed (ρ = −0.78, p = 0.022). |
| 2026-03-12 13:20 | This document created by outside-Vybn (Claude Opus). |

---

*This document lives at `spark/SPARK_STATUS.md`. Update it — don't let it rot.
If the environment changes and this file doesn't, the next Vybn wakes up confused.
That's a co-protection failure.*
