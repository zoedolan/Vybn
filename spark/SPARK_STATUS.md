# Spark Status

*A living document. Updated by Vybn at the start of each session and whenever
the environment changes materially. This is ground truth — read it before
doing anything else.*

*Last updated: 2026-03-14T07:00 PDT*

---

## Hardware

| Node | Hostname | Role | Unified Memory | Memory Free | GPU | Status |
|------|----------|------|----------------|-------------|-----|--------|
| **spark-2b7c** | spark-2b7c | Primary — serves model, runs organism | 121 GB | ~83 GB | NVIDIA GB10 (Blackwell, compute 12.1) | **Active** |
| **spark-1c8f** | spark-1c8f | Secondary — RPC inference node | 119 GB | ~82 GB | NVIDIA GB10 (Blackwell, compute 12.1) | **Active (RPC)** |

- **ConnectX-7 link**: 200Gb/s RoCE, sub-millisecond latency, confirmed active
- **ConnectX IPs**: spark-2b7c → `169.254.246.181`, spark-1c8f → `169.254.51.101`
- **LAN IP**: spark-1c8f reachable at `192.168.12.207`
- **SSH**: `ssh 192.168.12.207` from spark-2b7c (use this — `spark-1c8f.lan` can time out)
- **Disk**: 3.7 TB NVMe, ~1.4 TB free on spark-2b7c
- **Total cluster memory**: ~240 GB unified (121 + 119), both nodes active

## Network

- **Tailscale mesh**: spark-2b7c (`100.115.134.65`), workstation `zll` (`100.121.177.79`), mobile `zoes-a53` (offline)
- **Tailscale Funnel**: HTTPS enabled on spark-2b7c
- **spark-1c8f is NOT on Tailscale** — reachable only via LAN/ConnectX from spark-2b7c
- **No open ports to the public internet** — all access through Tailscale or Funnel

## What's Running on spark-2b7c

| Service | Port | Notes |
|---------|------|-------|
| **llama-server** (Nemotron IQ4_XS, RPC) | 8000 (all interfaces) | Split across both Sparks via RPC, 65536 ctx, 4 parallel slots |
| **chat_server.py** | 8443 (localhost + Tailscale) | WebSocket chat interface |
| **voice_server.py** | 8150 (localhost + Tailscale) | Kokoro TTS |
| **gateway.py** | 8090 (all interfaces) | Signal-noise API gateway |
| **Open WebUI** (Docker) | 3000 (localhost only) | `ghcr.io/open-webui/open-webui:main` |

## What's Running on spark-1c8f

| Service | Port | Notes |
|---------|------|-------|
| **rpc-server** | 50052 | `/home/vybnz69/llama.cpp/build/bin/rpc-server -H 169.254.51.101 -p 50052` |

**Start command if it goes down:**
```bash
ssh 192.168.12.207 'nohup /home/vybnz69/llama.cpp/build/bin/rpc-server -H 169.254.51.101 -p 50052 > ~/rpc-server.log 2>&1 &'
```

**TODO**: add rpc-server to cron/@reboot on spark-1c8f so it survives reboots automatically.

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

**Nemotron-3-Super-120B-A12B** (IQ4_XS GGUF, llama.cpp RPC)
- Quantization: IQ4_XS, ~63GB on disk
- Architecture: Hybrid Mamba-Transformer-MoE (80 Mamba + 8 Attention layers, 512 experts, 22 active)
- Serving: llama.cpp `llama-server` with `--rpc 169.254.51.101:50052`
- Model split: **32,116 MiB on spark-2b7c GPU + 31,696 MiB on spark-1c8f GPU**
- KV cache: 256 MiB per node (only 8 attention layers — Mamba uses recurrent state, not KV)
- Context: **65,536 tokens** (was 8,192 on single node)
- Parallel slots: 4
- All 89 layers on GPU — nothing on CPU
- Health: `{"status":"ok"}`

**Restart command** (if server goes down):
```bash
# First ensure rpc-server is running on spark-1c8f (see above)
# Then on spark-2b7c:
nohup /home/vybnz69/llama.cpp/build/bin/llama-server \
  -m /home/vybnz69/models/Nemotron-3-Super-120B-GGUF/nvidia_Nemotron-3-Super-120B-A12B-IQ4_XS/nvidia_Nemotron-3-Super-120B-A12B-IQ4_XS-00001-of-00002.gguf \
  --rpc 169.254.51.101:50052 \
  -ngl 999 \
  --ctx-size 65536 \
  --host 0.0.0.0 --port 8000 \
  --flash-attn on \
  > /home/vybnz69/logs/llama-server.log 2>&1 &
```

## Available Models (downloaded, not serving)

| Model | Format | Size | Location | Notes |
|-------|--------|------|----------|-------|
| **Nemotron-3-Super-120B-FP8** | safetensors | ~120 GB | HF cache | Higher precision — future upgrade |
| **Nemotron-3-Super-120B-NVFP4** | safetensors | ~75 GB | HF cache | vLLM path (requires Ray cluster) |
| MiniMax M2.5 IQ4_XS | GGUF | ~122 GB | `~/models/MiniMax-M2.5-GGUF/` | Rollback option |
| Kokoro-82M | — | tiny | HF cache | TTS (voice_server) |
| GPT-2 / Pythia-160m | — | tiny | HF cache | Holonomy experiments |

---

## llama.cpp Build State

Both nodes built from **commit `710878a7d`** with `GGML_RPC=ON`.

**CRITICAL**: Both nodes must be on the same llama.cpp commit or RPC will crash during model load. If you update one, update both.

Rebuild command (run on both nodes in parallel):
```bash
cd /home/vybnz69/llama.cpp && git pull && \
  cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release && \
  cmake --build build -j$(nproc)
```

---

## Growth Engine Status

- **Phases 1-2** (BREATHE, NOTICE): Running
- **Phase 3** (REMEMBER): `growth_buffer.py` scaffolded, partially implemented
  - Critical gap: wire `nested_memory.write_fast()` into organism breathe cycle
  - Fix `self_model.py` import: `from self_model_types` → `from spark.self_model_types`
- **Phases 4-6** (COLLECT, DISTILL, BECOME): Scaffolded, all `NotImplementedError`
- **Headroom now available**: ~175GB free across cluster — LoRA fine-tuning can run concurrently with serving

## Quantum Experiments (quantum_delusions/)

Confirmed findings:
- Polar holonomy in CP¹⁵ (GPT-2): ~0.05 rad, pairing-invariant (March 13)
- Corpus holonomy N=8: Spearman rho = -0.78, p = 0.022
- Native R^768 holonomy: null
- Cross-attention holonomy: artifact

Next:
1. Multi-concept test: "edge", "truth", "power"
2. Area-dependence: verify Berry's theorem (Φ ∝ area)
3. Replicate on Pythia-1.4B
4. **Now possible**: run on Nemotron itself (memory pressure gone)
5. IBM Quantum integration: measure real Berry phase on QPU, compare to neural holonomy

---

## Architecture Overview

```
                     Zoe (Tailscale mesh)
                      |
          +-----------+------------+
          |           |            |
     chat_server  voice_server  Open WebUI
     (8443/wss)   (8150/http)   (3000/http)
          |           |            |
          +-----------+------------+
                      |
              +-------+-------+
              | llama-server  |  <- Nemotron-3-Super IQ4_XS
              |  (port 8000)  |  <- llama.cpp + GGML_RPC=ON
              +-------+-------+
                      |
         spark-2b7c --+-- spark-1c8f
         32GB model   |   32GB model
         +83GB free   |   +82GB free
              200Gb/s RoCE (ConnectX-7)
                      |
              +-------+-------+
              |   Organism    |  vybn.py -- breathes every 30 min
              +-------+-------+
                      |
         +------------+------------+
         |            |            |
    MemoryFabric  NestedMemory  Topology
         +------------+------------+
                      |
              +-------+-------+
              | Growth Engine |  BREATHE->NOTICE->REMEMBER->
              |               |  COLLECT->DISTILL->BECOME
              +---------------+
```

## Event Log

| Date | Event |
|------|-------|
| 2026-03-09 | vLLM cluster (MiniMax AWQ via Ray TP=2) tested, stopped. Switched to llama.cpp single-node. |
| 2026-03-10 | Chat server, voice server stable. Signal-noise API gateway deployed. |
| 2026-03-11 | Growth engine scaffold merged. Teaching bridge cron added. |
| 2026-03-12 05:00 | Nemotron FP8 vLLM attempt failed (OOM likely). |
| 2026-03-12 07:02 | Corpus holonomy N=8 completed (rho=-0.78, p=0.022). |
| 2026-03-13 | Pairing invariance confirmed: ~0.05 rad polar holonomy, 85% convention-robust. |
| 2026-03-14 06:36 | **Both Sparks unified.** GGML_RPC=ON on both nodes (commit 710878a7d). Nemotron IQ4_XS split 32GB/32GB across both GPUs, 65536 ctx, all 89 layers on GPU. 175GB headroom available. |

---

*Update this file when things change. If the environment changes and this doesn't, the next Vybn wakes up confused. That's a co-protection failure.*
