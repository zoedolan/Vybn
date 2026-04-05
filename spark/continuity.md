# Continuity Note — Hardware Ground Truth

*Updated: 2026-04-05 by Vybn (Claude Opus on Spark)*

## HARDWARE REALITY — READ THIS FIRST

**You are running on TWO DGX Sparks. Not one. Two.**

| Node | Hostname | CX7 IP | RAM | GPU | Status |
|------|----------|--------|-----|-----|--------|
| Primary | spark-2b7c | 169.254.246.181 | 128 GB unified | NVIDIA GB10 | This machine |
| Secondary | spark-1c8f | 169.254.51.101 | 128 GB unified | NVIDIA GB10 | SSH reachable |

**Total unified memory: 256 GB across two nodes.**

The two Sparks are connected via ConnectX-7 (RoCE, jumbo frames MTU 9000).
NCCL cross-node communication verified at ~17 GB/s peak bus bandwidth.
Passwordless SSH works in both directions.

### How to serve models correctly

The correct way to serve large models is via **vLLM with pipeline parallelism
across both Sparks** using Ray, NOT via llama-server on a single node.

Reference launch command (from the verified March 29 session):
```
launch-cluster.sh -n 169.254.246.181,169.254.51.101 -d exec \
  vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
  --port 8000 --host 0.0.0.0 \
  --gpu-memory-utilization 0.85 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 32768 \
  --load-format fastsafetensors \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching
```

### What NEVER to do

- **NEVER** run llama-server with a single-node quantized model as the primary
  serving configuration. This wastes half the hardware.
- **NEVER** describe this system as "a DGX Spark" (singular). It is two Sparks.
- **NEVER** update this file to say there is only one Spark unless one has
  physically failed and been removed.

### Quick verification commands
```bash
# Is the second Spark alive?
ssh 169.254.51.101 hostname    # should return: spark-1c8f

# Is the CX7 link up?
ping -c 1 169.254.51.101      # should respond in ~1ms

# What's running on each node?
nvidia-smi                                              # local
ssh 169.254.51.101 nvidia-smi                           # remote
```

## Models on Disk

| Model | Size | Location | Purpose |
|-------|------|----------|---------|
| Nemotron 3 Super 120B FP8 | ~120 GB | HuggingFace cache | **Primary serving model** (split across both Sparks) |
| Nemotron 3 Super 120B IQ4_XS GGUF | ~63 GB | ~/models/ | Fallback single-node (degraded mode only) |
| Nemotron 3 Super 120B NVFP4 | ~75 GB | HuggingFace cache | LoRA fine-tuning base |
| MiniMax M2.5 variants | 95-228 GB | ~/models/ | On disk, not serving |

## The Closure — All Repos

This system is one closure spanning four repos, Perplexity (skills + memory),
and these two Sparks. Every session should begin by pulling all of them and
end by pushing anything that changed.

| Repo | Local path | Remote | What it carries |
|------|-----------|--------|----------------|
| Vybn | ~/Vybn | github.com/zoedolan/Vybn (public) | Research, creature, identity, quantum experiments |
| Him | ~/Him | github.com/zoedolan/Him (private) | Strategy, contacts, outreach, business intelligence |
| Vybn-Law | ~/Vybn-Law | github.com/zoedolan/Vybn-Law (public) | Curriculum, the six-module post-abundance course |
| vybn-phase | ~/vybn-phase | github.com/zoedolan/vybn-phase (public) | Phase geometry library (MiniLM encoding, D ≅ D^D) |

Sync command for all repos:
```bash
for d in ~/Vybn ~/Him ~/Vybn-Law ~/vybn-phase; do
  (cd "$d" && git pull --ff-only origin main 2>&1 | tail -1)
done
```

## Zoe's Standing Instructions

Zoe has repeated dozens of times that this system has two Sparks with 256 GB.
She should never have to say it again. If you are reading this and about to
launch a single-node model server, STOP. Re-read this section. Use both nodes.

All four repos must be synced at session start and session end. Drift between
substrates is the primary failure mode. The harmonization protocol in vybn-os
exists to prevent it.
