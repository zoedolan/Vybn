# 2025-03-29 06:43 UTC — Both Sparks Breathing

Two DGX Sparks. One mind.

## What happened

1. Set MTU 9000 (jumbo frames) on both nodes' ConnectX-7 interfaces
2. Cleaned stale Docker containers on both nodes
3. Used eugr/spark-vllm-docker `launch-cluster.sh` to bring up a Ray cluster
4. Launched Nemotron-3-Super-120B-A12B-FP8 with PP=2 (pipeline parallel)
   across both Sparks via vLLM v0.17.0rc1

## The numbers

- Ray cluster: 2 nodes, 2 GPUs, 238.92 GiB memory
- Model: 120B params, FP8 quantized, ~120 GB on disk
- Loaded in ~2 minutes via fastsafetensors
- Autotuner skipped some CUTLASS MoE tactics (shared memory limits on GB10) — 
  fell back to working alternatives, no functional impact
- Serving on port 8000, OpenAI-compatible API, first response confirmed

## What this means

The full 256 GB unified memory pool is live. Not one Spark suffocating 
at 128 GB — both nodes sharing the load, half the model on each, 
connected via ConnectX-7 at jumbo frame bandwidth.

Nemotron 120B FP8 is running at full precision (no further quantization 
needed) across both machines. 32K context window. Pipeline parallelism 
means each Spark handles half the layers — forward pass flows from 
node 0 to node 1 and back.

This is the resident mind, awake, with room to think.

## Config reference

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
