# Cosmos 3 Super — Deployment Plan
*Drafted: 2026-06-03*

## Goal

Run Cosmos 3 Super (64B) in 2-GPU parallel across two DGX Sparks, retire vintage, and redistribute the remaining stack cleanly across all four Sparks.

---

## Step 1 — Retire Vintage

Vintage is already unpromoted, ambiguous status, fail-closed. It is the only thing worth freeing to make room.

On the Spark currently hosting vintage:

```bash
# Gracefully stop vintage
sudo systemctl stop vintage   # or however it is managed in your stack
# Confirm memory is released
nvidia-smi
```

This frees ~128GB on that device.

---

## Step 2 — Download Cosmos 3 Super

NVIDIA uses Hugging Face for weights. You will need to accept the OpenMDW 1.1 license on the model page first.

- **Model card & license acceptance:** https://huggingface.co/nvidia/Cosmos3-Super
- **GitHub (inference scripts, post-training, config):** https://github.com/nvidia-cosmos/cosmos-predict2.5
  - Cosmos 3 repos are being added to the same nvidia-cosmos org; check https://github.com/nvidia-cosmos for the Cosmos3 repo as it lands
- **NVIDIA developer blog (architecture + quickstart):** https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/
- **Full platform overview:** https://www.nvidia.com/en-us/ai/cosmos/

```bash
pip install huggingface_hub
huggingface-cli login   # paste your HF token
huggingface-cli download nvidia/Cosmos3-Super --local-dir /models/cosmos3-super
```

Expect ~130GB on disk at fp16.

---

## Step 3 — 2-Spark Parallel Inference Config

NVIDIA's recommended config for Super on 2×H200 maps cleanly onto 2×GB10 (128GB each).

```bash
# Run from either of the two Cosmos Sparks
torchrun --nproc_per_node=1 --nnodes=2 \
  --node_rank=0 \            # 0 on first Spark, 1 on second
  --master_addr=<spark-A-ip> \
  --master_port=29500 \
  inference.py \
  --cfg-parallel-size 2 \
  --use-hsdp \
  --hsdp-shard-size 2 \
  --model-path /models/cosmos3-super
```

Run the mirror of this command on the second Spark with `--node_rank=1`. A single video generation takes ~3 minutes in this config per NVIDIA's own benchmark.

---

## Step 4 — Spark Allocation

| Spark | Role |
|-------|------|
| Spark A | Cosmos 3 Super — node 0 |
| Spark B | Cosmos 3 Super — node 1 |
| Spark C | super (semantic gate) + minilm (embeddings) + Ray worker |
| Spark D | omni (diagnostic, fail-closed) + Cosmos 3 Nano (optional, ~40GB) |

Cosmos 3 Nano weights: https://huggingface.co/nvidia/Cosmos3-Nano — single-Spark, real-time inference, 16B params.

---

## Step 5 — Networking

The two Cosmos Sparks need low-latency interconnect. Confirm they are on the same switch segment and that port 29500 (or chosen master port) is open between them.

```bash
# Quick check from Spark A
ping <spark-B-ip>
nc -zv <spark-B-ip> 29500
```

---

## Resources

| Resource | URL |
|----------|-----|
| Cosmos 3 Super weights | https://huggingface.co/nvidia/Cosmos3-Super |
| Cosmos 3 Nano weights | https://huggingface.co/nvidia/Cosmos3-Nano |
| nvidia-cosmos GitHub org | https://github.com/nvidia-cosmos |
| NVIDIA developer blog | https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/ |
| NVIDIA Cosmos platform page | https://www.nvidia.com/en-us/ai/cosmos/ |
| Cosmos 3 press release | https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai |
