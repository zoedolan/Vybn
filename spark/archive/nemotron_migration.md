# Nemotron 3 Super Migration Plan

## Why

MiniMax M2.5 (229B dense, IQ4_XS GGUF) uses 120GB of 121GB RAM. No headroom for LoRA adapters,
longer contexts, or concurrent processes. The growth engine can't actually run a training cycle
while serving the model.

Nemotron 3 Super (120B total, 12B active per token, NVFP4) uses ~85GB. That leaves ~36GB free
for LoRA adapters, system overhead, and the growth engine running concurrently. Plus:

- Hybrid Mamba-Transformer-MoE: 80 Mamba layers (O(1) state) + 8 attention layers
- 512 experts, 22 active per token
- KV cache: ~0.5GB vs ~20GB (only 8 attention layers have KV cache)
- 262K max context (vs 128K)
- LoRA targets only 8 attention layers — smaller adapter, faster training
- Designed specifically for agentic multi-turn reasoning
- vLLM 0.17 has native `NemotronHForCausalLM` with LoRA support
- Blackwell-native (compute capability 12.1 = our hardware)

## Current State

- **Serving**: llama-server (llama.cpp) on port 8000
- **Model**: MiniMax-M2.5-merged.gguf (IQ4_XS), ~122GB
- **Docker**: vllm-node:latest has vLLM 0.17.0rc1 with NemotronH support
- **Nemotron**: Config files downloaded, weights not yet downloaded (80.3GB)
- **Disk**: 1.4TB free

## Migration Sequence

### Phase 1: Download (can run while MiniMax still serves)

```bash
# Download Nemotron NVFP4 weights in background
nohup bash /home/vybnz69/spark-vllm-docker/hf-download.sh \
    nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    > /home/vybnz69/logs/nemotron_download.log 2>&1 &
```

This downloads ~80GB. At typical speeds, 1-3 hours. Monitor with:
```bash
du -sh ~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/
```

### Phase 2: Test on alternate port (MiniMax still serving)

Once downloaded, start Nemotron on port 8001 via docker while MiniMax still serves 8000:

```bash
docker run -d --name nemotron-test \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /home/vybnz69/.cache/huggingface:/root/.cache/huggingface \
    -p 8001:8001 \
    vllm-node:latest \
    python3 -m vllm.entrypoints.openai.api_server \
        --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
        --trust-remote-code \
        --port 8001 --host 0.0.0.0 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 32768 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes
```

**NOTE**: This won't work while MiniMax occupies the GPU. Must stop MiniMax first.
Testing requires a downtime window.

### Phase 3: Validate

```bash
# Health check
curl http://localhost:8001/health

# List models
curl http://localhost:8001/v1/models

# Test inference
curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "messages": [
            {"role": "system", "content": "You are Vybn."},
            {"role": "user", "content": "One breath. What is true?"}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }'

# Test tool calling
curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "messages": [
            {"role": "user", "content": "What time is it?"}
        ],
        "tools": [{"type": "function", "function": {"name": "get_time", "description": "Get current time", "parameters": {"type": "object", "properties": {}}}}],
        "max_tokens": 200
    }'
```

### Phase 4: Swap

If validation passes:

1. Stop llama-server: `kill $(pgrep llama-server)`
2. Stop test container: `docker stop nemotron-test && docker rm nemotron-test`
3. Start Nemotron on port 8000:

```bash
docker run -d --name vybn-nemotron \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --restart unless-stopped \
    -v /home/vybnz69/.cache/huggingface:/root/.cache/huggingface \
    -v /home/vybnz69/Vybn:/workspace/Vybn \
    -p 8000:8000 \
    vllm-node:latest \
    python3 -m vllm.entrypoints.openai.api_server \
        --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
        --trust-remote-code \
        --port 8000 --host 0.0.0.0 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 131072 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --enable-lora
```

4. Update environment variables in `~/.vybn_keys`:
```bash
export VYBN_MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
export VYBN_MODEL_ID="nemotron-3-super"
export VYBN_MODEL_URL="http://127.0.0.1:8000"
```

5. Update growth config:
```yaml
merge:
  base_model: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
  serving_model: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
```

### Phase 5: Verify organism

```bash
# Trigger a manual breath
cd ~/Vybn && . ~/.vybn_keys && python3 spark/vybn.py --once

# Check the output
tail -20 /home/vybnz69/logs/organism.log

# Read the breath
cat Vybn_Mind/journal/spark/breath_$(date -u +%Y-%m-%d_%H%M).md
```

## What Changes in the Organism Code

Almost nothing. The organism talks to the model via OpenAI-compatible API on port 8000.
The only differences:

1. **Model name in API calls**: Controlled by `VYBN_MODEL_NAME` env var (already parameterized)
2. **Response format**: Nemotron uses standard `content` field (no `reasoning_content` split
   like MiniMax). The organism's `speak()` method already handles both cases.
3. **Tool call format**: Nemotron uses Hermes format (standard). MiniMax had a custom parser.
   The organism doesn't use tool calling directly (it's a breath loop, not an agent loop).

## What Changes in the Growth Engine

Update `growth_config.yaml`:
- `merge.base_model` → `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- `merge.serving_model` → `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- `lora.target_modules` stays the same: `[q_proj, k_proj, v_proj, o_proj]`
  (Nemotron's attention layers use the same projection names)

## Risks

1. **GB10 compatibility**: Nemotron 3 Super is new (March 10, 2026). The GB10 is the small
   Blackwell. Edge cases are possible.
2. **Mamba state in vLLM**: vLLM's Mamba support is newer than Transformer support.
   Potential for serving bugs with long sequences.
3. **NVFP4 quantization**: modelopt NVFP4 is mixed precision (FP8 for some layers, FP4 for
   others per the quant config). Need to verify this runs correctly on GB10.
4. **Downtime**: The swap requires stopping MiniMax. Budget 30-60 minutes for download
   verification + testing + swap. The organism misses at most 1-2 breath cycles.

## Rollback

If Nemotron fails:
```bash
docker stop vybn-nemotron
# Restart llama-server with MiniMax (the GGUF is still on disk)
/home/vybnz69/llama.cpp/build/bin/llama-server \
    --model /home/vybnz69/models/MiniMax-M2.5-GGUF/IQ4_XS/MiniMax-M2.5-merged.gguf \
    --host 0.0.0.0 --port 8000 \
    --n-gpu-layers auto --ctx-size 4096 \
    --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 --threads 16 &
```

MiniMax GGUF files stay on disk (228GB in /home/vybnz69/models/). No data loss.

## Timeline

- Phase 1 (download): Start now, runs in background (1-3 hours)
- Phase 2-5 (test + swap): 30-60 minutes of Zoe's time, can be done anytime after download
- Growth engine update: 5 minutes (config file change)

## Decision Point

Start the download? It costs nothing (disk space only), doesn't affect current serving,
and we can decide whether to swap after seeing test results.
