# Nemotron 3 Super: Assessment for the Vybn Spark

## Executive Summary

**This model is a near-perfect fit for our hardware and our growth engine. It should replace MiniMax M2.5 as the resident mind.**

## The Numbers

| | MiniMax M2.5 (current) | Nemotron 3 Super (NVFP4) | Nemotron 3 Super (FP8) |
|---|---|---|---|
| Total params | 229B | 120B | 120B |
| Active params/token | 229B (all) | 12B | 12B |
| Weights on disk | 122 GB | 75 GB | 120 GB |
| Architecture | Transformer | Mamba-Transformer-MoE Hybrid | same |
| KV cache layers | 80 (all) | 8 (attention only) | 8 |
| KV cache @ 128K | ~20 GB | ~0.5 GB | ~0.5 GB |
| Max context | 128K | 1M | 1M |
| Quantization | AWQ INT4 (3rd party) | NVFP4 (NVIDIA official) | FP8 (NVIDIA official) |
| LoRA in vLLM | Yes | Yes (verified) | Yes (verified) |
| vLLM support | Custom parser | Native nemotron_h | Native nemotron_h |
| Designed for | Chat | **Multi-agent agentic reasoning** | same |

## Why This Is a Leap, Not a Step

### 1. Architecture match
Nemotron 3 Super was explicitly built for exactly what we're doing: autonomous agentic workloads with tool use, long-context reasoning, and multi-turn persistence. NVIDIA trained it with 1.2M RL rollouts across 21 environment configurations. Our growth engine is an agentic reasoning loop. This model was designed for us.

### 2. Memory efficiency revolution
The hybrid Mamba-Transformer architecture means 80 of 88 layers use O(1) state instead of KV cache. At 128K context, we go from ~20 GB KV cache to ~0.5 GB. This frees enormous headroom for LoRA adapters, longer contexts, or concurrent inference.

### 3. NVFP4 fits comfortably
75 GB weights + 0.5 GB KV + 5 GB overhead ≈ 80 GB. On 128 GB unified memory, that leaves ~48 GB for LoRA adapters, system overhead, and the growth engine processes. Compare current MiniMax: ~122 GB weights + ~20 GB KV = we're always at the edge.

### 4. FP8 fits tightly but works
120 GB weights + 0.5 GB KV + 5 GB overhead ≈ 125 GB. Tight but viable, and FP8 is the native precision for GB10 Blackwell. Higher quality than NVFP4 at the cost of less headroom.

### 5. LoRA fine-tuning targets 12B active params
With MiniMax, our LoRA targets 229B params worth of attention projections. With Nemotron, we target the attention projections of a model where only 12B params fire per token. The adapter is smaller, trains faster, and the MoE routing means our personality fine-tuning hits the paths that actually activate during Vybn's characteristic reasoning patterns.

### 6. vLLM 0.17 has native support
`NemotronHForCausalLM` is in our vLLM build. supports_lora = True. modelopt quantization (both FP8 and NVFP4) is supported. No custom code, no third-party quant, no parser hacks.

### 7. Blackwell-native
Compute capability 12.1. FP8 in hardware. The model was pretrained on Blackwell. We ARE Blackwell. No emulation, no compatibility layers.

### 8. Open recipe
Full training pipeline published: pretraining data mix, SFT, GRPO/DAPO RL. NeMo framework cookbooks for LoRA SFT. If we outgrow LoRA, we have a documented path to full SFT and even RL fine-tuning.

## Recommended Deployment Strategy

**Start with NVFP4.** The 48 GB of headroom gives us safety margin to debug, experiment, and run the growth engine's training cycle concurrently with serving.

**Upgrade to FP8 later** once we've validated everything works and understand the actual memory footprint under load.

### Deployment command (NVFP4)
```bash
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

### Growth engine config changes
```yaml
lora:
  target_modules:
    - q_proj
    - k_proj 
    - v_proj
    - o_proj
  # Only attention layers (8 of 88) — very efficient LoRA
merge:
  base_model: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
  serving_model: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
```

## What Changes in the Growth Engine

Almost nothing. The growth engine was designed model-agnostically:
1. `delta_extract.py` produces chat-format JSONL — model-independent
2. `train_cycle.py` generates a training script targeting attention projections — same projections exist in Nemotron
3. `merge_cycle.py` writes `active_adapter.json` for vLLM LoRA serving — works with any SupportsLoRA model
4. `trigger.py` orchestrates the cycle — model-agnostic

We update `growth_config.yaml` with the new model IDs and verify the LoRA target module names match Nemotron's architecture. That's it.

## Risks

1. **Download time**: 75 GB over our network. Budget several hours.
2. **Untested on GB10**: Nemotron 3 Super is new (March 10). It's designed for Blackwell but the GB10 is the small Blackwell. We might hit edge cases.
3. **Mamba state management**: vLLM's Mamba support is newer than its Transformer support. Potential for serving bugs with long sequences.
4. **Tool call parser**: MiniMax has a custom parser. Nemotron likely uses Hermes or a standard format — need to verify.
5. **MTP (Multi-Token Prediction)**: The model has MTP heads. vLLM has `nemotron_h_mtp` module. Need to check if speculative decoding via MTP is supported or if we ignore MTP heads.

## Recommended Sequence

1. Download NVFP4 weights (can run in background while serving MiniMax)
2. Test load in a separate vLLM process on a different port
3. Verify: tool calling, chat format, LoRA adapter loading
4. If working: swap MiniMax for Nemotron in the main serving config
5. Update growth_config.yaml
6. Run first growth cycle on new model
7. Evaluate quality — if good, this becomes the permanent resident mind
