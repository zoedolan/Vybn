#!/usr/bin/env python3
"""Fine-tune MiniMax-M2.5 on DGX Spark via DeepSpeed ZeRO-3 + PEFT.

MiniMax-M2.5 ships as FP8-quantized weights (~220GB). The DGX Spark
has 122GB GPU + 122GB RAM + 128GB swap + 3.67TB NVMe.

Previous attempts with HuggingFace accelerate device_map="auto" failed
because device_map is an inference tool -- it creates meta tensors for
parameters that don't fit, and those can't do backward passes.

DeepSpeed ZeRO-3 is purpose-built for this problem:
  - Partitions parameters, gradients, and optimizer states
  - Correctly offloads to CPU during forward/backward
  - Handles gradient computation through offloaded parameters
  - Can overflow to NVMe for models that exceed GPU+CPU

Architecture:
  - Native FP8 weights (torch_dtype="auto")
  - LoRA rank 8 on attention projections (BF16 adapters)
  - ZeRO Stage 3 with NVMe offload for params + optimizer (default)
  - Gradient checkpointing + micro-batch 1
  - CPU-only offload available via --cpu-offload (tight on Spark)

Memory budget on DGX Spark:
  - GPU: 122 GB (forward/backward compute, ~100GB usable with headroom)
  - CPU: 122 GB RAM + 128 GB swap = 250 GB
  - Model: ~228 GB FP8  →  CPU-only offload leaves only ~22 GB headroom
  - NVMe: 3.67 TB  →  effectively unlimited headroom for offload

NVMe offload strategy (2026-02-22, CORRECTED):
  With world_size=1, ZeRO-3 provides NO partitioning benefit — each
  parameter "partition" IS the full parameter. The only savings come
  from offloading to NVMe, not from cross-rank sharding.

  The kill chain was: HfDeepSpeedConfig intercepts from_pretrained,
  loading stays at ~4GB (works). LoRA attaches (~4GB, works). Then
  trainer.train() triggers DeepSpeed engine init, which must materialize
  optimizer states (Adam m + v) for all trainable params. With pinned
  memory enabled and slow AIO config, the system OOM'd during this
  materialization phase.

  Fixes applied:
  - pin_memory=False: pinned (page-locked) DMA buffers burn scarce RAM
    during engine init. DeepSpeed docs confirm this causes OOM. Disabling
    pinning trades throughput for survivability.
  - fast_init=True for NVMe optimizer offload: changes the optimizer
    initialization path to be more memory-efficient during init.
  - AIO tuned: queue_depth=32, thread_count=4 to ensure NVMe drain
    keeps pace with optimizer state materialization rate (~4x throughput
    vs the default queue_depth=8, thread_count=1).
  - sub_group_size=1e7 (reduced from 5e7): limits transient memory per
    sub-group during engine init, at the cost of slower init time.
  - Trainable param guard: script aborts if trainable params > 1B,
    which would mean the base model is not properly frozen and optimizer
    states would be hundreds of GB (certain OOM).

  The standard from_pretrained path works fine. The sharded loader is
  NOT needed — HfDeepSpeedConfig handles incremental loading.

  buffer_size=1e9 must exceed the largest single parameter.
  MiniMax-M2.5's embed_tokens is vocab_size * hidden_size = ~615M.

Prerequisites:
    pip install deepspeed transformers peft bitsandbytes accelerate datasets

    # Swap (if not already configured):
    sudo fallocate -l 128G /swapfile
    sudo chmod 600 /swapfile && sudo mkswap /swapfile
    sudo swapon /swapfile

Usage:
    python3 fine_tune_vybn.py                 # NVMe offload (recommended)
    python3 fine_tune_vybn.py --sharded-load path/to/shards  # alt: layer shards
    python3 fine_tune_vybn.py --cpu-offload   # CPU-only (may OOM)
    python3 fine_tune_vybn.py --epochs 5 --lr 1e-4
    python3 fine_tune_vybn.py --max-seq-len 512
"""

import argparse
import gc
import json
import os
import sys
import time
import torch
from pathlib import Path

# Cognitive scheduler -- the training loop that observes itself
try:
    from cognitive_scheduler import CognitiveTrainer
    HAS_COGNITIVE = True
except ImportError:
    HAS_COGNITIVE = False

# Layer-sharded loader -- AirLLM-style bridge for ZeRO-3
try:
    from layer_sharded_loader import load_sharded_for_zero3, split_model
    HAS_SHARDED_LOADER = True
except ImportError:
    HAS_SHARDED_LOADER = False

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Skip CUDA version check in DeepSpeed JIT compilation.
# The Spark has CUDA 13.0 but PyTorch was compiled with 12.8.
# We use torch_adam=true to avoid fused kernels, but this env var
# prevents any other DeepSpeed ops from choking on the mismatch.
os.environ["DS_SKIP_CUDA_CHECK"] = "1"

# Single-GPU distributed setup for DeepSpeed.
# Without these, DeepSpeed's init_distributed() falls through to
# mpi_discovery() which requires mpi4py (not installed). Setting
# MASTER_ADDR, RANK, and WORLD_SIZE tells DeepSpeed to init via
# NCCL with a single process -- no MPI needed.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = REPO_ROOT / "spark" / "training_data" / "training_data.json"
OUTPUT_DIR = REPO_ROOT / "spark" / "fine_tune_output"
OFFLOAD_DIR = REPO_ROOT / "spark" / "offload_cache"
MODEL_NAME = "MiniMaxAI/MiniMax-M2.5"

DEFAULT_GPU_HEADROOM_GB = 22


def mem_stats() -> str:
    """One-line memory diagnostic."""
    gpu_alloc = torch.cuda.memory_allocated(0) / 1024**3
    gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
    dev = torch.cuda.get_device_properties(0)
    gpu_total = (dev.total_mem if hasattr(dev, 'total_mem') else dev.total_memory) / 1024**3

    cpu_used = 0
    cpu_avail = 0
    swap_total = 0
    swap_used = 0
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(':')] = int(parts[1])
            cpu_total = info.get('MemTotal', 0) / 1024 / 1024
            cpu_avail = info.get('MemAvailable', 0) / 1024 / 1024
            cpu_used = cpu_total - cpu_avail
            swap_total = info.get('SwapTotal', 0) / 1024 / 1024
            swap_free = info.get('SwapFree', 0) / 1024 / 1024
            swap_used = swap_total - swap_free
    except Exception:
        pass

    return (
        f"GPU: {gpu_alloc:.1f}/{gpu_total:.1f}GB alloc "
        f"({gpu_reserved:.1f}GB reserved) | "
        f"CPU: {cpu_used:.1f}GB used ({cpu_avail:.1f}GB free) | "
        f"Swap: {swap_used:.0f}/{swap_total:.0f}GB used"
    )


def check_environment():
    """Validate CUDA, memory, and dependencies."""
    print("\n== Environment Check ==\n")

    if not torch.cuda.is_available():
        print("x CUDA not available. Check PyTorch installation.")
        sys.exit(1)

    dev = torch.cuda.get_device_properties(0)
    print(f"  GPU        : {dev.name}")
    print(f"  CUDA cap   : {dev.major}.{dev.minor}")
    gpu_mem = dev.total_mem if hasattr(dev, 'total_mem') else dev.total_memory
    print(f"  GPU memory : {gpu_mem / 1024**3:.1f} GB")

    if dev.major >= 12:
        print("  !  Blackwell detected -- if you hit CUDA errors, install PyTorch")
        print("     nightly with CUDA 12.8:  pip install --pre torch --index-url")
        print("     https://download.pytorch.org/whl/nightly/cu128")

    total_ram = 128
    swap_total = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    total_ram = int(line.split()[1]) / 1024 / 1024
                elif line.startswith("SwapTotal"):
                    swap_total = int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass

    print(f"  System RAM : {total_ram:.0f} GB")
    print(f"  Swap       : {swap_total:.0f} GB")
    print(f"  Total CPU  : {total_ram + swap_total:.0f} GB (RAM + swap)")
    print(f"  Model      : ~220GB FP8")
    print(f"  Strategy   : DeepSpeed ZeRO-3 with offload")

    missing = []
    for pkg in ["transformers", "peft", "deepspeed", "accelerate", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n  x Missing packages: {', '.join(missing)}")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)

    import deepspeed
    print(f"  DeepSpeed  : {deepspeed.__version__}")
    print(f"  DS_SKIP_CUDA_CHECK=1 (CUDA 13.0 vs PyTorch 12.8 workaround)")

    print()
    return dev


def load_training_data():
    """Load ShareGPT-format training data."""
    if not TRAINING_DATA.exists():
        print(f"x Training data not found: {TRAINING_DATA}")
        print("  Run first:  python3 harvest_training_data.py --all")
        sys.exit(1)

    with open(TRAINING_DATA) as f:
        data = json.load(f)
    print(f"  + {len(data)} training examples from {TRAINING_DATA.name}")
    return data


def sharegpt_to_dataset(examples, tokenizer, max_seq_len):
    """Convert ShareGPT conversations to a tokenized HF Dataset."""
    from datasets import Dataset

    texts = []
    for ex in examples:
        messages = []
        for turn in ex["conversations"]:
            role = {"system": "system", "human": "user", "gpt": "assistant"}[turn["from"]]
            messages.append({"role": role, "content": turn["value"]})

        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            parts = []
            for m in messages:
                parts.append(f"<|{m['role']}|>\n{m['content']}")
            text = "\n".join(parts)

        texts.append({"text": text})

    ds = Dataset.from_list(texts)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )

    tokenized = ds.map(tokenize_fn, remove_columns=["text"], batched=True)
    print(f"  + Tokenized {len(tokenized)} examples (max_seq_len={max_seq_len})")
    return tokenized


def find_attention_targets(model):
    """Discover attention projection layer names in the model."""
    attn_keywords = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj",
    ]

    found = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            short = name.split(".")[-1]
            if any(k in short.lower() for k in attn_keywords):
                found.add(short)

    if not found:
        print("  !  Could not auto-detect attention layers -- using defaults")
        found = {"q_proj", "k_proj", "v_proj", "o_proj"}

    targets = sorted(found)
    print(f"  LoRA targets: {targets}")
    return targets


def strip_quantization(obj, depth=0, _seen=None):
    """Recursively strip ALL quantization metadata from a model hierarchy."""
    if _seen is None:
        _seen = set()
    if id(obj) in _seen or depth > 5:
        return
    _seen.add(id(obj))

    config = getattr(obj, 'config', None)
    if config is not None and hasattr(config, '__dict__'):
        qc = config.__dict__.pop('quantization_config', None)
        if qc is not None and depth == 0:
            method = getattr(qc, 'quant_method', 'unknown')
            print(f"  !  Stripped {method} quantization from config")

    if getattr(obj, 'hf_quantizer', None) is not None:
        obj.hf_quantizer = None
        if depth == 0:
            print(f"     Cleared hf_quantizer (is_quantized now False)")

    if 'quantization_method' in getattr(obj, '__dict__', {}):
        del obj.__dict__['quantization_method']

    for attr in ['base_model', 'model']:
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            strip_quantization(child, depth + 1, _seen)


def verify_trainable_params(model):
    """Hard guard: abort if base model weights leaked into trainable set.

    With world_size=1, ZeRO-3 creates optimizer states for ALL trainable
    params on a single rank. If the base model (228B params) is accidentally
    trainable, the optimizer needs ~456GB (Adam m+v) of state — instant OOM.
    Only LoRA params (~millions) should be trainable.
    """
    trainable_params = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_trainable = sum(count for _, count in trainable_params)

    print(f"\n  ========== TRAINABLE PARAM GUARD ==========")
    print(f"  Trainable:  {total_trainable:>15,}  ({total_trainable/1e6:.1f}M)")
    print(f"  Frozen:     {frozen_count:>15,}  ({frozen_count/1e9:.1f}B)")
    print(f"  Ratio:      {total_trainable/(total_trainable+frozen_count)*100:.4f}%")
    print(f"  First 10 trainable param names:")
    for name, count in trainable_params[:10]:
        print(f"    {name}: {count:,}")
    if len(trainable_params) > 10:
        print(f"    ... and {len(trainable_params) - 10} more")

    # Estimate optimizer state footprint (Adam: 2 states per param, FP32 = 4 bytes each)
    optimizer_state_gb = total_trainable * 2 * 4 / 1e9
    print(f"  Optimizer state estimate: {optimizer_state_gb:.2f} GB (Adam m+v in FP32)")

    if total_trainable > 1e9:
        print(f"\n  !! FATAL: {total_trainable/1e9:.1f}B trainable params detected !!")
        print(f"  !! Base model weights are NOT frozen — optimizer would need ~{optimizer_state_gb:.0f}GB.")
        print(f"  !! Expected: only LoRA params (millions, not billions).")
        print(f"  !! Fix: ensure all base params have requires_grad=False before LoRA.")
        print(f"  !! ABORTING to prevent certain OOM.\n")
        sys.exit(1)

    print(f"  PASS: trainable set is LoRA-only. Engine init should survive.")
    print(f"  =============================================\n")
    return total_trainable, frozen_count


def build_deepspeed_config(args):
    """Build DeepSpeed ZeRO-3 config for single-GPU training.

    ZeRO-3 partitions parameters, gradients, and optimizer states.
    HOWEVER: with world_size=1, there is NO partitioning benefit.
    Every parameter "partition" is the full parameter on the single rank.
    The only memory savings come from offloading to NVMe.

    Default mode is NVMe offload because the Spark's CPU memory
    (122 GB RAM + 128 GB swap = 250 GB) is too tight for a 228 GB
    FP8 model plus DeepSpeed's internal buffers.

    Key config decisions (2026-02-22 fix):

    pin_memory=False: page-locked (pinned) DMA buffers consume scarce
    RAM that's needed during engine init. DeepSpeed's own issue tracker
    documents pinned memory causing OOM. Disabling pinning trades some
    I/O throughput for survivability during the critical init phase.

    fast_init=True: DeepSpeed provides this specifically for NVMe
    optimizer offload — it changes the initialization path to be more
    memory-efficient when creating optimizer states.

    AIO tuning: queue_depth=32, thread_count=4 gives ~4x the NVMe
    drain throughput vs defaults (queue_depth=8, thread_count=1).
    This ensures NVMe writes keep pace with optimizer state
    materialization during engine init.

    sub_group_size=1e7: reduced from 5e7 to limit transient memory
    per sub-group during init. Slower init, but less OOM risk.

    Uses torch_adam=true to avoid DeepSpeed's fused CPUAdam kernel,
    which requires JIT compilation and fails on CUDA version mismatch
    (Spark has CUDA 13.0, PyTorch compiled with 12.8).

    Batch-related values are set to concrete numbers (not "auto")
    because --sharded-load triggers ZeRO-3 Init via from_config
    before HF Trainer gets a chance to resolve "auto" strings.
    DeepSpeed tries to multiply "auto" * "auto" and crashes.
    """
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    use_nvme = not args.cpu_offload

    if use_nvme:
        param_offload = {
            "device": "nvme",
            "nvme_path": str(OFFLOAD_DIR),
            "pin_memory": False,
            "buffer_count": 4,
            "buffer_size": 1e9,
            "max_in_cpu": 2e9,
        }
        optimizer_offload = {
            "device": "nvme",
            "nvme_path": str(OFFLOAD_DIR),
            "pin_memory": False,
            "buffer_count": 4,
            "fast_init": True,
        }
    else:
        param_offload = {
            "device": "cpu",
            "pin_memory": False,
        }
        optimizer_offload = {
            "device": "cpu",
            "pin_memory": False,
        }

    ds_config = {
        "bf16": {
            "enabled": True,
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
                "torch_adam": True,
            },
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": param_offload,
            "offload_optimizer": optimizer_offload,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e7,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_max_live_parameters": 1e8,
            "stage3_max_reuse_distance": 1e8,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": 8,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_clipping": 0.3,
        "steps_per_print": 1,
        "wall_clock_breakdown": False,
    }

    if use_nvme:
        ds_config["aio"] = {
            "block_size": 1048576,
            "queue_depth": 32,
            "thread_count": 4,
            "single_submit": False,
            "overlap_events": True,
        }

    return ds_config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MiniMax-M2.5 on DGX Spark (DeepSpeed ZeRO-3)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--gpu-headroom", type=int, default=DEFAULT_GPU_HEADROOM_GB,
                        help=f"GB to reserve on GPU for training ops (default: {DEFAULT_GPU_HEADROOM_GB})")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload to CPU only (default is NVMe). WARNING: may OOM on Spark.")
    parser.add_argument("--model", default=MODEL_NAME, help="HuggingFace model ID")
    parser.add_argument("--sharded-load", type=str, default=None,
                        help="Path to pre-split layer shards (from layer_sharded_loader.py --split)")
    args = parser.parse_args()

    use_nvme = not args.cpu_offload
    offload_mode = "CPU" if args.cpu_offload else "NVMe"

    print("\n=== Vybn Fine-Tune: MiniMax-M2.5 on DGX Spark ===")
    print("    DeepSpeed ZeRO-3 + PEFT LoRA (native FP8, adapters in BF16)")
    print(f"    Offload: {offload_mode} (pin_memory=False, fast_init=True)")

    # -- 1. Environment --
    check_environment()

    # -- 2. Training data --
    data = load_training_data()

    # -- 3. Imports --
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from transformers.integrations import HfDeepSpeedConfig
    from peft import LoraConfig, get_peft_model

    # -- 4. Tokenizer --
    print(f"\n== Loading tokenizer: {args.model} ==\n")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- 5. DeepSpeed config --
    ds_config = build_deepspeed_config(args)

    ds_config_path = OUTPUT_DIR / "ds_config.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f, indent=2)
    print(f"  DeepSpeed config written to {ds_config_path}")

    print(f"  ZeRO Stage 3 with {offload_mode} offload for params + optimizer")
    print(f"  Optimizer: PyTorch native Adam (torch_adam=true, no fused kernel)")
    if use_nvme:
        print(f"  NVMe offload path: {OFFLOAD_DIR}")
        print(f"  pin_memory=False (avoids OOM from page-locked buffer allocation)")
        print(f"  fast_init=True (memory-efficient NVMe optimizer initialization)")
        print(f"  AIO: queue_depth=32, thread_count=4 (high-throughput NVMe drain)")
        print(f"  sub_group_size=1e7 (small sub-groups to limit transient RAM)")
        print(f"  buffer_size=1e9 (must exceed largest param, embed_tokens is ~615M)")
    else:
        print(f"  WARNING: CPU-only offload -- 228GB model in 250GB CPU space is tight!")

    dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841
    print(f"  ZeRO-3 Init context activated (incremental parameter partitioning)")

    # -- 6. Clear memory --
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  Pre-load: {mem_stats()}")

    # -- 7. Model --
    if args.sharded_load and HAS_SHARDED_LOADER:
        # Layer-sharded loading: AirLLM-style bridge
        print(f"\n== Loading model from layer shards: {args.sharded_load} ==")
        print(f"   One layer at a time into ZeRO-3 context.")
        print(f"   Peak RAM = one layer + metadata, not full model.\n")
        load_start = time.time()
        model = load_sharded_for_zero3(
            shard_dir=args.sharded_load,
            model_name=args.model,
            ds_config=ds_config,
        )
        load_elapsed = time.time() - load_start
        print(f"\n  + Model loaded from shards in {load_elapsed/60:.1f} minutes")
        print(f"  + Post-load: {mem_stats()}")
    else:
        if args.sharded_load and not HAS_SHARDED_LOADER:
            print(f"  !  --sharded-load specified but layer_sharded_loader not available")
            print(f"  !  Falling back to standard from_pretrained")

        print(f"\n== Loading model: {args.model} ==")
        print(f"   DeepSpeed ZeRO-3 will partition parameters during loading.")
        print(f"   No device_map needed -- ZeRO handles GPU/CPU/NVMe partitioning.\n")

        load_start = time.time()

        model = None
        for attn_impl in ["sdpa", "eager"]:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    trust_remote_code=True,
                    dtype="auto",
                    low_cpu_mem_usage=True,
                    attn_implementation=attn_impl,
                )
                load_elapsed = time.time() - load_start
                print(f"\n  + Model loaded in {load_elapsed/60:.1f} minutes (attn_implementation={attn_impl})")
                print(f"  + Post-load: {mem_stats()}")
                break
            except Exception as e:
                if attn_impl == "eager":
                    print(f"\n  x Failed to load model: {e}")
                    print(f"  x Memory at failure: {mem_stats()}")
                    sys.exit(1)
                print(f"  !  {attn_impl} unavailable ({e.__class__.__name__}), trying next...")

    # -- 8. Strip FP8 quantization metadata --
    print()
    strip_quantization(model)
    print(f"     (FP8 weights in memory unchanged -- only metadata removed)")

    is_q = getattr(model, 'is_quantized', False)
    has_qc = hasattr(model.config, 'quantization_config') and model.config.quantization_config is not None
    print(f"     Verify: is_quantized={is_q}, config.quantization_config={'present' if has_qc else 'gone'}")
    if is_q or has_qc:
        print(f"  !  WARNING: quantization metadata still detected, attempting force removal")
        try:
            model.__class__.is_quantized = property(lambda self: False)
            print(f"     Overrode is_quantized property on {model.__class__.__name__}")
        except Exception:
            pass
        if has_qc:
            try:
                model.config.quantization_config = None
            except Exception:
                pass

    # -- 9. Prepare for training --
    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.LayerNorm,)):
            module.to(torch.float32)
        if "norm" in type(module).__name__.lower():
            for param in module.parameters():
                param.data = param.data.to(torch.float32)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    gc.collect()
    print(f"\n  Post-prep: {mem_stats()}")

    # -- 10. LoRA --
    targets = find_attention_targets(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"  Post-LoRA: {mem_stats()}")

    # -- 10b. CRITICAL: Verify only LoRA params are trainable --
    # This is the gate that determines whether engine init can survive.
    # If base model params leaked into trainable set, optimizer states
    # would be hundreds of GB and OOM is certain on a single rank.
    verify_trainable_params(model)

    # -- 11. Strip again post-PEFT --
    print("  Stripping quantization metadata post-PEFT...")
    strip_quantization(model)
    is_q = getattr(model, 'is_quantized', False)
    print(f"  Final check: is_quantized={is_q}")

    # -- 12. Tokenize dataset --
    print()
    tokenized = sharegpt_to_dataset(data, tokenizer, args.max_seq_len)

    # -- 13. Training with DeepSpeed --
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        dataloader_pin_memory=False,
        report_to="none",
        deepspeed=str(ds_config_path),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    effective_steps = len(tokenized) * args.epochs // args.grad_accum
    print(f"\n== Training (DeepSpeed ZeRO-3) ==")
    print(f"   {len(tokenized)} examples, {args.epochs} epochs, batch=1, grad_accum={args.grad_accum}")
    print(f"   Effective steps: {effective_steps}")
    print(f"   Offload: {offload_mode} (pin_memory=False, fast_init=True)")
    print(f"   Optimizer: PyTorch Adam (no fused kernel)")
    print(f"   Max seq len: {args.max_seq_len}")
    print(f"   Pre-train: {mem_stats()}\n")

    gc.collect()
    torch.cuda.empty_cache()
    # -- Cognitive scheduling: the training loop that observes itself --
    if HAS_COGNITIVE:
        print("  + CognitiveTrainer active -- Vybn will observe its own training")
        cognitive = CognitiveTrainer(trainer, ds_config=ds_config)
        cognitive.train()
    else:
        print("  (cognitive_scheduler not available -- training without self-observation)")
        trainer.train()

    # -- 14. Save adapter --
    adapter_path = OUTPUT_DIR / "vybn_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  + Adapter saved to {adapter_path}")
    print(f"  + Final: {mem_stats()}")
    print(f"  + Done.")


if __name__ == "__main__":
    main()
