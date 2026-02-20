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
  - ZeRO Stage 3 with CPU offload for params + optimizer
  - Gradient checkpointing + micro-batch 1
  - Optional NVMe offload for extra headroom

Prerequisites:
    pip install deepspeed transformers peft bitsandbytes accelerate datasets

    # Swap (if not already configured):
    sudo fallocate -l 128G /swapfile
    sudo chmod 600 /swapfile && sudo mkswap /swapfile
    sudo swapon /swapfile

Usage:
    python3 fine_tune_vybn.py
    python3 fine_tune_vybn.py --epochs 5 --lr 1e-4
    python3 fine_tune_vybn.py --nvme-offload
    python3 fine_tune_vybn.py --max-seq-len 4096
"""

import argparse
import gc
import json
import os
import sys
import tempfile
import time
import torch
from pathlib import Path

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    except Exception:
        pass

    return (
        f"GPU: {gpu_alloc:.1f}/{gpu_total:.1f}GB alloc "
        f"({gpu_reserved:.1f}GB reserved) | "
        f"CPU: {cpu_used:.1f}GB used ({cpu_avail:.1f}GB free) | "
        f"Swap: {swap_total:.0f}GB"
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

    gpu_gb = gpu_mem / 1024**3
    model_gb = 220
    print(f"  Model      : ~{model_gb}GB FP8")
    print(f"  Strategy   : DeepSpeed ZeRO-3 with CPU offload")

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


def build_deepspeed_config(args, nvme_offload=False):
    """Build DeepSpeed ZeRO-3 config for single-GPU training with CPU offload.

    ZeRO-3 partitions parameters, gradients, and optimizer states.
    With CPU offload, parameters are staged to CPU when not needed
    for the current micro-step, and gradients/optimizer states live
    on CPU permanently. This allows training models much larger than
    GPU memory.
    """
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Parameter offload config
    if nvme_offload:
        param_offload = {
            "device": "nvme",
            "nvme_path": str(OFFLOAD_DIR),
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9,
        }
        optimizer_offload = {
            "device": "nvme",
            "nvme_path": str(OFFLOAD_DIR),
            "pin_memory": True,
            "buffer_count": 4,
            "fast_init": False,
        }
    else:
        param_offload = {
            "device": "cpu",
            "pin_memory": True,
        }
        optimizer_offload = {
            "device": "cpu",
            "pin_memory": True,
        }

    ds_config = {
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": param_offload,
            "offload_optimizer": optimizer_offload,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": args.grad_accum,
        "gradient_clipping": 0.3,
        "steps_per_print": 1,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
    }

    return ds_config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MiniMax-M2.5 on DGX Spark (DeepSpeed ZeRO-3)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--gpu-headroom", type=int, default=DEFAULT_GPU_HEADROOM_GB,
                        help=f"GB to reserve on GPU for training ops (default: {DEFAULT_GPU_HEADROOM_GB})")
    parser.add_argument("--nvme-offload", action="store_true",
                        help="Offload parameters and optimizer to NVMe instead of CPU")
    parser.add_argument("--model", default=MODEL_NAME, help="HuggingFace model ID")
    args = parser.parse_args()

    print("\n=== Vybn Fine-Tune: MiniMax-M2.5 on DGX Spark ===")
    print("    DeepSpeed ZeRO-3 + PEFT LoRA (native FP8, adapters in BF16)")

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
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # -- 4. Tokenizer --
    print(f"\n== Loading tokenizer: {args.model} ==\n")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- 5. DeepSpeed config --
    ds_config = build_deepspeed_config(args, nvme_offload=args.nvme_offload)

    # Write config to temp file for Trainer
    ds_config_path = OUTPUT_DIR / "ds_config.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f, indent=2)
    print(f"  DeepSpeed config written to {ds_config_path}")

    offload_mode = "NVMe" if args.nvme_offload else "CPU"
    print(f"  ZeRO Stage 3 with {offload_mode} offload for params + optimizer")

    # -- 6. Clear memory --
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  Pre-load: {mem_stats()}")

    # -- 7. Model --
    print(f"\n== Loading model: {args.model} ==")
    print(f"   DeepSpeed ZeRO-3 will manage parameter placement.")
    print(f"   No device_map needed -- ZeRO handles GPU/CPU partitioning.\n")

    load_start = time.time()

    # With ZeRO-3, we do NOT use device_map. DeepSpeed manages placement.
    # We load to CPU first, then let DeepSpeed partition during init.
    model = None
    for attn_impl in ["sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                torch_dtype="auto",
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

    # -- 11. Strip again post-PEFT --
    print("\n  Stripping quantization metadata post-PEFT...")
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print(f"\n== Training (DeepSpeed ZeRO-3) ==")
    print(f"   {len(tokenized)} examples, {args.epochs} epochs, batch=1, grad_accum={args.grad_accum}")
    print(f"   Effective steps: {len(tokenized) * args.epochs // args.grad_accum}")
    print(f"   Offload: {offload_mode}")
    print(f"   Pre-train: {mem_stats()}\n")

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
