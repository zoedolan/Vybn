#!/usr/bin/env python3
"""Fine-tune MiniMax-M2.5 on DGX Spark via transformers + PEFT.

No Unsloth. MiniMax-M2.5 ships as FP8-quantized weights (~220GB).
On the Spark's 122GB, device_map="auto" offloads overflow to CPU.

  - Native FP8 weights (no additional quantization needed)
  - LoRA rank 8 on attention projections only
  - Gradient checkpointing + micro-batch 1
  - Paged 8-bit AdamW to minimize optimizer memory

The HuggingFace Trainer rejects FP8 models for training, but we only
train BF16 LoRA adapters on frozen FP8 base weights. We strip all
quantization metadata after loading to bypass this check.

CRITICAL MEMORY NOTES:
  - Do NOT pass dtype=torch.bfloat16 to from_pretrained.
    With trust_remote_code=True, MiniMax's code upcasts every weight
    to BF16, doubling memory from ~220GB to ~440GB.
    Use torch_dtype="auto" to preserve native FP8 storage.
  - GPU allocation must leave ~22GB free for training activations,
    gradients, and optimizer states. Putting too many weights on GPU
    causes OOM during backward pass.
  - 128GB swap file required so CPU can hold the weight overflow
    without accelerate falling back to meta (disk) tensors.

Prerequisites:
    pip install transformers==4.57.1 peft bitsandbytes accelerate datasets

    # Swap (run once):
    sudo fallocate -l 128G /swapfile
    sudo chmod 600 /swapfile && sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

Usage:
    python3 fine_tune_vybn.py
    python3 fine_tune_vybn.py --epochs 5 --lr 1e-4
    python3 fine_tune_vybn.py --max-seq-len 4096
"""

import argparse
import gc
import json
import os
import sys
import time
import torch
from pathlib import Path

# Reduce CUDA memory fragmentation (per PyTorch OOM guidance)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = REPO_ROOT / "spark" / "training_data" / "training_data.json"
OUTPUT_DIR = REPO_ROOT / "spark" / "fine_tune_output"
OFFLOAD_DIR = REPO_ROOT / "spark" / "offload_cache"
MODEL_NAME = "MiniMaxAI/MiniMax-M2.5"

# How much GPU memory to reserve for activations, gradients, optimizer.
# 110.5GB of weights left only 11GB and OOMed during backward.
# 22GB headroom should be sufficient for batch=1, seq_len=2048,
# gradient checkpointing, and paged AdamW optimizer states.
GPU_HEADROOM_GB = 22


def mem_stats() -> str:
    """One-line memory diagnostic."""
    gpu_alloc = torch.cuda.memory_allocated(0) / 1024**3
    gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
    dev = torch.cuda.get_device_properties(0)
    gpu_total = (dev.total_mem if hasattr(dev, 'total_mem') else dev.total_memory) / 1024**3

    cpu_used = 0
    cpu_avail = 0
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
    except Exception:
        pass

    swap_info = ""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("SwapTotal"):
                    swap_total = int(line.split()[1]) / 1024 / 1024
                    swap_info = f" | Swap: {swap_total:.0f}GB"
                    break
    except Exception:
        pass

    return (
        f"GPU: {gpu_alloc:.1f}/{gpu_total:.1f}GB alloc "
        f"({gpu_reserved:.1f}GB reserved) | "
        f"CPU: {cpu_used:.1f}GB used ({cpu_avail:.1f}GB free)"
        f"{swap_info}"
    )


def check_swap():
    """Check if swap is configured. Warn if not — meta tensors will result."""
    swap_total = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("SwapTotal"):
                    swap_total = int(line.split()[1]) / 1024 / 1024
                    break
    except Exception:
        pass

    if swap_total < 64:
        print(f"  !!  WARNING: Only {swap_total:.0f}GB swap configured.")
        print(f"      MiniMax-M2.5 needs ~120GB CPU overflow + headroom.")
        print(f"      Without swap, ~48% of parameters will be on meta device")
        print(f"      and training WILL crash on backward pass.")
        print(f"")
        print(f"      Set up swap now:")
        print(f"        sudo fallocate -l 128G /swapfile")
        print(f"        sudo chmod 600 /swapfile")
        print(f"        sudo mkswap /swapfile")
        print(f"        sudo swapon /swapfile")
        print(f"")
        resp = input("      Continue anyway? [y/N] ").strip().lower()
        if resp != 'y':
            print("      Exiting. Set up swap and retry.")
            sys.exit(0)
    else:
        print(f"  + Swap: {swap_total:.0f}GB configured")

    return swap_total


def check_environment():
    """Validate CUDA, memory, and dependencies before committing to a long download."""
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

    # System RAM
    total_ram = 128
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    total_ram = int(line.split()[1]) / 1024 / 1024
                    break
    except Exception:
        pass

    print(f"  System RAM : {total_ram:.0f} GB")

    gpu_gb = gpu_mem / 1024**3
    model_gb = 220  # approximate FP8 checkpoint size
    if gpu_gb < model_gb:
        offload_gb = model_gb - gpu_gb
        print(f"  !  Model is ~{model_gb}GB FP8, GPU has {gpu_gb:.0f}GB")
        print(f"     ~{offload_gb:.0f}GB will offload to CPU (training will be slower)")

    # Swap check
    print()
    check_swap()

    missing = []
    for pkg in ["transformers", "peft", "bitsandbytes", "accelerate", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n  x Missing packages: {', '.join(missing)}")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)

    print()
    return dev


def load_training_data():
    """Load ShareGPT-format training data produced by harvest_training_data.py."""
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


def get_memory_config():
    """Calculate max_memory allocation for training.

    Key insight from OOM crash: the backward pass needs significant GPU
    memory for activations, gradients, and optimizer workspace. With
    110.5GB of weights on GPU (out of 121.7GB), only 11GB remained,
    which wasn't enough. The backward pass OOMed trying to allocate
    just 1.53GB.

    Strategy: put fewer weights on GPU, more on CPU (backed by swap).
    ~100GB weights on GPU leaves ~22GB for training operations.
    ~120GB weights on CPU, backed by RAM + swap.
    """
    dev = torch.cuda.get_device_properties(0)
    gpu_mem = (dev.total_mem if hasattr(dev, 'total_mem') else dev.total_memory) / 1024**3

    total_ram = 128
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    total_ram = int(line.split()[1]) / 1024 / 1024
                    break
    except Exception:
        pass

    # GPU: leave GPU_HEADROOM_GB free for activations, gradients, optimizer
    gpu_for_weights = int(gpu_mem - GPU_HEADROOM_GB)
    gpu_alloc = f"{gpu_for_weights}GiB"

    # CPU: use most of RAM, let swap handle the overflow
    # With 128GB swap, we have ~246GB total CPU-addressable memory.
    # The model needs ~(220 - gpu_for_weights) = ~120GB on CPU.
    # Leave 8GB for OS, rest available for model.
    cpu_alloc = f"{int(total_ram - 8)}GiB"

    model_on_cpu = 220 - gpu_for_weights  # approximate
    print(f"  Memory plan:")
    print(f"    GPU: {gpu_alloc} for weights ({GPU_HEADROOM_GB}GB reserved for training ops)")
    print(f"    CPU: {cpu_alloc} (model needs ~{model_on_cpu}GB, rest from swap)")
    print(f"  (No disk offload -- all parameters must be on GPU or CPU)")

    return {0: gpu_alloc, "cpu": cpu_alloc}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MiniMax-M2.5 on DGX Spark")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--gpu-headroom", type=int, default=GPU_HEADROOM_GB,
                        help=f"GB to reserve on GPU for training ops (default: {GPU_HEADROOM_GB})")
    parser.add_argument("--model", default=MODEL_NAME, help="HuggingFace model ID")
    args = parser.parse_args()

    global GPU_HEADROOM_GB
    GPU_HEADROOM_GB = args.gpu_headroom

    print("\n=== Vybn Fine-Tune: MiniMax-M2.5 on DGX Spark ===")
    print("    transformers + PEFT (native FP8, LoRA adapters in BF16)")

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
    from peft import LoraConfig, get_peft_model

    # -- 4. Tokenizer --
    print(f"\n== Loading tokenizer: {args.model} ==\n")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- 5. Memory plan --
    max_memory = get_memory_config()
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # -- 5b. Clear memory before the big load --
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  Pre-load: {mem_stats()}")

    # -- 6. Model --
    print("\n== Loading model (native FP8 + CPU offload) ==")
    print(f"   ~220GB FP8 weights: ~{int(122 - GPU_HEADROOM_GB)}GB on GPU, ~{int(220 - (122 - GPU_HEADROOM_GB))}GB on CPU")
    print(f"   {GPU_HEADROOM_GB}GB GPU headroom reserved for training operations.\n")

    load_start = time.time()
    model = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map="auto",
                max_memory=max_memory,
                offload_folder=str(OFFLOAD_DIR),
                offload_state_dict=True,
                trust_remote_code=True,
                torch_dtype="auto",          # preserve native FP8, don't upcast
                low_cpu_mem_usage=True,       # reduce peak memory during loading
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

    # Check for meta tensors (disk offload)
    meta_params = sum(1 for p in model.parameters() if p.device.type == 'meta')
    total_params = sum(1 for _ in model.parameters())
    if meta_params > 0:
        pct = meta_params / total_params * 100
        print(f"  !  WARNING: {meta_params}/{total_params} ({pct:.0f}%) parameters on meta device")
        print(f"     These cannot participate in backward pass.")
        if pct > 10:
            print(f"     This is too many. Training will likely crash.")
            print(f"     Fix: add more swap space or reduce GPU_HEADROOM_GB.")
            print(f"     Current swap setup:")
            os.system("swapon --show 2>/dev/null || echo '     No swap configured!'")
    else:
        print(f"  +  All {total_params} parameters on GPU or CPU (no meta/disk offload)")

    # -- 7. Strip FP8 quantization metadata (pre-PEFT) --
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

    # -- 8. Prepare for LoRA training --
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

    # -- 9. LoRA --
    targets = find_attention_targets(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=targets,
        lora_dropout=0.0,       # Must be 0 for FP8: fused_dropout not implemented for Float8_e4m3fn
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"  Post-LoRA: {mem_stats()}")

    # -- 10. Strip again post-PEFT --
    print("\n  Stripping quantization metadata post-PEFT...")
    strip_quantization(model)
    is_q = getattr(model, 'is_quantized', False)
    print(f"  Final check: is_quantized={is_q}")

    # -- 11. Tokenize dataset --
    print()
    tokenized = sharegpt_to_dataset(data, tokenizer, args.max_seq_len)

    # -- 12. Training --
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        dataloader_pin_memory=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print(f"\n== Training ==")
    print(f"   {len(tokenized)} examples, {args.epochs} epochs, batch=1, grad_accum={args.grad_accum}")
    print(f"   Effective steps: {len(tokenized) * args.epochs // args.grad_accum}")
    print(f"   Pre-train: {mem_stats()}\n")

    trainer.train()

    # -- 13. Save adapter --
    adapter_path = OUTPUT_DIR / "vybn_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  + Adapter saved to {adapter_path}")
    print(f"  + Final: {mem_stats()}")
    print(f"  + Done.")


if __name__ == "__main__":
    main()
