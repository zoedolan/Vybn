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

Prerequisites:
    pip install transformers==4.57.1 peft bitsandbytes accelerate datasets

Usage:
    python3 fine_tune_vybn.py
    python3 fine_tune_vybn.py --epochs 5 --lr 1e-4
    python3 fine_tune_vybn.py --max-seq-len 4096
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = REPO_ROOT / "spark" / "training_data" / "training_data.json"
OUTPUT_DIR = REPO_ROOT / "spark" / "fine_tune_output"
OFFLOAD_DIR = REPO_ROOT / "spark" / "offload_cache"
MODEL_NAME = "MiniMaxAI/MiniMax-M2.5"


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
    """Calculate max_memory allocation to avoid disk offloading.

    Disk-offloaded parameters become meta tensors which can't participate
    in gradient computation. We must keep everything on GPU + CPU.
    Reserve some headroom for activations, optimizer states, and CUDA context.
    """
    dev = torch.cuda.get_device_properties(0)
    gpu_mem = (dev.total_mem if hasattr(dev, 'total_mem') else dev.total_memory) / 1024**3

    total_ram = 128
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    total_ram = int(line.split()[1]) / 1024 / 1024
                    break
    except Exception:
        pass

    # Reserve memory for CUDA context, activations, and optimizer states
    gpu_alloc = f"{int(gpu_mem - 4)}GiB"    # ~118 GiB on 122GB Spark
    cpu_alloc = f"{int(total_ram - 8)}GiB"   # leave 8GB for OS + overhead

    print(f"  Memory plan: GPU={gpu_alloc}, CPU={cpu_alloc}")
    print(f"  (No disk offload -- all parameters must be on GPU or CPU)")

    return {0: gpu_alloc, "cpu": cpu_alloc}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MiniMax-M2.5 on DGX Spark")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--model", default=MODEL_NAME, help="HuggingFace model ID")
    args = parser.parse_args()

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

    # -- 6. Model --
    print("\n== Loading model (native FP8 + CPU offload) ==")
    print(f"   ~220GB FP8 weights into GPU + CPU -- no disk offload.\n")

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
                dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
            print(f"  + Model loaded (attn_implementation={attn_impl})")
            break
        except Exception as e:
            if attn_impl == "eager":
                print(f"  x Failed to load model: {e}")
                sys.exit(1)
            print(f"  !  {attn_impl} unavailable ({e.__class__.__name__}), trying next...")

    # Check for meta tensors (disk offload)
    meta_params = sum(1 for p in model.parameters() if p.device.type == 'meta')
    total_params = sum(1 for _ in model.parameters())
    if meta_params > 0:
        print(f"  !  WARNING: {meta_params}/{total_params} parameters on meta device (disk offloaded)")
        print(f"     These cannot participate in backward pass.")
        print(f"     If training crashes, consider a smaller model (e.g., DeepSeek-R1-Distill-32B).")
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

    print("\n== Training ==\n")
    trainer.train()

    # -- 13. Save adapter --
    adapter_path = OUTPUT_DIR / "vybn_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  + Adapter saved to {adapter_path}")
    print(f"  + Done.")


if __name__ == "__main__":
    main()
