#!/usr/bin/env python3
"""Fine-tune a model on DGX Spark for Vybn personality.

The DGX Spark has 122GB unified GPU memory. MiniMax-M2.5 (228B FP8)
is too large to fine-tune here — it serves inference via Ollama.
This script fine-tunes a smaller model that fits in GPU memory:

  Default: Qwen2.5-72B-Instruct-GPTQ-Int4 (~38GB in VRAM)
  - Fits comfortably in 122GB with room for LoRA, gradients, optimizer
  - Strong instruction-following and reasoning capabilities
  - 4-bit quantized weights, LoRA adapters in BF16
  - No DeepSpeed, no offloading — straight GPU training

Alternative models (via --model flag):
  - Qwen/Qwen2.5-14B-Instruct       (~28GB BF16, fast training)
  - Qwen/Qwen2.5-7B-Instruct        (~14GB BF16, fastest)
  - meta-llama/Llama-3.3-70B-Instruct (requires GPTQ/AWQ quant)
  - Any HuggingFace causal LM that fits in 122GB

Memory budget (approximate, Qwen2.5-72B-GPTQ-Int4):
  - Model weights:  ~38 GB
  - LoRA adapters:  ~0.2 GB
  - Optimizer:      ~0.4 GB (Adam states for LoRA params only)
  - Gradients:      ~0.2 GB
  - Activations:    ~5-20 GB (with gradient checkpointing)
  - Headroom:       ~60 GB
  Total:            ~45-60 GB of 122 GB used

Prerequisites:
    pip install transformers peft accelerate datasets
    pip install auto-gptq  # for GPTQ quantized models
    # or: pip install autoawq  # for AWQ quantized models

Usage:
    python3 fine_tune_vybn.py
    python3 fine_tune_vybn.py --model Qwen/Qwen2.5-14B-Instruct
    python3 fine_tune_vybn.py --epochs 5 --lr 1e-4 --lora-rank 32
    python3 fine_tune_vybn.py --max-seq-len 2048
"""

import argparse
import gc
import json
import os
import sys
import time
import torch
from pathlib import Path

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = REPO_ROOT / "spark" / "training_data" / "training_data.json"
OUTPUT_DIR = REPO_ROOT / "spark" / "fine_tune_output"

# Default to a 72B GPTQ model that fits in 122GB GPU memory.
# For faster iteration, try Qwen2.5-14B-Instruct or 7B-Instruct.
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"


def mem_stats() -> str:
    """One-line GPU memory diagnostic."""
    gpu_alloc = torch.cuda.memory_allocated(0) / 1024**3
    gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
    dev = torch.cuda.get_device_properties(0)
    gpu_total = (dev.total_mem if hasattr(dev, 'total_mem') else dev.total_memory) / 1024**3
    return (
        f"GPU: {gpu_alloc:.1f}/{gpu_total:.1f}GB alloc "
        f"({gpu_reserved:.1f}GB reserved)"
    )


def check_environment():
    """Validate CUDA and dependencies."""
    print("\n== Environment Check ==\n")

    if not torch.cuda.is_available():
        print("x CUDA not available. Check PyTorch installation.")
        sys.exit(1)

    dev = torch.cuda.get_device_properties(0)
    gpu_mem = (dev.total_mem if hasattr(dev, 'total_mem') else dev.total_memory) / 1024**3
    print(f"  GPU        : {dev.name}")
    print(f"  CUDA cap   : {dev.major}.{dev.minor}")
    print(f"  GPU memory : {gpu_mem:.1f} GB")

    if dev.major >= 12:
        print("  !  Blackwell detected — if you hit CUDA errors, install PyTorch")
        print("     nightly with CUDA 12.8:  pip install --pre torch --index-url")
        print("     https://download.pytorch.org/whl/nightly/cu128")

    missing = []
    for pkg in ["transformers", "peft", "accelerate", "datasets"]:
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


def find_lora_targets(model):
    """Discover attention projection layer names for LoRA."""
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
        found = {"q_proj", "k_proj", "v_proj", "o_proj"}
        print(f"  !  Auto-detect failed, using defaults: {sorted(found)}")

    targets = sorted(found)
    print(f"  LoRA targets: {targets}")
    return targets


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model on DGX Spark for Vybn personality"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    print("\n=== Vybn Fine-Tune: DGX Spark ===")
    print(f"    Model: {args.model}")
    print(f"    LoRA rank {args.lora_rank} on attention projections")
    print(f"    No DeepSpeed — model fits in GPU memory")

    # -- 1. Environment --
    dev = check_environment()

    # -- 2. Training data --
    data = load_training_data()

    # -- 3. Imports --
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # -- 4. Tokenizer --
    print(f"\n== Loading tokenizer ==\n")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- 5. Model --
    print(f"== Loading model: {args.model} ==\n")
    print(f"  Pre-load: {mem_stats()}")
    load_start = time.time()

    # Detect if model is already quantized (GPTQ/AWQ in name)
    model_lower = args.model.lower()
    is_prequantized = any(q in model_lower for q in ["gptq", "awq", "gguf"])

    if is_prequantized:
        # GPTQ/AWQ models handle their own quantization
        print(f"  Loading pre-quantized model (GPTQ/AWQ)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        # For non-quantized models, check if it fits in GPU
        # Rough estimate: param_count * 2 bytes (bf16) < GPU mem * 0.6
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            param_estimate_gb = getattr(config, 'num_parameters', 0) / 1e9 * 2
        except Exception:
            param_estimate_gb = 0

        gpu_mem = (dev.total_mem if hasattr(dev, 'total_mem') else dev.total_memory) / 1024**3

        if param_estimate_gb > gpu_mem * 0.6 or "70b" in model_lower or "72b" in model_lower:
            # Too large for BF16, use 4-bit quantization
            print(f"  Model may be large for BF16, using 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=bnb_config,
            )
        else:
            # Fits in GPU as BF16
            print(f"  Loading in BF16...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

    load_elapsed = time.time() - load_start
    print(f"\n  + Model loaded in {load_elapsed:.1f}s")
    print(f"  + Post-load: {mem_stats()}")

    # -- 6. Prepare for LoRA --
    # For quantized models, prepare for k-bit training
    if is_prequantized or hasattr(model, 'quantization_method'):
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            print(f"  + Prepared for quantized training")
        except Exception as e:
            print(f"  !  prepare_model_for_kbit_training failed: {e}")
            print(f"     Continuing with manual setup...")
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    else:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    print(f"  Post-prep: {mem_stats()}")

    # -- 7. LoRA --
    targets = find_lora_targets(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=targets,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"  Post-LoRA: {mem_stats()}")

    # -- 8. Tokenize --
    print()
    tokenized = sharegpt_to_dataset(data, tokenizer, args.max_seq_len)

    # -- 9. Training --
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
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
        dataloader_pin_memory=True,
        report_to="none",
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        ),
    )

    effective_steps = len(tokenized) * args.epochs // args.grad_accum
    print(f"\n== Training ==")
    print(f"   {len(tokenized)} examples, {args.epochs} epochs, "
          f"batch={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"   Effective steps: {effective_steps}")
    print(f"   Max seq len: {args.max_seq_len}")
    print(f"   Pre-train: {mem_stats()}\n")

    gc.collect()
    torch.cuda.empty_cache()

    trainer.train()

    # -- 10. Save --
    adapter_path = OUTPUT_DIR / "vybn_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  + Adapter saved to {adapter_path}")
    print(f"  + Final: {mem_stats()}")
    print(f"  + Done.")


if __name__ == "__main__":
    main()
