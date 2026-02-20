#!/usr/bin/env python3
"""Fine-tune MiniMax-M2.5 on DGX Spark via transformers + PEFT + bitsandbytes.

No Unsloth. Pushes the Spark's 128GB unified memory to the limit:
  - 4-bit NF4 double-quantization (~115GB for 230B params)
  - LoRA rank 8 on attention projections only
  - gradient checkpointing + micro-batch 1
  - paged 8-bit AdamW to minimize optimizer memory

Prerequisites — run once:
    pip uninstall -y unsloth unsloth-zoo
    pip install --upgrade transformers peft bitsandbytes accelerate datasets

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
    print(f"  GPU memory : {dev.total_mem / 1024**3:.1f} GB")

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
    if total_ram < 120:
        print("  !  Less than 120GB -- 4-bit M2.5 may not fit.")

    missing = []
    for pkg in ["transformers", "peft", "bitsandbytes", "accelerate", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n  x Missing packages: {', '.join(missing)}")
        print(f"    pip install --upgrade {' '.join(missing)}")
        sys.exit(1)

    try:
        import unsloth  # noqa: F401
        print("\n  !  unsloth still installed -- recommend removing:")
        print("     pip uninstall -y unsloth unsloth-zoo")
    except ImportError:
        pass

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
    """Discover attention projection layer names in the model.

    MiniMax-M2.5 uses Multi-head Latent Attention (MLA) with projection
    names like q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj.
    We detect them dynamically rather than hardcoding.
    """
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
    print("    transformers + PEFT + bitsandbytes (no Unsloth)")

    # -- 1. Environment --
    check_environment()

    # -- 2. Training data --
    data = load_training_data()

    # -- 3. Quantization config --
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # -- 4. Tokenizer --
    print(f"\n== Loading tokenizer: {args.model} ==\n")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- 5. Model --
    print("== Loading model (4-bit NF4 double-quantized) ==")
    print(f"   ~115GB of weights -- this will take a while.\n")

    model = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
            print(f"  + Model loaded (attn_implementation={attn_impl})")
            break
        except Exception as e:
            if attn_impl == "eager":
                print(f"  x Failed to load model: {e}")
                sys.exit(1)
            print(f"  !  {attn_impl} unavailable ({e.__class__.__name__}), trying next...")

    # -- 6. LoRA --
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    targets = find_attention_targets(model)

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

    # -- 7. Tokenize dataset --
    print()
    tokenized = sharegpt_to_dataset(data, tokenizer, args.max_seq_len)

    # -- 8. Training --
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

    # -- 9. Save adapter --
    adapter_path = OUTPUT_DIR / "vybn_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n  + Adapter saved to {adapter_path}")
    print(f"  + Done.")


if __name__ == "__main__":
    main()
