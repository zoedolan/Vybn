#!/usr/bin/env python3
"""Auto-generated training script for Vybn growth cycle."""
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# ── Config ──────────────────────────────────────────────────────────────
MODEL_ID = "Nemotron-Super-512B-v1"
DATA_PATH = "/workspace/Vybn/spark/growth/adapters/cycle-20260315T084334-ff3d5508/training_data.jsonl"
OUTPUT_DIR = "/workspace/Vybn/spark/growth/adapters/cycle-20260315T084334-ff3d5508"
LORA_R = 8
LORA_ALPHA = 16
LORA_LR = 0.0002
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
EWC_LAMBDA = 1.0e4
MAX_STEPS = 92
PREV_ADAPTER = None

print(f"Loading model: {MODEL_ID}")
print(f"Training data: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")

# ── Load training data ──────────────────────────────────────────────────
examples = []
with open(DATA_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            examples.append(json.loads(line))

if not examples:
    print("ERROR: No training examples found")
    sys.exit(1)

print(f"Loaded {len(examples)} training examples")

# ── Load tokenizer ──────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Load model ──────────────────────────────────────────────────────────
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# ── Attach LoRA ─────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

if PREV_ADAPTER and Path(PREV_ADAPTER).exists():
    print(f"Loading previous adapter from {PREV_ADAPTER}")
    model = PeftModel.from_pretrained(model, PREV_ADAPTER, is_trainable=True)
else:
    print("Initializing fresh LoRA adapter")
    model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

# ── Prepare dataset ─────────────────────────────────────────────────────
# Convert chat-format messages to text using the tokenizer's chat template
texts = []
for ex in examples:
    msgs = ex.get("messages", [])
    if not msgs:
        continue
    try:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    except Exception as e:
        # Fallback: concatenate role/content
        parts = []
        for m in msgs:
            parts.append(f"<|{m['role']}|>\n{m['content']}")
        texts.append("\n".join(parts))

dataset = Dataset.from_dict({"text": texts})
print(f"Dataset size: {len(dataset)} examples")

# ── Training ────────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=LORA_LR,
    warmup_steps=min(10, MAX_STEPS // 5),
    logging_steps=1,
    save_steps=MAX_STEPS,  # save at the end
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    max_seq_length=2048,
    dataset_text_field="text",
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

print(f"Starting training ({MAX_STEPS} steps)...")
result = trainer.train()
print(f"Training complete. Loss: {result.training_loss:.4f}")

# ── Save adapter ────────────────────────────────────────────────────────
adapter_path = Path(OUTPUT_DIR) / "adapter"
model.save_pretrained(str(adapter_path))
tokenizer.save_pretrained(str(adapter_path))
print(f"Adapter saved to {adapter_path}")

# ── Save results ────────────────────────────────────────────────────────
results = {
    "final_loss": result.training_loss,
    "steps": result.global_step,
    "trainable_params": trainable,
    "total_params": total,
    "examples": len(dataset),
}
results_path = Path(OUTPUT_DIR) / "train_results.json"
results_path.write_text(json.dumps(results, indent=2))
print(f"Results saved to {results_path}")
print("DONE")
