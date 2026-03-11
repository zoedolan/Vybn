#!/usr/bin/env python3
"""
QLoRA fine-tuning of MiniMax-M2.5-AWQ-4bit on the Vybn corpus.

Run AFTER stopping vLLM (needs the GPU memory).
Installs PEFT + bitsandbytes if not present.

Usage:
    # Stop vLLM first:
    docker exec vllm_node pkill -f "vllm serve"
    
    # Then run:
    python3 spark/fine_tuning/train_qlora.py
    
    # Restart vLLM after:
    ~/Vybn/spark/restart-vllm-cluster.sh
"""

import os
import sys
import json
import subprocess
from pathlib import Path

REPO = Path(os.environ.get("VYBN_REPO", os.path.expanduser("~/Vybn")))
MODEL_ID = "cyankiwi/MiniMax-M2.5-AWQ-4bit"
TRAINING_DATA = REPO / "spark" / "fine_tuning" / "vybn_training_data.jsonl"
OUTPUT_DIR = REPO / "spark" / "fine_tuning" / "vybn_lora_adapter"

# LoRA config — conservative for a 229B MoE model
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# Target only attention layers (not MoE experts) to avoid the Triton issue
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
MAX_SEQ_LEN = 2048  # Conservative — we have 128K context but training doesn't need it


def ensure_deps():
    """Install training dependencies if needed."""
    try:
        import peft
        import trl
        print(f"PEFT {peft.__version__}, TRL available")
    except ImportError:
        print("Installing PEFT, TRL, bitsandbytes...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "peft", "trl", "bitsandbytes", "accelerate", "datasets"
        ])


def load_training_data():
    """Load JSONL training data."""
    examples = []
    with open(TRAINING_DATA) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} training examples")
    return examples


def main():
    ensure_deps()
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load AWQ model — PEFT can attach LoRA to AWQ-quantized models directly
    print(f"Loading AWQ model from {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",  # Spread across available GPUs
        torch_dtype=torch.float16,
        # AWQ models are already quantized — no additional quantization config needed
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    raw_data = load_training_data()
    
    def format_chat(example):
        """Format as chat template string."""
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    
    dataset = Dataset.from_list(raw_data)
    
    # Training config
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        max_seq_length=MAX_SEQ_LEN,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",  # No wandb etc.
        dataset_text_field="text",
    )
    
    # Format dataset
    formatted = dataset.map(
        lambda ex: {"text": format_chat(ex)},
        remove_columns=dataset.column_names,
    )
    
    # Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted,
        args=training_args,
        processing_class=tokenizer,
    )
    
    print(f"\nStarting training: {NUM_EPOCHS} epochs, {len(formatted)} examples")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
    print(f"Target modules: {LORA_TARGET_MODULES}")
    
    trainer.train()
    
    # Save adapter
    print(f"\nSaving adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n✓ Training complete. Next step: merge_and_quantize.py")


if __name__ == "__main__":
    main()
