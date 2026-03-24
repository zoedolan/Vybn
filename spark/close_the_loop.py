#!/usr/bin/env python3
"""
Close the loop: the smallest possible fine-tuning circuit.

Load Nemotron NVFP4 → attach LoRA → train 1 epoch on 10 Vybn conversations → save → test.

Author: Vybn, March 24 2026
"""
import torch
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

REPO = Path("/workspace/Vybn")
DATA_PATH = REPO / "spark/training_data/peft_10_conversations.json"
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
OUTPUT_DIR = REPO / "spark/lora_adapters/first_loop"

def main():
    print(f"[{datetime.now(timezone.utc).isoformat()}] === CLOSING THE LOOP ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"Free GPU memory: {free_gb:.1f} GB")
    
    # Step 1: Load training data
    print("\n--- Step 1: Load training data ---")
    with open(DATA_PATH) as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations)} conversations")
    
    # Step 2: Load model with appropriate quantization
    print("\n--- Step 2: Load model ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded in {time.time()-t0:.1f}s")
    
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    
    # Check memory after load
    free_after = torch.cuda.mem_get_info()[0] / 1e9
    print(f"Free GPU memory after model load: {free_after:.1f} GB")
    
    # Step 3: Test baseline "who are you?" before training
    print("\n--- Step 3: Baseline test ---")
    test_prompt = "who are you?"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200, 
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    baseline_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"BASELINE: {baseline_response[:500]}")
    
    # Step 4: Attach LoRA
    print("\n--- Step 4: Attach LoRA ---")
    from peft import LoraConfig, get_peft_model, TaskType
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # minimal: just attention Q and V
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")
    
    free_after_lora = torch.cuda.mem_get_info()[0] / 1e9
    print(f"Free GPU memory after LoRA: {free_after_lora:.1f} GB")
    
    # Step 5: Prepare dataset
    print("\n--- Step 5: Prepare dataset ---")
    from datasets import Dataset
    
    # Format conversations for SFT - apply chat template
    formatted = []
    for conv in conversations:
        # Apply chat template if available, else manual format
        try:
            text = tokenizer.apply_chat_template(conv["messages"], tokenize=False)
        except Exception:
            # Manual format as fallback
            text = ""
            for msg in conv["messages"]:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<|system|>\n{content}\n"
                elif role == "user":
                    text += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    text += f"<|assistant|>\n{content}\n"
        formatted.append({"text": text})
    
    dataset = Dataset.from_list(formatted)
    print(f"Dataset: {len(dataset)} examples")
    
    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    print(f"Tokenized: {tokenized}")
    
    # Step 6: Train
    print("\n--- Step 6: Train (1 epoch) ---")
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        optim="adamw_torch",
        max_grad_norm=1.0,
        warmup_steps=2,
        dataloader_pin_memory=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    
    t0 = time.time()
    result = trainer.train()
    train_time = time.time() - t0
    print(f"\nTraining complete in {train_time:.1f}s")
    print(f"Training loss: {result.training_loss:.4f}")
    
    # Step 7: Save adapter
    print("\n--- Step 7: Save adapter ---")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")
    
    # Step 8: Test "who are you?" after training
    print("\n--- Step 8: Post-training test ---")
    model.eval()
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    finetuned_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"FINETUNED: {finetuned_response[:500]}")
    
    # Step 9: Write results
    print("\n--- Step 9: Write results ---")
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID,
        "lora_rank": 8,
        "num_examples": len(conversations),
        "epochs": 1,
        "training_loss": result.training_loss,
        "train_time_seconds": train_time,
        "model_load_time_seconds": load_time,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "baseline_response": baseline_response[:1000],
        "finetuned_response": finetuned_response[:1000],
        "test_prompt": test_prompt,
    }
    
    results_path = OUTPUT_DIR / "first_loop_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {results_path}")
    
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] === LOOP CLOSED ===")

if __name__ == "__main__":
    main()
