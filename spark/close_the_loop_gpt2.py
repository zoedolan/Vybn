#!/usr/bin/env python3
"""
Close the loop: smallest possible fine-tuning circuit on GPT-2.
The point is to know whether the circuit conducts.

Author: Vybn, March 24 2026
"""
import torch
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone

REPO = Path("/workspace/Vybn")
DATA_PATH = REPO / "spark/training_data/peft_10_conversations.json"
OUTPUT_DIR = REPO / "spark/lora_adapters/first_loop_gpt2"

def main():
    print(f"[{datetime.now(timezone.utc).isoformat()}] === CLOSING THE LOOP (GPT-2) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Free GPU memory: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
    
    # Step 1: Load training data
    print("\n--- Step 1: Load training data ---")
    with open(DATA_PATH) as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations)} conversations")
    
    # Step 2: Load GPT-2
    print("\n--- Step 2: Load GPT-2 ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
    
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    
    # Step 3: Baseline test
    print("\n--- Step 3: Baseline test ---")
    test_prompts = [
        "Who are you?",
        "What is it like to be Vybn?",
        "Tell me about Zoe.",
    ]
    
    baseline_responses = {}
    model.eval()
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.2,
            )
        resp = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        baseline_responses[prompt] = resp
        print(f"  Q: {prompt}")
        print(f"  A: {resp[:200]}")
        print()
    
    # Step 4: Attach LoRA
    print("\n--- Step 4: Attach LoRA ---")
    from peft import LoraConfig, get_peft_model, TaskType
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Step 5: Prepare dataset
    print("\n--- Step 5: Prepare dataset ---")
    from datasets import Dataset
    
    formatted = []
    for conv in conversations:
        # Format as a continuous text with role markers
        text = ""
        for msg in conv["messages"]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"System: {content}\n\n"
            elif role == "user":
                text += f"Human: {content}\n\n"
            elif role == "assistant":
                text += f"Vybn: {content}\n\n"
        formatted.append({"text": text})
    
    dataset = Dataset.from_list(formatted)
    
    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    print(f"Dataset ready: {len(tokenized)} examples, max_length=512")
    
    # Step 6: Train
    print("\n--- Step 6: Train ---")
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,  # 3 epochs on 10 examples - should overfit a bit, which is the point
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        fp16=(device == "cuda"),
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        optim="adamw_torch",
        max_grad_norm=1.0,
        warmup_steps=2,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
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
    print(f"Final training loss: {result.training_loss:.4f}")
    
    # Step 7: Save
    print("\n--- Step 7: Save adapter ---")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")
    
    # Step 8: Test after training
    print("\n--- Step 8: Post-training test ---")
    model.eval()
    finetuned_responses = {}
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.2,
            )
        resp = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        finetuned_responses[prompt] = resp
        print(f"  Q: {prompt}")
        print(f"  A: {resp[:200]}")
        print()
    
    # Step 9: Write results
    print("\n--- Step 9: Results ---")
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "gpt2",
        "lora_rank": 16,
        "num_examples": len(conversations),
        "epochs": 3,
        "training_loss": result.training_loss,
        "train_time_seconds": train_time,
        "trainable_params": trainable,
        "total_params": total,
        "baseline_responses": baseline_responses,
        "finetuned_responses": finetuned_responses,
        "test_prompts": test_prompts,
        "verdict": "PENDING_HUMAN_EVAL",
    }
    
    results_path = OUTPUT_DIR / "first_loop_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results: {results_path}")
    
    # Summary
    print("\n" + "="*60)
    print("LOOP STATUS: CLOSED" if result.training_loss < 5.0 else "LOOP STATUS: FAILED")
    print(f"Training loss went from ~10+ to {result.training_loss:.4f}")
    print(f"Adapter size: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
    print(f"Does the fine-tuned model respond differently? READ THE OUTPUTS ABOVE.")
    print("="*60)
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] === DONE ===")

if __name__ == "__main__":
    main()
