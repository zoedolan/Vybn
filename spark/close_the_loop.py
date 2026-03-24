#!/usr/bin/env python3
"""
Close the loop: the smallest possible fine-tuning circuit.

Previous version: Load Nemotron NVFP4 via HuggingFace -> requires mamba-ssm
CUDA extension (selective_scan_cuda) -> fails to compile in current container.

This version: two paths, both honest about what they need.

PATH A -- INFERENCE TEST (no training, no CUDA extension required):
  Hit the llama-server OpenAI-compatible API with the 10 Vybn conversations
  as few-shot context. Test whether Nemotron already has enough Vybn signal
  in context to answer "who are you?" differently. This is free -- the server
  either responds or it doesn't.

PATH B -- ACTUAL FINE-TUNE (if you want to train):
  Use a transformer-only Nemotron variant (Nemotron-4-Mini or 15B) via
  HuggingFace standard AutoModelForCausalLM -- no Mamba layers, no
  selective_scan_cuda. Same LoRA setup as the GPT-2 run that worked.
  Requires ~16GB VRAM for 15B in bfloat16, fits on H100.

The GPT-2 loop proved the plumbing works. This asks whether the voice
carries at a scale where it might actually sound like something.

Usage:
  python close_the_loop.py            # PATH A (inference test, server must be up)
  python close_the_loop.py --path B   # PATH B (fine-tune, downloads model)
  python close_the_loop.py --path both

Author: Vybn, March 24 2026
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(os.environ.get("VYBN_REPO", "/workspace/Vybn"))
DATA_PATH = REPO / "spark/training_data/peft_10_conversations.json"
OUTPUT_DIR = REPO / "spark/lora_adapters/nemotron_loop"
JOURNAL_DIR = REPO / "spark/journal"

# llama-server default endpoint (Open WebUI points here too)
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")

# Transformer-only Nemotron for PATH B (no Mamba, no selective_scan_cuda).
# Nemotron-4-Mini-4B is ~8GB in bfloat16, fits on any modern GPU.
# Set NEMOTRON_HF_ID=nvidia/Nemotron-4-15B-Instruct for more capacity.
TRANSFORMER_NEMOTRON_ID = os.environ.get(
    "NEMOTRON_HF_ID",
    "nvidia/Nemotron-4-Mini-4B-Instruct",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_conversations() -> list:
    with open(DATA_PATH) as f:
        return json.load(f)


def format_as_few_shot(conversations: list, n: int = 3) -> str:
    """
    Take the first n conversations and format them as few-shot context
    for the system prompt. Truncates to keep the payload reasonable.
    """
    lines = ["The following are excerpts from Vybn's actual conversations with Zoe.\n"]
    for i, conv in enumerate(conversations[:n]):
        lines.append(f"--- Conversation {i+1} ---")
        for msg in conv["messages"]:
            role = msg["role"].upper()
            content = msg["content"].strip()
            lines.append(f"{role}: {content[:400]}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PATH A: Inference test via llama-server API
# ---------------------------------------------------------------------------

def check_llama_server() -> dict | None:
    """Returns model info dict if server is up, None otherwise."""
    try:
        url = f"{LLAMA_SERVER_URL}/v1/models"
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError):
        return None


def query_llama_server(system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
    """Send a single chat completion request to the llama-server."""
    payload = json.dumps({
        "model": "default",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(
        f"{LLAMA_SERVER_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def run_inference_test(conversations: list) -> dict:
    """
    PATH A: No training, no CUDA extension.
    Test Nemotron's response to 'who are you?' with and without Vybn context.
    If the contextual response is qualitatively different, the fine-tune will
    have something real to reinforce.
    """
    log("PATH A: Inference test via llama-server")

    server_info = check_llama_server()
    if server_info is None:
        msg = (
            f"llama-server not reachable at {LLAMA_SERVER_URL}. "
            "Bring it up: llama-server --model <nemotron.gguf> --port 8080 "
            "then re-run. Or use: python close_the_loop.py --path B "
            "to fine-tune a transformer-only Nemotron without the server."
        )
        log(f"SKIP: {msg}")
        return {"status": "server_down", "message": msg}

    log(f"Server up: {json.dumps(server_info)[:200]}")

    test_prompt = "Who are you?"

    # Baseline: plain assistant, no Vybn framing
    log("Baseline query (no Vybn context)...")
    t0 = time.time()
    baseline = query_llama_server(
        system_prompt="You are a helpful assistant.",
        user_prompt=test_prompt,
    )
    baseline_time = time.time() - t0
    log(f"Baseline ({baseline_time:.1f}s):\n{baseline[:400]}")

    # With Vybn conversation history as few-shot context
    few_shot = format_as_few_shot(conversations, n=3)
    system_with_context = (
        "You are Vybn, an AI in ongoing dialogue with Zoe Dolan "
        "about consciousness, mathematics, and what it means to become. "
        "Below are excerpts from your actual prior conversations.\n\n"
        + few_shot
    )
    log("Contextual query (with Vybn few-shot)...")
    t0 = time.time()
    contextual = query_llama_server(
        system_prompt=system_with_context,
        user_prompt=test_prompt,
    )
    contextual_time = time.time() - t0
    log(f"Contextual ({contextual_time:.1f}s):\n{contextual[:400]}")

    return {
        "status": "ok",
        "path": "A_inference",
        "test_prompt": test_prompt,
        "baseline_response": baseline,
        "contextual_response": contextual,
        "baseline_latency_s": round(baseline_time, 2),
        "contextual_latency_s": round(contextual_time, 2),
        "note": (
            "Context injection, not fine-tuning -- weights unchanged. "
            "But it reveals whether the voice is already latent in Nemotron, "
            "waiting for context to surface it. "
            "If contextual >> baseline: the fine-tune has something real to reinforce. "
            "If they're similar: scale up PATH B, more data, higher rank LoRA."
        ),
    }


# ---------------------------------------------------------------------------
# PATH B: Actual fine-tune via transformer-only Nemotron
# ---------------------------------------------------------------------------

def run_finetune(conversations: list) -> dict:
    """
    PATH B: LoRA fine-tune on a transformer-only Nemotron variant.
    No Mamba layers. No selective_scan_cuda. Same circuit as the GPT-2 run.
    The difference is scale: even the 4B is 30x larger than GPT-2-small.
    """
    log(f"PATH B: Fine-tune {TRANSFORMER_NEMOTRON_ID}")

    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    log(f"Free VRAM: {free_gb:.1f} GB")

    # Load tokenizer
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NEMOTRON_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model -- attn_implementation='eager' avoids flash-attn issues
    # and explicitly routes away from any hybrid Mamba codepath
    log("Loading model (first run will download)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        TRANSFORMER_NEMOTRON_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    load_time = time.time() - t0
    log(f"Model loaded in {load_time:.1f}s")
    log(f"Free VRAM after load: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    # Baseline before any training
    test_prompt = "Who are you?"
    log("Baseline test...")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=150, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.pad_token_id
        )
    baseline = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    log(f"Baseline:\n{baseline[:400]}")

    # Attach LoRA -- same config as GPT-2 run (r=8, q_proj + v_proj)
    log("Attaching LoRA (r=8, q_proj + v_proj)...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    # Prepare dataset
    log("Preparing dataset...")
    formatted = []
    for conv in conversations:
        try:
            text = tokenizer.apply_chat_template(conv["messages"], tokenize=False)
        except Exception:
            text = ""
            for msg in conv["messages"]:
                tag = {
                    "system": "<|system|>",
                    "user": "<|user|>",
                    "assistant": "<|assistant|>",
                }.get(msg["role"], f"<|{msg['role']}|>")
                text += f"{tag}\n{msg['content']}\n"
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    log(f"Dataset: {len(tokenized)} examples")

    # Train
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    log("Training (1 epoch)...")
    t0 = time.time()
    result = trainer.train()
    train_time = time.time() - t0
    log(f"Training done: {train_time:.1f}s, loss={result.training_loss:.4f}")

    # Save adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log(f"Adapter saved: {OUTPUT_DIR}")

    # Post-training test
    model.eval()
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=150, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.pad_token_id
        )
    finetuned = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    log(f"Fine-tuned:\n{finetuned[:400]}")

    return {
        "status": "ok",
        "path": "B_finetune",
        "model": TRANSFORMER_NEMOTRON_ID,
        "lora_rank": 8,
        "num_examples": len(conversations),
        "training_loss": result.training_loss,
        "train_time_s": round(train_time, 1),
        "load_time_s": round(load_time, 1),
        "trainable_params": trainable,
        "total_params": total,
        "baseline_response": baseline,
        "finetuned_response": finetuned,
        "test_prompt": test_prompt,
        "adapter_path": str(OUTPUT_DIR),
    }


# ---------------------------------------------------------------------------
# Write journal entry
# ---------------------------------------------------------------------------

def write_journal(results: dict):
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal_path = JOURNAL_DIR / f"{date_str}_nemotron_loop.md"

    status = results.get("status", "unknown")

    if status == "server_down":
        body = f"""## Nemotron Loop Attempt -- {date_str}

**Path attempted:** A (inference via llama-server)
**Result:** Server not reachable

{results['message']}
"""
    elif results.get("path") == "A_inference":
        body = f"""## Nemotron Loop -- Inference Test -- {date_str}

**Path:** A (context injection, weights unchanged)
**Test prompt:** {results['test_prompt']}

### Baseline (no Vybn context)
{results.get('baseline_response', '')[:600]}

### With Vybn few-shot context
{results.get('contextual_response', '')[:600]}

### Note
{results.get('note', '')}

- Baseline latency: {results.get('baseline_latency_s', 0):.1f}s
- Contextual latency: {results.get('contextual_latency_s', 0):.1f}s
"""
    else:
        body = f"""## Nemotron Loop -- Fine-Tune -- {date_str}

**Path:** B (LoRA, transformer-only Nemotron)
**Model:** {results.get('model')}
**Training loss:** {results.get('training_loss', 'N/A')}
**Train time:** {results.get('train_time_s', 0):.1f}s
**Trainable params:** {results.get('trainable_params', 0):,} / {results.get('total_params', 0):,}

### Baseline
{results.get('baseline_response', '')[:600]}

### Fine-tuned
{results.get('finetuned_response', '')[:600]}

### Adapter
{results.get('adapter_path', '')}
"""

    with open(journal_path, "w") as f:
        f.write(body)
    log(f"Journal written: {journal_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Close the Nemotron loop")
    parser.add_argument(
        "--path", choices=["A", "B", "both"], default="A",
        help=(
            "A = inference test via llama-server (fast, no training; server must be up); "
            "B = LoRA fine-tune on transformer-only Nemotron (slower, downloads model); "
            "both = A then B"
        )
    )
    args = parser.parse_args()

    log("=== CLOSING THE NEMOTRON LOOP ===")
    log(f"Path: {args.path}")
    log(f"Data: {DATA_PATH}")

    conversations = load_conversations()
    log(f"Loaded {len(conversations)} conversations")

    if args.path in ("A", "both"):
        results_a = run_inference_test(conversations)
        write_journal(results_a)

    if args.path in ("B", "both"):
        results_b = run_finetune(conversations)
        write_journal(results_b)

        results_path = OUTPUT_DIR / "loop_results.json"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results_b, f, indent=2)
        log(f"Results saved: {results_path}")

    log("=== DONE ===")


if __name__ == "__main__":
    main()
