#!/usr/bin/env python3
"""spark.growth.test_loop_gpt2 — GPT-2 proof-of-concept for the training loop.

Validates the training core — delta extraction → LoRA fine-tuning → adapter saved —
using GPT-2 instead of Nemotron 120B.  No SIGSTOP, no Docker, no hot-load.

Why GPT-2:
    The full pipeline (peft_train.py → train_cycle.py → merge_cycle.py) is solid
    infrastructure, but it has never completed a full training cycle because the
    120B model exceeds single-node memory.  GPT-2 loads in ~3 GB and trains in
    minutes, letting us prove that every upstream step works before returning to
    the 120B problem via QLoRA or two-node FSDP.

What this exercises:
    1. buffer.jsonl loading and surprise-weighted selection (same as trigger.py)
    2. Chat-format tokenization (same template as peft_train.py)
    3. PEFT LoRA injection with the same config from growth_config.yaml
    4. x-weighted SFT loss with composite quality weights
    5. Adapter save as .safetensors

What this skips:
    - SIGSTOP/SIGCONT memory management (not needed at ~3 GB)
    - Docker container execution (runs directly on host)
    - GGUF conversion + llama-server hot-load (architecture mismatch — GPT-2
      adapter can't be loaded into the Nemotron serving model)

The governing equation M′ = α·M + x·e^(iθ) still applies — α is the LoRA
adapter, x·e^(iθ) is the phase-rotated training delta.  Only M is smaller.

Usage (on the Spark, directly — no Docker):
    python3 spark/growth/test_loop_gpt2.py

    # Or with explicit paths:
    python3 spark/growth/test_loop_gpt2.py \\
        --buffer spark/growth/buffer.jsonl \\
        --model gpt2 \\
        --top-k 50 \\
        --epochs 1 \\
        --output-dir spark/growth/adapters/test-gpt2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Buffer loading — reads buffer.jsonl directly, no NestedMemory needed
# ---------------------------------------------------------------------------

def load_buffer(buffer_path: str, top_k: int = 50) -> list[dict]:
    """Load buffer entries and return the top-k by surprise score.

    buffer.jsonl entries have metadata.surprise — we sort descending and
    take the top_k.  This mirrors what DeltaExtractor does but without
    the NestedMemory / StubNested dependency that caused the ingest bug.
    """
    entries = []
    with open(buffer_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # buffer.jsonl may have either format:
            #   a) "messages" list (chat format from peft_train.py)
            #   b) raw "content" string (from GrowthBuffer)
            # Wrap raw content into messages format so downstream
            # tokenization works the same way.
            if "messages" in obj and obj["messages"]:
                entries.append(obj)
            elif obj.get("content"):
                obj["messages"] = [{"role": "assistant", "content": obj["content"]}]
                entries.append(obj)

    if not entries:
        raise RuntimeError(f"No valid entries in {buffer_path}")

    # Sort by surprise score (descending) — highest surprise first.
    # buffer.jsonl uses top-level "surprise_score"; metadata.surprise
    # is the nested form from DeltaExtractor. Check both.
    def _surprise(e: dict) -> float:
        meta_s = float(e.get("metadata", {}).get("surprise", 0.0))
        top_s = float(e.get("surprise_score", 0.0))
        return max(meta_s, top_s)

    entries.sort(key=_surprise, reverse=True)

    selected = entries[:top_k]
    print(f"[test_loop] loaded {len(entries)} entries, selected top {len(selected)} by surprise")

    if selected:
        surprises = [_surprise(e) for e in selected]
        print(f"[test_loop] surprise range: {min(surprises):.4f} — {max(surprises):.4f}")

    return selected


def get_x_weight(example: dict) -> float:
    """Extract composite x-weight (same as peft_train.py).

    Falls back to top-level holonomy_score if metadata.x_weight
    isn't present (raw buffer entries use holonomy_score directly).
    """
    meta = example.get("metadata", {})
    xw = meta.get("x_weight", {})
    composite = xw.get("composite")
    if composite is not None:
        return float(composite)
    # Fallback: use holonomy_score as the weight
    h = example.get("holonomy_score")
    if h is not None:
        return float(h)
    return 1.0


# ---------------------------------------------------------------------------
# Phase computation — same as peft_train.py
# ---------------------------------------------------------------------------

def compute_encounter_phase(examples: list[dict]) -> dict:
    import hashlib
    from collections import Counter

    now = datetime.now(timezone.utc)
    source_counts: Counter[str] = Counter()
    x_weights: list[float] = []

    for ex in examples:
        meta = ex.get("metadata", {})
        source_counts[meta.get("source_type", "unknown")] += 1
        xw = meta.get("x_weight", {})
        if "composite" in xw:
            x_weights.append(float(xw["composite"]))

    content_hash = hashlib.sha256(
        json.dumps(dict(source_counts), sort_keys=True).encode()
        + now.isoformat().encode()
    ).hexdigest()
    theta_radians = (int(content_hash[:8], 16) / 0xFFFFFFFF) * 2 * math.pi

    return {
        "theta_radians": round(theta_radians, 6),
        "training_timestamp": now.isoformat(),
        "source_distribution": dict(source_counts),
        "n_examples": len(examples),
        "mean_x_weight": round(sum(x_weights) / len(x_weights), 4) if x_weights else None,
        "model": "gpt2 (proof-of-concept)",
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_gpt2(
    examples: list[dict],
    model_name: str = "gpt2",
    output_dir: str = "spark/growth/adapters/test-gpt2",
    epochs: int = 1,
    config_path: str | None = None,
) -> dict:
    """LoRA fine-tune GPT-2 on buffer entries.

    Uses the same PEFT config as peft_train.py (from growth_config.yaml)
    and the same x-weighted SFT loss.  The only differences:
      - Model is GPT-2 instead of Nemotron 120B
      - target_modules mapped to GPT-2's attention projections
      - No Docker, no SIGSTOP, no hot-load
    """
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import yaml

    # Load config
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent / "growth_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lora_cfg = cfg.get("lora", {})
    rank = lora_cfg.get("fast_rank", 8)
    alpha = lora_cfg.get("alpha", 16)
    lr = lora_cfg.get("fast_lr", 2e-4)

    # GPT-2 attention projections (equivalent to q_proj/k_proj/v_proj/o_proj)
    # GPT-2 uses Conv1D named c_attn (fused QKV) and c_proj (output)
    gpt2_target_modules = ["c_attn", "c_proj"]

    theta = compute_encounter_phase(examples)
    print(f"[test_loop] θ = {theta['theta_radians']:.4f} rad")
    print(f"[test_loop] mean_x_weight = {theta.get('mean_x_weight')}")

    # Load GPT-2
    print(f"[test_loop] loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,  # GPT-2 is small enough for fp32
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[test_loop] model on {device}, {sum(p.numel() for p in model.parameters()):,} params")

    # PEFT LoRA — same rank/alpha as the real pipeline
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        target_modules=gpt2_target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[test_loop] LoRA: {trainable:,} trainable / {total:,} total "
        f"({100 * trainable / total:.2f}%)"
    )

    # Tokenize examples — same chat template as peft_train.py
    max_seq_len = 512
    all_input_ids = []
    all_labels = []
    all_x_weights = []

    for ex in examples:
        messages = ex["messages"]
        text_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text_parts.append(f"<|{role}|>\n{content}")
        text = "\n".join(text_parts)

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_x_weights.append(get_x_weight(ex))

    dataset_input_ids = torch.stack(all_input_ids).to(device)
    dataset_labels = torch.stack(all_labels).to(device)
    dataset_x_weights = torch.tensor(all_x_weights, dtype=torch.float32, device=device)

    n_examples = len(all_input_ids)
    batch_size = min(4, n_examples)
    print(f"[test_loop] {n_examples} examples, batch_size={batch_size}, epochs={epochs}")

    # Optimizer — AdamW (MuonAdamW requires 2D LoRA matrices which GPT-2's
    # Conv1D projections provide, but plain AdamW keeps this test simple)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    model.train()
    steps_trained = 0
    loss_history: list[float] = []
    t_start = time.monotonic()

    for epoch in range(epochs):
        indices = torch.randperm(n_examples, device=device)

        for batch_start in range(0, n_examples, batch_size):
            batch_idx = indices[batch_start : batch_start + batch_size]
            input_ids = dataset_input_ids[batch_idx]
            labels = dataset_labels[batch_idx]
            x_w = dataset_x_weights[batch_idx]

            # Per-sample x-weighted SFT loss — same as peft_train.py
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            token_losses = torch.nn.functional.cross_entropy(
                flat_logits, flat_labels, reduction="none"
            ).view(len(batch_idx), -1)
            mask = (shift_labels != -100).float()
            per_sample_loss = (token_losses * mask).sum(-1) / mask.sum(-1).clamp(min=1)
            loss = (per_sample_loss * x_w).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            steps_trained += 1
            loss_val = loss.item()
            loss_history.append(loss_val)

            if steps_trained % 5 == 0 or steps_trained == 1:
                elapsed = time.monotonic() - t_start
                print(
                    f"[test_loop] epoch={epoch+1}/{epochs} step={steps_trained} "
                    f"loss={loss_val:.4f} elapsed={elapsed:.1f}s"
                )

    elapsed = time.monotonic() - t_start

    # Loss curve summary
    if loss_history:
        first_5 = loss_history[:5]
        last_5 = loss_history[-5:]
        print(f"\n[test_loop] === Loss Curve ===")
        print(f"[test_loop] first 5 steps: {[round(l, 4) for l in first_5]}")
        print(f"[test_loop] last 5 steps:  {[round(l, 4) for l in last_5]}")
        improvement = (first_5[0] - last_5[-1]) / first_5[0] * 100
        print(f"[test_loop] improvement:   {improvement:.1f}%")

    # Save adapter
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_path / "adapter"
    model.save_pretrained(adapter_dir)

    adapter_file = None
    for f in adapter_dir.rglob("*.safetensors"):
        adapter_file = str(f)
        break
    if adapter_file is None:
        adapter_file = str(adapter_dir / "adapter_model.safetensors")

    print(f"\n[test_loop] adapter saved: {adapter_file}")
    print(f"[test_loop] total time: {elapsed:.1f}s")
    print(f"[test_loop] final loss: {loss_history[-1]:.4f}")

    result = {
        "model": model_name,
        "proof_of_concept": True,
        "final_loss": round(loss_history[-1], 6),
        "initial_loss": round(loss_history[0], 6),
        "steps_trained": steps_trained,
        "n_examples": n_examples,
        "epochs": epochs,
        "adapter_path": adapter_file,
        "elapsed_seconds": round(elapsed, 1),
        "theta": theta,
        "loss_history": [round(l, 4) for l in loss_history],
        "lora_rank": rank,
        "lora_alpha": alpha,
        "target_modules": gpt2_target_modules,
        "note": (
            "This adapter targets GPT-2 and cannot be hot-loaded into the "
            "Nemotron serving model. It proves the training loop works — "
            "delta selection, tokenization, x-weighted loss, LoRA injection, "
            "adapter save. The 120B training path (QLoRA or two-node FSDP) "
            "can now be attempted with confidence that everything upstream "
            "of the model load is correct."
        ),
    }

    print(f"\n{json.dumps(result, ensure_ascii=False, indent=2)}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT-2 proof-of-concept for Vybn's growth engine training loop"
    )
    parser.add_argument(
        "--buffer",
        default="spark/growth/buffer.jsonl",
        help="Path to buffer.jsonl (default: spark/growth/buffer.jsonl)",
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="HuggingFace model name (default: gpt2)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of highest-surprise entries to train on (default: 50)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        default="spark/growth/adapters/test-gpt2",
        help="Output directory for the adapter (default: spark/growth/adapters/test-gpt2)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to growth_config.yaml (default: auto-detect)",
    )
    args = parser.parse_args()

    examples = load_buffer(args.buffer, top_k=args.top_k)
    train_gpt2(
        examples,
        model_name=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
