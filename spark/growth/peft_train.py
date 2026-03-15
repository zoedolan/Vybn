#!/usr/bin/env python3
"""spark.growth.peft_train — LoRA fine-tuning inside the vllm_node container.

Executes one growth cycle's training:
    M′ = α·M + x·e^(iθ)

Where α is the LoRA adapter trained via PEFT with MuonAdamW, x·e^(iθ) is
the phase-rotated training delta from DeltaExtractor.

Usage (inside vllm_node container):
    python3 peft_train.py \\
        --data /workspace/Vybn/spark/growth/adapters/<cycle>/training_data.jsonl \\
        --output-dir /workspace/Vybn/spark/growth/adapters/<cycle>/ \\
        --config /workspace/Vybn/spark/growth/growth_config.yaml

Prints a JSON result object to stdout on completion:
    {"final_loss": ..., "steps_trained": ..., "adapter_path": ..., "theta": {...}}
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import torch
import yaml


# ---------------------------------------------------------------------------
# GC discipline (inlined from eval_harness.py for container independence)
# ---------------------------------------------------------------------------

@contextmanager
def gc_discipline(collect_every_n_steps: int = 5000) -> Iterator[None]:
    """Disable Python GC during training to avoid ~500ms stalls."""
    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
        gc.collect()


def gc_checkpoint(step: int, collect_every: int = 5000) -> None:
    """Periodic GC collection inside a gc_discipline block."""
    if step > 0 and step % collect_every == 0:
        gc.collect()


# ---------------------------------------------------------------------------
# TimeBudget (inlined from eval_harness.py for container independence)
# ---------------------------------------------------------------------------

class TimeBudget:
    """Wall-clock training budget tracker."""

    def __init__(self, budget_seconds: int, warmup_steps: int = 0) -> None:
        self.budget_seconds = budget_seconds
        self.warmup_steps = warmup_steps
        self._training_time = 0.0
        self._step = 0

    def tick(self, step_duration: float) -> None:
        self._step += 1
        if self._step > self.warmup_steps:
            self._training_time += step_duration

    @property
    def exhausted(self) -> bool:
        return self._training_time >= self.budget_seconds

    @property
    def elapsed(self) -> float:
        return self._training_time

    @property
    def remaining(self) -> float:
        return max(0, self.budget_seconds - self._training_time)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chat_jsonl(path: str) -> list[dict]:
    """Load JSONL file containing chat-format training examples.

    Each line: {"messages": [{"role": ..., "content": ...}, ...], "metadata": {...}}
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj and obj["messages"]:
                examples.append(obj)
    return examples


def compute_encounter_phase(examples: list[dict]) -> dict:
    """Compute θ — the encounter phase of the training data.

    θ encodes the temporal/contextual orientation:
    - Timestamp of training
    - Source distribution (breath vs reflection vs replay)
    - Temporal spread of training examples
    """
    now = datetime.now(timezone.utc)
    source_counts: Counter[str] = Counter()
    timestamps: list[str] = []

    for ex in examples:
        meta = ex.get("metadata", {})
        source_type = meta.get("source_type", "unknown")
        source_counts[source_type] += 1
        ts = meta.get("ingested_at", "")
        if ts:
            timestamps.append(ts)

    # Temporal spread: difference between earliest and latest example
    temporal_spread_hours = 0.0
    if len(timestamps) >= 2:
        sorted_ts = sorted(timestamps)
        try:
            earliest = datetime.fromisoformat(sorted_ts[0].replace("Z", "+00:00"))
            latest = datetime.fromisoformat(sorted_ts[-1].replace("Z", "+00:00"))
            temporal_spread_hours = (latest - earliest).total_seconds() / 3600
        except (ValueError, TypeError):
            pass

    # θ as a hash-derived angle in [0, 2π) — deterministic from data content
    content_hash = hashlib.sha256(
        json.dumps(dict(source_counts), sort_keys=True).encode()
        + now.isoformat().encode()
    ).hexdigest()
    theta_radians = (int(content_hash[:8], 16) / 0xFFFFFFFF) * 2 * math.pi

    return {
        "theta_radians": round(theta_radians, 6),
        "training_timestamp": now.isoformat(),
        "source_distribution": dict(source_counts),
        "temporal_spread_hours": round(temporal_spread_hours, 2),
        "n_examples": len(examples),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    output_dir: str,
    config_path: str,
) -> dict:
    """Run LoRA fine-tuning with MuonAdamW.

    Returns a result dict printed as JSON to stdout by the CLI wrapper.
    """
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Import MuonAdamW — handle both in-container path and package import
    try:
        from spark.growth.muon_adamw import MuonAdamW, build_param_groups
    except ImportError:
        # Running inside container where spark may not be a package
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from muon_adamw import MuonAdamW, build_param_groups

    # --- Load config ---
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lora_cfg = cfg.get("lora", {})
    ewc_cfg = cfg.get("ewc", {})

    rank = lora_cfg.get("fast_rank", 8)
    alpha = lora_cfg.get("alpha", 16)
    target_modules = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    lr = lora_cfg.get("fast_lr", 2e-4)
    time_budget_seconds = lora_cfg.get("time_budget_seconds", 7200)
    gc_collect_every = lora_cfg.get("gc_collect_every", 5000)
    epochs = lora_cfg.get("epochs", 2)

    # --- Load training data ---
    examples = load_chat_jsonl(data_path)
    if not examples:
        raise RuntimeError(f"No training examples found in {data_path}")

    # Compute encounter phase θ
    theta = compute_encounter_phase(examples)

    print(f"[peft_train] {len(examples)} training examples loaded", file=sys.stderr)
    print(f"[peft_train] θ = {theta['theta_radians']:.4f} rad", file=sys.stderr)

    # --- Load base model (4-bit quantized) ---
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    )
    # Also check for snapshot dirs
    snapshots_dir = Path(model_path) / "snapshots"
    if snapshots_dir.exists():
        snapshot_dirs = sorted(snapshots_dir.iterdir())
        if snapshot_dirs:
            model_path = str(snapshot_dirs[-1])

    print(f"[peft_train] Loading model from {model_path}", file=sys.stderr)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # --- Apply LoRA ---
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[peft_train] LoRA: {trainable:,} trainable / {total:,} total params "
        f"({100 * trainable / total:.2f}%)",
        file=sys.stderr,
    )

    # --- Build optimizer with MuonAdamW ---
    param_groups = build_param_groups(
        model,
        muon_lr=lr,
        adamw_lr=lr,
        weight_decay=0.2,
    )
    optimizer = MuonAdamW(param_groups)

    # --- Tokenize training data ---
    max_seq_len = 512
    all_input_ids = []
    all_labels = []

    for ex in examples:
        messages = ex["messages"]
        # Build text from messages
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
        # Labels = input_ids shifted (causal LM), mask padding with -100
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    dataset_input_ids = torch.stack(all_input_ids)
    dataset_labels = torch.stack(all_labels)
    n_examples = len(all_input_ids)

    # --- Training loop ---
    batch_size = min(4, n_examples)
    budget = TimeBudget(budget_seconds=time_budget_seconds, warmup_steps=1)
    model.train()

    steps_trained = 0
    running_loss = 0.0
    final_loss = float("inf")

    print(
        f"[peft_train] Training: {epochs} epochs, batch_size={batch_size}, "
        f"budget={time_budget_seconds}s",
        file=sys.stderr,
    )

    with gc_discipline(collect_every_n_steps=gc_collect_every):
        for epoch in range(epochs):
            if budget.exhausted:
                print(f"[peft_train] Time budget exhausted at epoch {epoch}", file=sys.stderr)
                break

            # Shuffle indices each epoch
            indices = torch.randperm(n_examples)

            for batch_start in range(0, n_examples, batch_size):
                if budget.exhausted:
                    break

                t0 = time.monotonic()

                batch_idx = indices[batch_start : batch_start + batch_size]
                input_ids = dataset_input_ids[batch_idx].to(model.device)
                labels = dataset_labels[batch_idx].to(model.device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                step_time = time.monotonic() - t0
                budget.tick(step_time)
                steps_trained += 1

                loss_val = loss.item()
                running_loss = 0.9 * running_loss + 0.1 * loss_val if running_loss > 0 else loss_val
                final_loss = loss_val

                gc_checkpoint(steps_trained, collect_every=gc_collect_every)

                if steps_trained % 10 == 0:
                    print(
                        f"[peft_train] step={steps_trained} loss={loss_val:.4f} "
                        f"smooth={running_loss:.4f} "
                        f"elapsed={budget.elapsed:.0f}s/{time_budget_seconds}s",
                        file=sys.stderr,
                    )

    # --- Save adapter ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_path / "adapter"
    model.save_pretrained(adapter_dir)

    # Find the .safetensors file
    adapter_file = None
    for f in adapter_dir.rglob("*.safetensors"):
        adapter_file = str(f)
        break
    if adapter_file is None:
        adapter_file = str(adapter_dir / "adapter_model.safetensors")

    print(f"[peft_train] Adapter saved to {adapter_file}", file=sys.stderr)
    print(f"[peft_train] Final loss: {final_loss:.4f}, steps: {steps_trained}", file=sys.stderr)

    result = {
        "final_loss": round(final_loss, 6),
        "steps_trained": steps_trained,
        "adapter_path": adapter_file,
        "n_examples": n_examples,
        "epochs_completed": epoch + 1 if steps_trained > 0 else 0,
        "theta": theta,
        "elapsed_seconds": round(budget.elapsed, 1),
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning with MuonAdamW for Vybn growth engine",
    )
    parser.add_argument("--data", required=True, help="Path to training JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory for adapter output")
    parser.add_argument("--config", required=True, help="Path to growth_config.yaml")
    args = parser.parse_args()

    result = train(
        data_path=args.data,
        output_dir=args.output_dir,
        config_path=args.config,
    )

    # Print result as JSON to stdout (train_cycle.py reads this)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
