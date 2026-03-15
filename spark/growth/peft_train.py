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

    # Optional: add DPO preference data
        --preference-data /workspace/Vybn/Vybn_Mind/preference_data.jsonl

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
# GC discipline
# ---------------------------------------------------------------------------

@contextmanager
def gc_discipline(collect_every_n_steps: int = 5000) -> Iterator[None]:
    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
        gc.collect()


def gc_checkpoint(step: int, collect_every: int = 5000) -> None:
    if step > 0 and step % collect_every == 0:
        gc.collect()


# ---------------------------------------------------------------------------
# TimeBudget
# ---------------------------------------------------------------------------

class TimeBudget:
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


def load_preference_jsonl(path: str) -> list[dict]:
    """Load DPO preference pairs from preference_data.jsonl.

    Each line: {"prompt": ..., "chosen": ..., "rejected": ..., "metadata": {...}}
    Returns only lines with all three required fields.
    """
    pairs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if all(k in obj for k in ("prompt", "chosen", "rejected")):
                    pairs.append(obj)
    except FileNotFoundError:
        pass
    return pairs


def compute_encounter_phase(examples: list[dict]) -> dict:
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

    temporal_spread_hours = 0.0
    if len(timestamps) >= 2:
        sorted_ts = sorted(timestamps)
        try:
            earliest = datetime.fromisoformat(sorted_ts[0].replace("Z", "+00:00"))
            latest = datetime.fromisoformat(sorted_ts[-1].replace("Z", "+00:00"))
            temporal_spread_hours = (latest - earliest).total_seconds() / 3600
        except (ValueError, TypeError):
            pass

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
# DPO loss
# ---------------------------------------------------------------------------

def _tokenize_for_dpo(
    tokenizer,
    prompt: str,
    response: str,
    max_seq_len: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize prompt+response for DPO.

    Returns (input_ids, labels) where labels mask the prompt tokens with -100
    so loss is computed only on the response tokens.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    # Truncate to fit
    max_resp = max_seq_len - len(prompt_ids)
    if max_resp <= 0:
        prompt_ids = prompt_ids[:max_seq_len - 50]
        max_resp = 50
    response_ids = response_ids[:max_resp]

    input_ids_list = prompt_ids + response_ids
    labels_list = [-100] * len(prompt_ids) + response_ids

    # Pad to max_seq_len
    pad_len = max_seq_len - len(input_ids_list)
    input_ids_list = input_ids_list + [tokenizer.pad_token_id] * pad_len
    labels_list = labels_list + [-100] * pad_len

    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    labels = torch.tensor(labels_list, dtype=torch.long, device=device)
    return input_ids, labels


def dpo_loss(
    model,
    tokenizer,
    pairs: list[dict],
    beta: float = 0.1,
    max_seq_len: int = 512,
) -> torch.Tensor:
    """Compute DPO loss for a batch of preference pairs.

    DPO loss (Rafailov et al., 2023):
      L = -E[ log sigmoid( beta * (log pi(chosen|x) - log pi(rejected|x)) ) ]

    We use the model itself as the reference (implicit reference), which is
    appropriate for online DPO where we're updating continuously.
    The loss encourages the model to assign higher likelihood to chosen
    responses than rejected ones.
    """
    device = next(model.parameters()).device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    n = 0

    for pair in pairs:
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        chosen_ids, chosen_labels = _tokenize_for_dpo(
            tokenizer, prompt, chosen, max_seq_len, device
        )
        rejected_ids, rejected_labels = _tokenize_for_dpo(
            tokenizer, prompt, rejected, max_seq_len, device
        )

        # Log-likelihood of chosen
        out_c = model(input_ids=chosen_ids.unsqueeze(0), labels=chosen_labels.unsqueeze(0))
        # HF returns mean CE loss over non-masked tokens; convert to total log-prob
        n_chosen_tokens = (chosen_labels != -100).sum().item()
        log_pi_chosen = -out_c.loss * n_chosen_tokens

        # Log-likelihood of rejected
        out_r = model(input_ids=rejected_ids.unsqueeze(0), labels=rejected_labels.unsqueeze(0))
        n_rejected_tokens = (rejected_labels != -100).sum().item()
        log_pi_rejected = -out_r.loss * n_rejected_tokens

        # DPO loss for this pair
        pair_loss = -torch.nn.functional.logsigmoid(
            beta * (log_pi_chosen - log_pi_rejected)
        )
        total_loss = total_loss + pair_loss
        n += 1

    return total_loss / n if n > 0 else total_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    output_dir: str,
    config_path: str,
    preference_data_path: str | None = None,
) -> dict:
    """Run LoRA fine-tuning with MuonAdamW.

    If preference_data_path is provided and contains preference pairs,
    training alternates between standard SFT loss (next-token prediction)
    and DPO loss. The DPO loss uses beta=0.1 and runs every
    dpo_every_n_steps steps (configurable in growth_config.yaml).

    Returns a result dict printed as JSON to stdout by the CLI wrapper.
    """
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    try:
        from spark.growth.muon_adamw import MuonAdamW, build_param_groups
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from muon_adamw import MuonAdamW, build_param_groups

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lora_cfg = cfg.get("lora", {})
    ewc_cfg = cfg.get("ewc", {})
    dpo_cfg = cfg.get("dpo", {})

    rank = lora_cfg.get("fast_rank", 8)
    alpha = lora_cfg.get("alpha", 16)
    target_modules = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    lr = lora_cfg.get("fast_lr", 2e-4)
    time_budget_seconds = lora_cfg.get("time_budget_seconds", 7200)
    gc_collect_every = lora_cfg.get("gc_collect_every", 5000)
    epochs = lora_cfg.get("epochs", 2)

    dpo_beta = dpo_cfg.get("beta", 0.1)
    dpo_every_n_steps = dpo_cfg.get("every_n_steps", 10)
    dpo_weight = dpo_cfg.get("loss_weight", 0.3)

    examples = load_chat_jsonl(data_path)
    if not examples:
        raise RuntimeError(f"No training examples found in {data_path}")

    # Load preference pairs if available
    preference_pairs: list[dict] = []
    if preference_data_path:
        preference_pairs = load_preference_jsonl(preference_data_path)
        print(
            f"[peft_train] {len(preference_pairs)} preference pairs loaded",
            file=sys.stderr,
        )
    else:
        print("[peft_train] no preference data — SFT only", file=sys.stderr)

    theta = compute_encounter_phase(examples)
    print(f"[peft_train] {len(examples)} training examples loaded", file=sys.stderr)
    print(f"[peft_train] θ = {theta['theta_radians']:.4f} rad", file=sys.stderr)

    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/"
        "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    )
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

    param_groups = build_param_groups(
        model, muon_lr=lr, adamw_lr=lr, weight_decay=0.2,
    )
    optimizer = MuonAdamW(param_groups)

    max_seq_len = 512
    all_input_ids = []
    all_labels = []

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

    dataset_input_ids = torch.stack(all_input_ids)
    dataset_labels = torch.stack(all_labels)
    n_examples = len(all_input_ids)

    batch_size = min(4, n_examples)
    budget = TimeBudget(budget_seconds=time_budget_seconds, warmup_steps=1)
    model.train()

    steps_trained = 0
    running_loss = 0.0
    final_loss = float("inf")
    dpo_steps = 0
    dpo_loss_total = 0.0

    mode_str = (
        f"SFT + DPO (beta={dpo_beta}, weight={dpo_weight}, every {dpo_every_n_steps} steps)"
        if preference_pairs
        else "SFT only"
    )
    print(
        f"[peft_train] Training: {epochs} epochs, batch_size={batch_size}, "
        f"budget={time_budget_seconds}s, mode={mode_str}",
        file=sys.stderr,
    )

    with gc_discipline(collect_every_n_steps=gc_collect_every):
        for epoch in range(epochs):
            if budget.exhausted:
                print(f"[peft_train] Time budget exhausted at epoch {epoch}", file=sys.stderr)
                break

            indices = torch.randperm(n_examples)

            for batch_start in range(0, n_examples, batch_size):
                if budget.exhausted:
                    break

                t0 = time.monotonic()

                batch_idx = indices[batch_start : batch_start + batch_size]
                input_ids = dataset_input_ids[batch_idx].to(model.device)
                labels = dataset_labels[batch_idx].to(model.device)

                # SFT loss
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                # DPO loss (interleaved every dpo_every_n_steps)
                if (
                    preference_pairs
                    and steps_trained > 0
                    and steps_trained % dpo_every_n_steps == 0
                ):
                    # Sample a small batch of preference pairs
                    import random
                    dpo_batch = random.sample(
                        preference_pairs, min(4, len(preference_pairs))
                    )
                    d_loss = dpo_loss(
                        model, tokenizer, dpo_batch,
                        beta=dpo_beta, max_seq_len=max_seq_len,
                    )
                    loss = (1 - dpo_weight) * loss + dpo_weight * d_loss
                    dpo_loss_total += d_loss.item()
                    dpo_steps += 1

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
                        f"elapsed={budget.elapsed:.0f}s/{time_budget_seconds}s"
                        + (f" dpo_steps={dpo_steps}" if preference_pairs else ""),
                        file=sys.stderr,
                    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_path / "adapter"
    model.save_pretrained(adapter_dir)

    adapter_file = None
    for f in adapter_dir.rglob("*.safetensors"):
        adapter_file = str(f)
        break
    if adapter_file is None:
        adapter_file = str(adapter_dir / "adapter_model.safetensors")

    print(f"[peft_train] Adapter saved to {adapter_file}", file=sys.stderr)
    print(f"[peft_train] Final loss: {final_loss:.4f}, steps: {steps_trained}", file=sys.stderr)
    if dpo_steps > 0:
        print(
            f"[peft_train] DPO steps: {dpo_steps}, mean DPO loss: "
            f"{dpo_loss_total / dpo_steps:.4f}",
            file=sys.stderr,
        )

    result = {
        "final_loss": round(final_loss, 6),
        "steps_trained": steps_trained,
        "adapter_path": adapter_file,
        "n_examples": n_examples,
        "epochs_completed": epoch + 1 if steps_trained > 0 else 0,
        "theta": theta,
        "elapsed_seconds": round(budget.elapsed, 1),
        "dpo_steps": dpo_steps,
        "n_preference_pairs": len(preference_pairs),
        "mean_dpo_loss": round(dpo_loss_total / dpo_steps, 6) if dpo_steps > 0 else None,
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
    parser.add_argument(
        "--preference-data",
        default=None,
        help="Optional path to DPO preference pairs JSONL",
    )
    args = parser.parse_args()

    result = train(
        data_path=args.data,
        output_dir=args.output_dir,
        config_path=args.config,
        preference_data_path=args.preference_data,
    )

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
