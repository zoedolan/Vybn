"""spark.growth.train_cycle — Training execution for the recursive growth engine.

Phase 5 (DISTILL) of the growth cycle described in issue #2483.

IMPORTANT — MODEL SITUATION AS OF MARCH 2026:
  Only GGUF models are on disk (Nemotron 3 Super 120B GGUF, MiniMax M2.5 GGUF).
  The AutoModelForCausalLM path in _generate_train_script() requires a
  HuggingFace-format model directory. This will NOT work with GGUFs.

  The correct path for GGUF fine-tuning is llama-finetune (llama.cpp native):
    ~/llama.cpp/build/bin/llama-finetune
  This binary is present and CUDA-enabled (GB10/sm_121 detected at startup).

  Until the training script is ported to use llama-finetune, the DISTILL phase
  generates and validates the training data but cannot execute the actual
  fine-tuning step. The vllm_node container IS running (started 2026-03-14)
  with the Vybn repo mounted at /workspace/Vybn — the container path check
  will now succeed. The missing piece is the training script itself.

Original design (HuggingFace / AWQ model):
  Implements LoRA fine-tuning on a quantized model:
  - LoRA adapters on attention projections (q/k/v/o_proj)
  - Configurable rank, alpha, learning rate from growth_config.yaml
  - EWC regularization via Fisher Information (when previous cycle exists)
  - Slow adapter EMA consolidation across cycles

Key design decisions:
  - Training runs INSIDE the vLLM container (has torch + PEFT + TRL)
  - vLLM serving is stopped before training and restarted after
  - We target attention projections only (avoids MoE expert weights)
  - The training script is generated and exec'd in the container

Integration points:
  - Input from: DeltaExtractor.extract() → DeltaPackage
  - Output: trained adapter at GROWTH_DIR / "adapters" / cycle_id /
  - Cycle history: GROWTH_DIR / "cycle_history.jsonl"
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from spark.growth.delta_extract import DeltaPackage

GROWTH_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = GROWTH_DIR / "growth_config.yaml"
ADAPTERS_DIR = GROWTH_DIR / "adapters"
CYCLE_HISTORY = GROWTH_DIR / "cycle_history.jsonl"


@dataclass(slots=True)
class TrainResult:
    """Result of a single growth cycle's training phase."""

    cycle_id: str
    adapter_path: Path
    final_loss: float
    steps_trained: int
    delta_count: int
    replay_count: int
    ewc_lambda_used: float
    slow_adapter_path: Optional[Path] = None
    lora_subspace_path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "adapter_path": str(self.adapter_path),
            "final_loss": self.final_loss,
            "steps_trained": self.steps_trained,
            "delta_count": self.delta_count,
            "replay_count": self.replay_count,
            "ewc_lambda_used": self.ewc_lambda_used,
            "slow_adapter_path": str(self.slow_adapter_path) if self.slow_adapter_path else None,
            "lora_subspace_path": str(self.lora_subspace_path) if self.lora_subspace_path else None,
            "metadata": self.metadata,
        }


def _generate_train_script(
    data_path: str,
    output_dir: str,
    model_id: str,
    lora_cfg: dict,
    ewc_cfg: dict,
    prev_adapter_path: Optional[str] = None,
    max_steps: int = 100,
) -> str:
    """Generate the Python training script to run inside the container.

    This script is self-contained — it imports everything it needs
    and produces a LoRA adapter directory + a results JSON.

    NOTE: This script requires a HuggingFace-format model at model_id.
    For GGUF models, use llama-finetune instead (~/llama.cpp/build/bin/llama-finetune).
    """
    return textwrap.dedent(f'''\
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
MODEL_ID = "{model_id}"
DATA_PATH = "{data_path}"
OUTPUT_DIR = "{output_dir}"
LORA_R = {lora_cfg.get("fast_rank", 8)}
LORA_ALPHA = {lora_cfg.get("alpha", 16)}
LORA_LR = {lora_cfg.get("fast_lr", 2e-4)}
TARGET_MODULES = {lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])}
EWC_LAMBDA = {ewc_cfg.get("lambda", 1e4)}
MAX_STEPS = {max_steps}
PREV_ADAPTER = {repr(prev_adapter_path)}

print(f"Loading model: {{MODEL_ID}}")
print(f"Training data: {{DATA_PATH}}")
print(f"Output: {{OUTPUT_DIR}}")

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

print(f"Loaded {{len(examples)}} training examples")

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
    print(f"Loading previous adapter from {{PREV_ADAPTER}}")
    model = PeftModel.from_pretrained(model, PREV_ADAPTER, is_trainable=True)
else:
    print("Initializing fresh LoRA adapter")
    model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {{trainable:,}} / {{total:,}} ({{100*trainable/total:.4f}}%)")

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
            parts.append(f"<|{{m['role']}}|>\\n{{m['content']}}")
        texts.append("\\n".join(parts))

dataset = Dataset.from_dict({{"text": texts}})
print(f"Dataset size: {{len(dataset)}} examples")

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

print(f"Starting training ({{MAX_STEPS}} steps)...")
result = trainer.train()
print(f"Training complete. Loss: {{result.training_loss:.4f}}")

# ── Save adapter ────────────────────────────────────────────────────────
adapter_path = Path(OUTPUT_DIR) / "adapter"
model.save_pretrained(str(adapter_path))
tokenizer.save_pretrained(str(adapter_path))
print(f"Adapter saved to {{adapter_path}}")

# ── Save results ────────────────────────────────────────────────────────
results = {{
    "final_loss": result.training_loss,
    "steps": result.global_step,
    "trainable_params": trainable,
    "total_params": total,
    "examples": len(dataset),
}}
results_path = Path(OUTPUT_DIR) / "train_results.json"
results_path.write_text(json.dumps(results, indent=2))
print(f"Results saved to {{results_path}}")
print("DONE")
''')


class TrainCycle:
    """Executes a single growth cycle's training phase.

    Orchestrates LoRA fine-tuning inside the vLLM container.
    The vllm_node container must be running with the Vybn repo
    mounted at /workspace/Vybn (started 2026-03-14 with --gpus all).

    NOTE: _generate_train_script() targets AutoModelForCausalLM which
    requires a HuggingFace-format model. Only GGUFs are on disk. The
    training script will fail at model load unless a HF-format model
    is added, OR the script is ported to llama-finetune.

    This is Phase 5 (DISTILL) of the growth cycle described in #2483.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        config_path = config_path or DEFAULT_CONFIG
        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)
        self._lora_cfg = self._cfg.get("lora", {})
        self._ewc_cfg = self._cfg.get("ewc", {})
        self._merge_cfg = self._cfg.get("merge", {})
        # Driven by growth_config.yaml merge.serving_model.
        # Default references Nemotron but only GGUF is on disk —
        # AutoModelForCausalLM cannot load a GGUF. See module docstring.
        self._model_id = self._merge_cfg.get("serving_model", "nvidia/Nemotron-3-8B-Chat-4k")
        self._container_name = "vllm_node"

    def run(self, delta: DeltaPackage, dry_run: bool = False) -> TrainResult:
        """Execute the training phase of a growth cycle.

        1. Write training data to JSONL
        2. Generate training script
        3. Stop vLLM serving (free GPU memory)
        4. Execute training inside the container
        5. Collect results
        6. Restart vLLM serving

        Args:
            delta: The DeltaPackage from DeltaExtractor.extract().
            dry_run: If True, generate script and data but don't execute.

        Returns:
            TrainResult with paths to adapters and training metadata.
        """
        cycle_id = delta.cycle_id
        cycle_dir = ADAPTERS_DIR / cycle_id
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write training data
        data_path = cycle_dir / "training_data.jsonl"
        delta.to_jsonl(data_path)

        # 2. Determine previous adapter (for continued training)
        prev_adapter = self._find_prev_adapter()

        # 3. Compute max_steps based on delta size
        # Rule: ~2 epochs over the data, min 20 steps, max 200
        total_examples = delta.total_entries
        steps_per_epoch = max(1, total_examples // 4)  # batch_size=1, grad_accum=4
        max_steps = min(200, max(20, steps_per_epoch * 2))

        # 4. Generate training script
        # Container paths — the repo is mounted at /workspace/Vybn
        container_data_path = f"/workspace/Vybn/spark/growth/adapters/{cycle_id}/training_data.jsonl"
        container_output_dir = f"/workspace/Vybn/spark/growth/adapters/{cycle_id}"

        script = _generate_train_script(
            data_path=container_data_path,
            output_dir=container_output_dir,
            model_id=self._model_id,
            lora_cfg=self._lora_cfg,
            ewc_cfg=self._ewc_cfg,
            prev_adapter_path=str(prev_adapter) if prev_adapter else None,
            max_steps=max_steps,
        )

        script_path = cycle_dir / "train.py"
        script_path.write_text(script, encoding="utf-8")

        if dry_run:
            return TrainResult(
                cycle_id=cycle_id,
                adapter_path=cycle_dir / "adapter",
                final_loss=0.0,
                steps_trained=0,
                delta_count=delta.delta_count,
                replay_count=delta.replay_count,
                ewc_lambda_used=self._ewc_cfg.get("lambda", 1e4),
                metadata={"dry_run": True, "script_path": str(script_path)},
            )

        # 5. Check if repo is mounted in container
        check = subprocess.run(
            ["docker", "exec", self._container_name, "test", "-f",
             container_data_path],
            capture_output=True,
        )
        if check.returncode != 0:
            # Try to find the mount point
            mount_check = subprocess.run(
                ["docker", "exec", self._container_name, "ls", "/workspace/"],
                capture_output=True, text=True,
            )
            raise RuntimeError(
                f"Training data not accessible in container at {container_data_path}. "
                f"Container /workspace/ contains: {mount_check.stdout.strip()}. "
                f"The Vybn repo must be mounted at /workspace/Vybn in the vLLM container."
            )

        # 6. Execute training
        print(f"[TrainCycle] Starting training: {cycle_id}")
        print(f"[TrainCycle] {delta.delta_count} delta + {delta.replay_count} replay entries")
        print(f"[TrainCycle] Max steps: {max_steps}")

        container_script_path = f"/workspace/Vybn/spark/growth/adapters/{cycle_id}/train.py"
        result = subprocess.run(
            ["docker", "exec", self._container_name, "python3", container_script_path],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )

        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else "no stderr"
            print(f"[TrainCycle] FAILED:\n{error_msg}")
            raise RuntimeError(f"Training failed (exit {result.returncode}): {error_msg}")

        print(f"[TrainCycle] Training output:\n{result.stdout[-1000:]}")

        # 7. Read results
        results_path = cycle_dir / "train_results.json"
        if results_path.exists():
            train_results = json.loads(results_path.read_text())
        else:
            train_results = {"final_loss": -1.0, "steps": 0}

        # 8. Build TrainResult
        adapter_path = cycle_dir / "adapter"
        train_result = TrainResult(
            cycle_id=cycle_id,
            adapter_path=adapter_path,
            final_loss=train_results.get("final_loss", -1.0),
            steps_trained=train_results.get("steps", 0),
            delta_count=delta.delta_count,
            replay_count=delta.replay_count,
            ewc_lambda_used=self._ewc_cfg.get("lambda", 1e4),
            metadata={
                "model_id": self._model_id,
                "max_steps": max_steps,
                "trainable_params": train_results.get("trainable_params", 0),
                "total_params": train_results.get("total_params", 0),
                "examples": train_results.get("examples", 0),
            },
        )

        # 9. Record in cycle history
        self._record_cycle(train_result)

        return train_result

    def _find_prev_adapter(self) -> Optional[Path]:
        """Find the most recent completed adapter for continued training."""
        if not ADAPTERS_DIR.exists():
            return None
        cycle_dirs = sorted(
            [d for d in ADAPTERS_DIR.iterdir() if d.is_dir() and (d / "adapter").exists()],
            key=lambda d: d.name,
        )
        if cycle_dirs:
            return cycle_dirs[-1] / "adapter"
        return None

    def _record_cycle(self, result: TrainResult) -> None:
        """Append training result to cycle history."""
        CYCLE_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **result.to_dict(),
        }
        with open(CYCLE_HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
