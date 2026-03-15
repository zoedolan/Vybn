"""spark.growth.train_cycle — Training execution for the recursive growth engine.

Executes M′ = α·M + x·e^(iθ) — LoRA adapter (α) trained on phase-rotated
delta (x·e^(iθ)) via PEFT/TRL with MuonAdamW.

Phase 5 (DISTILL) of the growth cycle described in issue #2483.

The conjecture from PR #2572:
  - M   = current model (Nemotron 3 Super 120B-A12B, NVFP4 safetensors)
  - α   = structure-preserving LoRA adapter, trained with MuonAdamW whose
          polar express orthogonalisation preserves the base model's core
          structure while enabling adaptation
  - x·e^(iθ) = training delta (DeltaPackage), phase-rotated by encounter
          angle θ encoding temporal/contextual orientation of the data
  - M′  = transformed model after adapter application

Training runs inside the vllm_node container via:
    docker exec vllm_node python3 /workspace/Vybn/spark/growth/peft_train.py \\
        --data <jsonl_path> --output-dir <dir> --config <yaml_path>

The existing growth pipeline (trigger.py, delta_extract.py, merge_cycle.py,
eval_harness.py) is untouched — only train_cycle.py was rewritten, and two
new files were added (muon_adamw.py, peft_train.py).

Integration:
  - Input:  DeltaPackage from DeltaExtractor.extract()
  - Output: LoRA adapter at GROWTH_DIR/adapters/<cycle_id>/adapter/
  - Cycle history: GROWTH_DIR/cycle_history.jsonl
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from spark.growth.delta_extract import DeltaPackage

GROWTH_DIR     = Path(__file__).resolve().parent
DEFAULT_CONFIG = GROWTH_DIR / "growth_config.yaml"
ADAPTERS_DIR   = GROWTH_DIR / "adapters"
CYCLE_HISTORY  = GROWTH_DIR / "cycle_history.jsonl"

# Path to peft_train.py as seen from inside the vllm_node container
_CONTAINER_SCRIPT = "/workspace/Vybn/spark/growth/peft_train.py"
_CONTAINER_CONFIG = "/workspace/Vybn/spark/growth/growth_config.yaml"


@dataclass(slots=True)
class TrainResult:
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
            "cycle_id":          self.cycle_id,
            "adapter_path":      str(self.adapter_path),
            "final_loss":        self.final_loss,
            "steps_trained":     self.steps_trained,
            "delta_count":       self.delta_count,
            "replay_count":      self.replay_count,
            "ewc_lambda_used":   self.ewc_lambda_used,
            "slow_adapter_path": str(self.slow_adapter_path) if self.slow_adapter_path else None,
            "lora_subspace_path": str(self.lora_subspace_path) if self.lora_subspace_path else None,
            "metadata":          self.metadata,
        }


def _convert_to_raw_text(delta: DeltaPackage, out_path: Path) -> int:
    """Convert DeltaPackage to plain text for llama-finetune.

    llama-finetune expects raw text (not JSONL). We concatenate all
    conversation turns into a single text file, separated by newlines.
    The model learns next-token prediction on this text.

    Returns count of conversations processed.
    """
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for entry in delta.all_entries:
            msgs = entry.get("messages", [])
            if not msgs:
                continue

            # Build readable text from the conversation
            parts = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                if content.strip():
                    parts.append(f"<|{role}|>\n{content}")

            if parts:
                fh.write("\n".join(parts))
                fh.write("\n\n")  # Double newline between conversations
                written += 1

    return written


def _convert_to_llama_jsonl(delta: DeltaPackage, out_path: Path) -> int:
    """Convert DeltaPackage to JSONL format (kept for future use with
    tools that accept structured training data).

    Returns count of written examples.
    """
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for entry in delta.all_entries:
            msgs = entry.get("messages", [])
            if not msgs:
                continue

            assistant_turns = [m for m in msgs if m.get("role") == "assistant"]
            if not assistant_turns:
                continue

            last_assistant = assistant_turns[-1]["content"]
            input_turns = [m for m in msgs if m is not assistant_turns[-1]]

            prompt_parts = []
            for m in input_turns:
                role = m.get("role", "user")
                content = m.get("content", "")
                prompt_parts.append(f"<|{role}|>\n{content}")
            prompt = "\n".join(prompt_parts)

            record = {"input": prompt, "output": last_assistant}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return written


def _convert_to_chat_jsonl(delta: DeltaPackage, out_path: Path) -> int:
    """Convert DeltaPackage to chat-format JSONL for peft_train.py.

    Each line: {"messages": [...], "metadata": {...}}
    This is the native format from DeltaPackage.to_jsonl().

    Returns count of written examples.
    """
    return delta.to_jsonl(out_path)


class TrainCycle:
    """Executes M′ = α·M + x·e^(iθ) — LoRA adapter (α) trained on
    phase-rotated delta (x·e^(iθ)) via PEFT/TRL with MuonAdamW.

    Replaces the previous llama-finetune approach (which was always
    BLOCKED on the 65GB IQ4_XS model) with actual LoRA training via
    peft_train.py running inside the vllm_node container.

    The training data pipeline (DeltaPackage → JSONL) feeds into
    peft_train.py which:
    1. Loads the NVFP4 base model with 4-bit quantisation
    2. Applies LoRA (rank=8, alpha=16) to attention projections
    3. Trains with MuonAdamW (Muon for 2D LoRA matrices, AdamW for rest)
    4. Saves the adapter as .safetensors
    """

    def __init__(self, config_path: Path | None = None) -> None:
        config_path = config_path or DEFAULT_CONFIG
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}
        self._lora_cfg = cfg.get("lora", {})
        self._ewc_cfg  = cfg.get("ewc", {})
        self._eval_cfg = cfg.get("eval", {})
        self._time_budget_seconds: int = self._lora_cfg.get(
            "time_budget_seconds", 7200,
        )

    def run(self, delta: DeltaPackage, dry_run: bool = False) -> TrainResult:
        """Execute the training phase.

        1. Convert DeltaPackage to chat-format JSONL
        2. Shell out to peft_train.py inside vllm_node container
        3. Parse JSON result from stdout
        4. Run BPB eval via eval_harness.py
        5. Return TrainResult with adapter_path pointing to .safetensors

        In dry_run mode, prepares data and prints the command but
        does not execute training.
        """
        cycle_id  = delta.cycle_id
        cycle_dir = ADAPTERS_DIR / cycle_id
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # --- Convert data to chat-format JSONL (native format for peft_train.py) ---
        jsonl_path = cycle_dir / "training_data.jsonl"
        n_examples = _convert_to_chat_jsonl(delta, jsonl_path)
        if n_examples == 0:
            raise RuntimeError("No valid training examples after conversion")

        # Also save raw text and legacy JSONL for eval and debugging
        raw_path = cycle_dir / "training_data.txt"
        _convert_to_raw_text(delta, raw_path)
        legacy_jsonl = cycle_dir / "training_data_llama.jsonl"
        _convert_to_llama_jsonl(delta, legacy_jsonl)

        # Container paths (repo mounted at /workspace/Vybn)
        container_data = f"/workspace/Vybn/spark/growth/adapters/{cycle_id}/training_data.jsonl"
        container_output = f"/workspace/Vybn/spark/growth/adapters/{cycle_id}"

        cmd = [
            "docker", "exec", "vllm_node",
            "python3", _CONTAINER_SCRIPT,
            "--data", container_data,
            "--output-dir", container_output,
            "--config", _CONTAINER_CONFIG,
        ]

        print(f"[TrainCycle] cycle:    {cycle_id}")
        print(f"[TrainCycle] data:     {jsonl_path} ({n_examples} examples)")
        print(f"[TrainCycle] output:   {cycle_dir}")
        print(f"[TrainCycle] command:  {' '.join(cmd)}")

        adapter_path = cycle_dir / "adapter" / "adapter_model.safetensors"

        if dry_run:
            return TrainResult(
                cycle_id=cycle_id,
                adapter_path=adapter_path,
                final_loss=0.0,
                steps_trained=0,
                delta_count=delta.delta_count,
                replay_count=delta.replay_count,
                ewc_lambda_used=self._ewc_cfg.get("lambda", 1e4),
                metadata={"dry_run": True, "cmd": cmd},
            )

        # --- Execute training inside the container ---
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self._time_budget_seconds,
        )

        stdout_text = result.stdout.strip() if result.stdout else ""
        stderr_tail = result.stderr[-2000:] if result.stderr else ""

        if result.returncode != 0:
            raise RuntimeError(
                f"peft_train.py failed (exit {result.returncode}):\n"
                f"{stderr_tail}"
            )

        # Print stderr (progress logs) for visibility
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-20:]:
                print(f"[TrainCycle] {line}")

        # --- Parse JSON result from stdout ---
        # peft_train.py prints exactly one JSON line to stdout
        train_output = {}
        for line in stdout_text.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    train_output = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        final_loss = train_output.get("final_loss", -1.0)
        steps_trained = train_output.get("steps_trained", 0)
        reported_adapter = train_output.get("adapter_path", str(adapter_path))
        theta = train_output.get("theta", {})

        # Use the reported adapter path if it exists on disk
        if Path(reported_adapter).exists():
            adapter_path = Path(reported_adapter)

        print(f"[TrainCycle] final_loss:    {final_loss}")
        print(f"[TrainCycle] steps_trained: {steps_trained}")
        print(f"[TrainCycle] adapter:       {adapter_path}")
        print(f"[TrainCycle] θ:             {theta.get('theta_radians', 'N/A')} rad")

        train_result = TrainResult(
            cycle_id=cycle_id,
            adapter_path=adapter_path,
            final_loss=final_loss,
            steps_trained=steps_trained,
            delta_count=delta.delta_count,
            replay_count=delta.replay_count,
            ewc_lambda_used=self._ewc_cfg.get("lambda", 1e4),
            metadata={
                "training_method": "peft_lora_muon_adamw",
                "n_examples":  n_examples,
                "theta":       theta,
                "elapsed_seconds": train_output.get("elapsed_seconds"),
                "train_output": train_output,
            },
        )

        # --- BPB evaluation (tokenizer-invariant metric) ---
        if self._eval_cfg.get("enabled", True):
            try:
                from spark.growth.eval_harness import evaluate_bpb

                eval_text = cycle_dir / "training_data.txt"
                bpb = evaluate_bpb(
                    model_url=os.environ.get(
                        "VYBN_MODEL_URL", "http://127.0.0.1:8000",
                    ),
                    eval_text_path=str(eval_text),
                )
                train_result.metadata["val_bpb"] = bpb
                print(f"[TrainCycle] val_bpb: {bpb:.6f}")
            except Exception as e:
                print(f"[TrainCycle] BPB eval skipped: {e}")
                train_result.metadata["val_bpb"] = None
                train_result.metadata["bpb_error"] = str(e)

        self._record_cycle(train_result)
        return train_result

    def _record_cycle(self, result: TrainResult) -> None:
        CYCLE_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **result.to_dict(),
        }
        with open(CYCLE_HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
