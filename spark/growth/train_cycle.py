"""spark.growth.train_cycle — Training execution for the recursive growth engine.

Phase 5 (DISTILL) of the growth cycle described in issue #2483.

MODEL SITUATION (March 2026):
  Only GGUF models are on disk.  The llama-finetune binary on this
  machine does **full-parameter fine-tuning** on GGUF models — it does
  NOT produce LoRA adapters.  It expects:

    llama-finetune \
      --model <gguf_path>          # -m
      --file  <raw_text_file>      # -f  (plain text, NOT JSONL)
      --output <output_gguf>       # -o  (full finetuned model)
      --n-gpu-layers 999           # offload everything
      --ctx-size <ctx>             # -c
      --epochs <n>
      --learning-rate <lr>
      --batch-size <bs>            # -b
      --optimizer adamw

  Key constraints:
    - The README states "for FP32 models and limited hardware setups".
    - Full fine-tuning of a 120B IQ4_XS model (~65 GB) requires
      optimizer states and gradients that will NOT fit in 128 GB.
    - Until either (a) LoRA support lands in llama.cpp's finetune, or
      (b) we have an FP32/FP16 small model on disk, training will
      fail at the pre-flight check with a clear diagnostic.

  The training data pipeline (DeltaPackage → JSONL → raw text) is kept
  working so the moment we have a viable model, the loop closes.

Integration:
  - Input:  DeltaPackage from DeltaExtractor.extract()
  - Output: finetuned model at GROWTH_DIR/adapters/<cycle_id>/finetuned.gguf
            (or adapter.gguf if LoRA support is added later)
  - Cycle history: GROWTH_DIR/cycle_history.jsonl
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
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

# llama-finetune binary locations, in order of preference
_LLAMA_FINETUNE_CANDIDATES = [
    Path.home() / "llama.cpp" / "build" / "bin" / "llama-finetune",
    Path("/usr/local/bin/llama-finetune"),
    Path("/usr/bin/llama-finetune"),
]

# GGUF base model candidates
_GGUF_CANDIDATES = [
    # Actual model on disk — Nemotron-3-Super-120B, IQ4_XS, split GGUF
    Path.home() / "models" / "Nemotron-3-Super-120B-GGUF"
        / "nvidia_Nemotron-3-Super-120B-A12B-IQ4_XS"
        / "nvidia_Nemotron-3-Super-120B-A12B-IQ4_XS-00001-of-00002.gguf",
    Path.home() / "models" / "nemotron"
        / "Nemotron-Super-512B-v1.Q4_K_M.gguf",
    Path("/models/nemotron/Nemotron-Super-512B-v1.Q4_K_M.gguf"),
]

# Models larger than this (in bytes) are too big for full fine-tuning
# in 128 GB unified memory (need ~3× model size for optimizer + grads).
_MAX_FINETUNE_SIZE_BYTES = 20 * 1024**3  # 20 GB — safe for full fine-tuning

# Quantization types that are NOT suitable for gradient computation.
# llama-finetune README: "for FP32 models and limited hardware setups"
_BLOCKED_QUANT_PATTERNS = {"IQ4", "IQ3", "IQ2", "IQ1", "Q4_K", "Q3_K",
                           "Q2_K", "Q5_K", "Q6_K", "Q8_0", "Q4_0", "Q4_1"}


def _find_llama_finetune() -> Optional[Path]:
    for p in _LLAMA_FINETUNE_CANDIDATES:
        if p.exists() and os.access(p, os.X_OK):
            return p
    found = shutil.which("llama-finetune")
    return Path(found) if found else None


def _find_gguf() -> Optional[Path]:
    for p in _GGUF_CANDIDATES:
        if p.exists():
            return p
    models_dir = Path.home() / "models"
    if models_dir.exists():
        for p in sorted(models_dir.rglob("*.gguf")):
            return p
    return None


def _check_model_viability(gguf_path: Path) -> Optional[str]:
    """Pre-flight check: can this model be fine-tuned?

    Returns None if viable, or a human-readable error string if not.
    """
    # --- Size check ---
    if gguf_path.stat().st_size > _MAX_FINETUNE_SIZE_BYTES:
        # Check for split GGUFs (sum all parts)
        parent = gguf_path.parent
        stem = gguf_path.stem
        # Pattern: name-00001-of-00004.gguf
        parts = list(parent.glob(
            stem.rsplit("-00001", 1)[0] + "-*.gguf"
        )) if "-00001" in stem else [gguf_path]
        total_size = sum(p.stat().st_size for p in parts)
        if total_size > _MAX_FINETUNE_SIZE_BYTES:
            size_gb = total_size / (1024**3)
            return (
                f"Model too large for full fine-tuning: {size_gb:.1f} GB "
                f"(limit: {_MAX_FINETUNE_SIZE_BYTES / (1024**3):.0f} GB). "
                f"Full fine-tuning needs ~3× model size for optimizer states. "
                f"Options: (a) download a small FP32 model (≤8B params), "
                f"(b) wait for LoRA support in llama.cpp finetune, or "
                f"(c) use PEFT+TRL in the vllm_node container with an HF-format model."
            )

    # --- Quantization check ---
    name_upper = gguf_path.name.upper()
    for quant in _BLOCKED_QUANT_PATTERNS:
        if quant in name_upper:
            return (
                f"Model uses {quant} quantization, which is not suitable "
                f"for gradient-based fine-tuning. llama-finetune requires "
                f"FP32 (or possibly FP16) models. "
                f"Options: (a) download an FP32/FP16 small model, "
                f"(b) use PEFT+TRL in the container with an HF-format model."
            )

    return None  # Model looks viable


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


class TrainCycle:
    """Executes a single growth cycle's training phase using llama-finetune.

    Current state (March 2026):
      llama-finetune does full-parameter fine-tuning on GGUF models.
      It requires FP32 models and outputs a complete fine-tuned GGUF.
      It does NOT support LoRA adapter creation.

      With only IQ4_XS quantized models on disk, the pre-flight check
      will block training with a clear diagnostic. The data pipeline
      (delta extraction → text conversion) is kept working so that
      when a viable model is available, the loop closes immediately.
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

    def run(self, delta: DeltaPackage, dry_run: bool = False) -> TrainResult:
        """Execute the training phase.

        1. Find llama-finetune binary and GGUF base model
        2. Pre-flight check: model size and quantization
        3. Convert DeltaPackage to raw text
        4. Run llama-finetune (or report why it can't run)
        5. Return TrainResult

        In dry_run mode, prepares data and prints the command but
        does not execute training.
        """
        cycle_id  = delta.cycle_id
        cycle_dir = ADAPTERS_DIR / cycle_id
        cycle_dir.mkdir(parents=True, exist_ok=True)

        binary = _find_llama_finetune()
        gguf   = _find_gguf()

        if not binary:
            msg = (
                "llama-finetune not found. "
                f"Checked: {[str(p) for p in _LLAMA_FINETUNE_CANDIDATES]}. "
                "Build llama.cpp with CUDA: "
                "cd ~/llama.cpp && cmake -B build -DGGML_CUDA=ON && "
                "cmake --build build -j --config Release"
            )
            raise RuntimeError(msg)

        if not gguf:
            msg = (
                "No GGUF base model found. "
                f"Checked: {[str(p) for p in _GGUF_CANDIDATES]}."
            )
            raise RuntimeError(msg)

        # --- Pre-flight: can this model be fine-tuned? ---
        viability_err = _check_model_viability(gguf)
        if viability_err:
            # Still convert the data so it's ready when a viable model arrives
            raw_path  = cycle_dir / "training_data.txt"
            jsonl_path = cycle_dir / "training_data_llama.jsonl"
            n_raw  = _convert_to_raw_text(delta, raw_path)
            n_jsonl = _convert_to_llama_jsonl(delta, jsonl_path)
            print(f"[TrainCycle] Data prepared: {n_raw} conversations → "
                  f"{raw_path.name}, {n_jsonl} examples → {jsonl_path.name}")
            print(f"[TrainCycle] BLOCKED: {viability_err}")

            result = TrainResult(
                cycle_id=cycle_id,
                adapter_path=cycle_dir / "finetuned.gguf",
                final_loss=-1.0,
                steps_trained=0,
                delta_count=delta.delta_count,
                replay_count=delta.replay_count,
                ewc_lambda_used=self._ewc_cfg.get("lambda", 1e4),
                metadata={
                    "blocked": True,
                    "reason": viability_err,
                    "data_ready": True,
                    "raw_text_path": str(raw_path),
                    "jsonl_path": str(jsonl_path),
                    "n_conversations": n_raw,
                    "n_jsonl_examples": n_jsonl,
                },
            )
            self._record_cycle(result)
            return result

        # --- Convert data ---
        raw_path = cycle_dir / "training_data.txt"
        n_examples = _convert_to_raw_text(delta, raw_path)
        if n_examples == 0:
            raise RuntimeError("No valid training examples after conversion")

        # Also save JSONL for future use / debugging
        jsonl_path = cycle_dir / "training_data_llama.jsonl"
        _convert_to_llama_jsonl(delta, jsonl_path)

        output_path = cycle_dir / "finetuned.gguf"

        # llama-finetune parameters (correct flags for current binary)
        epochs = self._lora_cfg.get("epochs", 2)
        ctx    = 512   # Keep small to save memory during training
        lr     = 1e-5
        batch  = 512

        cmd = [
            str(binary),
            "--model",          str(gguf),        # -m
            "--file",           str(raw_path),    # -f  (raw text)
            "--output",         str(output_path), # -o
            "--n-gpu-layers",   "999",
            "--ctx-size",       str(ctx),         # -c
            "--epochs",         str(epochs),
            "--learning-rate",  str(lr),          # -lr
            "--batch-size",     str(batch),       # -b
            "--ubatch-size",    str(batch),       # -ub
            "--optimizer",      "adamw",
            "--threads",        "8",
        ]

        print(f"[TrainCycle] binary:  {binary}")
        print(f"[TrainCycle] model:   {gguf}")
        print(f"[TrainCycle] data:    {raw_path} ({n_examples} conversations)")
        print(f"[TrainCycle] output:  {output_path}")
        print(f"[TrainCycle] command: {' '.join(cmd)}")

        if dry_run:
            return TrainResult(
                cycle_id=cycle_id,
                adapter_path=output_path,
                final_loss=0.0,
                steps_trained=0,
                delta_count=delta.delta_count,
                replay_count=delta.replay_count,
                ewc_lambda_used=self._ewc_cfg.get("lambda", 1e4),
                metadata={"dry_run": True, "cmd": cmd},
            )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour max
        )

        stdout_tail = result.stdout[-2000:] if result.stdout else ""
        stderr_tail = result.stderr[-2000:] if result.stderr else ""

        if result.returncode != 0:
            raise RuntimeError(
                f"llama-finetune failed (exit {result.returncode}):\n"
                f"{stderr_tail}"
            )

        print(f"[TrainCycle] stdout tail:\n{stdout_tail}")

        # Parse loss from output
        losses = re.findall(r"loss\s*=\s*([0-9.]+)", stdout_tail + stderr_tail)
        final_loss = float(losses[-1]) if losses else -1.0

        train_result = TrainResult(
            cycle_id=cycle_id,
            adapter_path=output_path,
            final_loss=final_loss,
            steps_trained=epochs,
            delta_count=delta.delta_count,
            replay_count=delta.replay_count,
            ewc_lambda_used=self._ewc_cfg.get("lambda", 1e4),
            metadata={
                "binary":     str(binary),
                "gguf":       str(gguf),
                "examples":   n_examples,
                "epochs":     epochs,
                "ctx":        ctx,
                "lr":         lr,
            },
        )

        self._record_cycle(train_result)
        return train_result

    def _find_prev_adapter(self) -> Optional[Path]:
        """Find the most recent fine-tuned output (for continued training)."""
        if not ADAPTERS_DIR.exists():
            return None
        dirs = sorted(
            [d for d in ADAPTERS_DIR.iterdir()
             if d.is_dir() and (
                 (d / "finetuned.gguf").exists() or
                 (d / "adapter.gguf").exists()
             )],
            key=lambda d: d.name,
        )
        if not dirs:
            return None
        last = dirs[-1]
        for name in ("finetuned.gguf", "adapter.gguf"):
            p = last / name
            if p.exists():
                return p
        return None

    def _record_cycle(self, result: TrainResult) -> None:
        CYCLE_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **result.to_dict(),
        }
        with open(CYCLE_HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
