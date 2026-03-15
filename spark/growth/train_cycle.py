"""spark.growth.train_cycle — Training execution for the recursive growth engine.

Phase 5 (DISTILL) of the growth cycle described in issue #2483.

MODEL SITUATION (March 2026 onwards):
  Only GGUF models are on disk. This module now uses llama-finetune
  (~/llama.cpp/build/bin/llama-finetune) which natively handles GGUF.
  AutoModelForCausalLM is gone — it was always going to fail.

llama-finetune invocation:
  llama-finetune \
    --model-base <gguf_path> \
    --lora-out   <adapter_gguf_out> \
    --train-data <jsonl_path> \
    --n-gpu-layers 999 \
    --epochs 3 \
    --ctx 2048 \
    --lora-r <rank> \
    --lora-alpha <alpha> \
    --adam-iter <steps>

The training JSONL is written in llama.cpp format:
  { "input": "<prompt>", "output": "<completion>" }

The resulting adapter is a GGUF LoRA file that llama-server loads
with --lora <path> alongside the base model.

Integration:
  - Input: DeltaPackage from DeltaExtractor.extract()
  - Output: adapter at GROWTH_DIR/adapters/<cycle_id>/adapter.gguf
  - Cycle history: GROWTH_DIR/cycle_history.jsonl
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from spark.growth.delta_extract import DeltaPackage

GROWTH_DIR    = Path(__file__).resolve().parent
DEFAULT_CONFIG = GROWTH_DIR / "growth_config.yaml"
ADAPTERS_DIR  = GROWTH_DIR / "adapters"
CYCLE_HISTORY = GROWTH_DIR / "cycle_history.jsonl"

# llama-finetune binary locations, in order of preference
_LLAMA_FINETUNE_CANDIDATES = [
    Path.home() / "llama.cpp" / "build" / "bin" / "llama-finetune",
    Path("/usr/local/bin/llama-finetune"),
    Path("/usr/bin/llama-finetune"),
]

# GGUF base model candidates
_GGUF_CANDIDATES = [
    # Actual model on disk — Nemotron-3-Super-120B, IQ4_XS, split GGUF
    Path.home() / "models" / "Nemotron-3-Super-120B-GGUF" / "nvidia_Nemotron-3-Super-120B-A12B-IQ4_XS" / "nvidia_Nemotron-3-Super-120B-A12B-IQ4_XS-00001-of-00002.gguf",
    Path.home() / "models" / "nemotron" / "Nemotron-Super-512B-v1.Q4_K_M.gguf",
    Path("/models/nemotron/Nemotron-Super-512B-v1.Q4_K_M.gguf"),
]


def _find_llama_finetune() -> Optional[Path]:
    for p in _LLAMA_FINETUNE_CANDIDATES:
        if p.exists() and os.access(p, os.X_OK):
            return p
    # Try PATH
    found = shutil.which("llama-finetune")
    return Path(found) if found else None


def _find_gguf() -> Optional[Path]:
    for p in _GGUF_CANDIDATES:
        if p.exists():
            return p
    # Search ~/models recursively for any .gguf
    models_dir = Path.home() / "models"
    if models_dir.exists():
        for p in sorted(models_dir.rglob("*.gguf")):
            return p
    return None


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


def _convert_to_llama_jsonl(delta: DeltaPackage, out_path: Path) -> int:
    """Convert DeltaPackage to llama-finetune JSONL format.

    llama-finetune expects lines of:
        {"input": "<prompt>", "output": "<completion>"}

    We convert from the chat-message format by treating everything
    except the last assistant turn as input, and the last assistant
    turn as output.

    Returns count of written examples.
    """
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for entry in delta.entries:
            msgs = entry.get("messages", [])
            if not msgs:
                continue

            # Separate input (system + user turns) from output (last assistant)
            assistant_turns = [m for m in msgs if m.get("role") == "assistant"]
            if not assistant_turns:
                continue

            last_assistant = assistant_turns[-1]["content"]
            input_turns    = [m for m in msgs if m is not assistant_turns[-1]]

            # Build a simple text prompt from input turns
            prompt_parts = []
            for m in input_turns:
                role    = m.get("role", "user")
                content = m.get("content", "")
                prompt_parts.append(f"<|{role}|>\n{content}")
            prompt = "\n".join(prompt_parts)

            record = {"input": prompt, "output": last_assistant}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return written


class TrainCycle:
    """Executes a single growth cycle’s training phase using llama-finetune.

    llama-finetune operates directly on the GGUF base model and produces
    a GGUF LoRA adapter. No container, no HuggingFace, no internet required.

    The adapter can be loaded by llama-server with:
        --lora <adapter.gguf>
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
        2. Convert DeltaPackage to llama-finetune JSONL format
        3. Run llama-finetune
        4. Return TrainResult pointing at the adapter GGUF

        In dry_run mode, prepares data and prints the command but
        does not execute training.
        """
        cycle_id  = delta.cycle_id
        cycle_dir = ADAPTERS_DIR / cycle_id
        cycle_dir.mkdir(parents=True, exist_ok=True)

        binary  = _find_llama_finetune()
        gguf    = _find_gguf()

        if not binary:
            msg = (
                "llama-finetune not found. "
                f"Checked: {[str(p) for p in _LLAMA_FINETUNE_CANDIDATES]}. "
                "Build llama.cpp with CUDA: cd ~/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j --config Release"
            )
            raise RuntimeError(msg)

        if not gguf:
            msg = (
                "No GGUF base model found. "
                f"Checked: {[str(p) for p in _GGUF_CANDIDATES]}. "
                "Place a GGUF at ~/models/nemotron/Nemotron-Super-512B-v1.Q4_K_M.gguf"
            )
            raise RuntimeError(msg)

        # Convert data
        data_path    = cycle_dir / "training_data_llama.jsonl"
        n_examples   = _convert_to_llama_jsonl(delta, data_path)
        if n_examples == 0:
            raise RuntimeError("No valid training examples after conversion")

        adapter_out = cycle_dir / "adapter.gguf"

        # llama-finetune parameters
        rank        = self._lora_cfg.get("fast_rank", 8)
        alpha       = self._lora_cfg.get("alpha", 16)
        epochs      = 3
        ctx         = 2048
        # Approximate steps: epochs * ceil(n_examples / batch=1)
        adam_iter   = epochs * n_examples
        adam_iter   = min(500, max(20, adam_iter))

        cmd = [
            str(binary),
            "--model-base",   str(gguf),
            "--lora-out",     str(adapter_out),
            "--train-data",   str(data_path),
            "--n-gpu-layers", "999",      # offload all layers to CUDA
            "--ctx",          str(ctx),
            "--lora-r",       str(rank),
            "--lora-alpha",   str(float(alpha)),
            "--adam-iter",    str(adam_iter),
            "--threads",      "8",
        ]

        # Check for previous adapter (continued training)
        prev = self._find_prev_adapter()
        if prev and prev.exists():
            cmd += ["--lora-init-scale", "0.01"]  # warm-start from near-zero

        print(f"[TrainCycle] binary:  {binary}")
        print(f"[TrainCycle] model:   {gguf}")
        print(f"[TrainCycle] data:    {data_path} ({n_examples} examples)")
        print(f"[TrainCycle] output:  {adapter_out}")
        print(f"[TrainCycle] command: {' '.join(cmd)}")

        if dry_run:
            return TrainResult(
                cycle_id=cycle_id,
                adapter_path=adapter_out,
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
            timeout=7200,   # 2 hour max
        )

        stdout_tail = result.stdout[-2000:] if result.stdout else ""
        stderr_tail = result.stderr[-2000:] if result.stderr else ""

        if result.returncode != 0:
            raise RuntimeError(
                f"llama-finetune failed (exit {result.returncode}):\n{stderr_tail}"
            )

        print(f"[TrainCycle] stdout tail:\n{stdout_tail}")

        # Parse loss from llama-finetune output
        # llama-finetune prints lines like: "iter 99: loss = 1.2345"
        import re
        losses = re.findall(r"loss\s*=\s*([0-9.]+)", stdout_tail + stderr_tail)
        final_loss = float(losses[-1]) if losses else -1.0

        train_result = TrainResult(
            cycle_id=cycle_id,
            adapter_path=adapter_out,
            final_loss=final_loss,
            steps_trained=adam_iter,
            delta_count=delta.delta_count,
            replay_count=delta.replay_count,
            ewc_lambda_used=self._ewc_cfg.get("lambda", 1e4),
            metadata={
                "binary":      str(binary),
                "gguf":        str(gguf),
                "examples":    n_examples,
                "adam_iter":   adam_iter,
                "lora_rank":   rank,
                "lora_alpha":  alpha,
            },
        )

        self._record_cycle(train_result)
        return train_result

    def _find_prev_adapter(self) -> Optional[Path]:
        if not ADAPTERS_DIR.exists():
            return None
        dirs = sorted(
            [d for d in ADAPTERS_DIR.iterdir() if d.is_dir() and (d / "adapter.gguf").exists()],
            key=lambda d: d.name,
        )
        return (dirs[-1] / "adapter.gguf") if dirs else None

    def _record_cycle(self, result: TrainResult) -> None:
        CYCLE_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(timezone.utc).isoformat(), **result.to_dict()}
        with open(CYCLE_HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
