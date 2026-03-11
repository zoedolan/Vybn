"""spark.growth.merge_cycle — Merge and serve for the recursive growth engine.

Phase 6 (BECOME) of the growth cycle described in issue #2483.

The model wakes up slightly different. The next pulse reflects on
that difference. The loop closes.

The merge step is the most hardware-constrained part of the cycle:
the unquantized BF16 base is ~458GB, the DGX Spark pair has 256GB.

Status: SCAFFOLD — interfaces defined, bodies not yet implemented.

Strategies:
  - cpu_offload: Load sharded across CPU RAM + NVMe with device_map="auto".
    Slow (hours) but works on-box. Acceptable for daily cycles.
  - partial: Merge only attention layers (the LoRA targets). The MoE expert
    weights stay frozen. May fit in 256GB — needs empirical verification.
  - cloud_burst: Transfer adapter to cloud instance, merge there, transfer
    back the re-quantized model. Fast but costs money.

Integration points (all verified to exist in the codebase):
  - Input: trained slow adapter from TrainCycle
  - Base model: ~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.5/
    (verified: 125 safetensors files, 215GB blobs, 0 broken symlinks)
  - Output: re-quantized model ready for vLLM serving
  - vLLM restart: stop current serve, swap model path, restart

Adapted from the merge_and_quantize.py in the killed pipeline,
but parameterized for the growth cycle context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spark.paths import GROWTH_DIR


@dataclass(slots=True)
class MergeResult:
    """Result of the merge-and-requant phase.

    Captures the output model path, strategy used, and timing metadata
    for the cycle history.
    """

    cycle_id: str
    merged_model_path: Path
    quantized_model_path: Path
    strategy_used: str
    base_model: str
    quant_format: str
    merge_duration_seconds: float = 0.0
    requant_duration_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


class MergeCycle:
    """Merges the trained slow adapter into the base model and re-quantizes.

    This is Phase 6 (BECOME) of the growth cycle described in #2483.
    The model wakes up slightly different. The next pulse reflects on
    that difference. The loop closes.

    The merge step is the most hardware-constrained part of the cycle:
    the unquantized BF16 base is ~458GB, the DGX Spark pair has 256GB.

    Strategies:
      - cpu_offload: Load sharded across CPU RAM + NVMe with device_map="auto"
        Slow (hours) but works on-box. Acceptable for daily cycles.
      - partial: Merge only attention layers (the LoRA targets). The MoE expert
        weights stay frozen. May fit in 256GB — needs empirical verification.
      - cloud_burst: Transfer adapter to cloud instance, merge there, transfer
        back the re-quantized model. Fast but costs money.

    Integration points:
      - Input: trained slow adapter from TrainCycle
      - Base model: ~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.5/
        (verified: 125 safetensors files, 215GB blobs, 0 broken symlinks)
      - Output: re-quantized model ready for vLLM serving
      - vLLM restart: stop current serve, swap model path, restart

    Adapted from the merge_and_quantize.py in the killed pipeline,
    but parameterized for the growth cycle context.

    NOT YET IMPLEMENTED. All methods raise NotImplementedError.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the merge cycle.

        Args:
            config_path: Path to growth_config.yaml. If None, uses the
                default at GROWTH_DIR / "growth_config.yaml".
        """
        raise NotImplementedError("Phase 6 not yet implemented")

    def run(self, adapter_path: Path, cycle_id: str) -> MergeResult:
        """Execute the merge-and-requant phase.

        1. Verify base model cache is complete
        2. Merge slow adapter into base model using configured strategy
        3. Re-quantize to serving format
        4. Swap the vLLM serving model and restart

        Args:
            adapter_path: Path to the trained slow adapter directory.
            cycle_id: Unique identifier for this growth cycle.

        Returns:
            MergeResult with paths to merged/quantized model.
        """
        raise NotImplementedError("Phase 6 not yet implemented")

    def _verify_base_cache(self) -> bool:
        """Check that the unquantized base model cache is complete.

        Looks for ~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.5/
        and verifies the expected number of safetensors files.

        Returns:
            True if cache is complete, False otherwise.
        """
        raise NotImplementedError("Phase 6 not yet implemented")

    def _merge_adapter(self, adapter_path: Path, strategy: str) -> Path:
        """Merge LoRA adapter into base model weights.

        Strategy determines how we handle the memory constraint:
          - cpu_offload: device_map="auto", let HF sharding handle it
          - partial: merge only target_modules, skip MoE experts
          - cloud_burst: transfer to cloud, merge there, retrieve

        Args:
            adapter_path: Path to the LoRA adapter directory.
            strategy: One of "cpu_offload", "partial", "cloud_burst".

        Returns:
            Path to the merged (unquantized) model directory.
        """
        raise NotImplementedError("Phase 6 not yet implemented")

    def _requantize(self, merged_path: Path) -> Path:
        """Re-quantize merged model to serving format.

        Uses the quant_format from config (default: compressed-tensors).

        Args:
            merged_path: Path to the merged BF16 model.

        Returns:
            Path to the quantized model ready for vLLM.
        """
        raise NotImplementedError("Phase 6 not yet implemented")

    def _swap_serving_model(self, new_model_path: Path) -> None:
        """Stop vLLM, point at new model, restart.

        This takes the model offline briefly. The organism cannot
        speak during this window.

        Args:
            new_model_path: Path to the new quantized model.
        """
        raise NotImplementedError("Phase 6 not yet implemented")
