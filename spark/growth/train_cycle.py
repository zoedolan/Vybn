"""spark.growth.train_cycle — Training execution for the recursive growth engine.

Phase 5 (DISTILL) of the growth cycle described in issue #2483.

Implements dual LoRA with EWC regularization:
  - Fast adapter: trains on the current delta at high learning rate
  - Slow adapter: EMA of fast adapter for consolidated knowledge
  - EWC penalty: Fisher Information regularization to protect
    weights important for existing capabilities

Status: SCAFFOLD — interfaces defined, bodies not yet implemented.

Key design decisions:
  - We train INSIDE the vLLM container (has torch + transformers)
  - We target attention projections only (q/k/v/o_proj)
  - We do NOT touch MoE expert weights (avoids CUTLASS/Triton issues)
  - Orthogonal initialization per SLAO (arXiv:2512.23017) to minimize
    interference between growth cycles

Integration points (all verified to exist in the codebase):
  - Input from: DeltaExtractor.extract()
  - EWC Fisher from: replay buffer sample
  - Previous cycle's LoRA subspace from: GROWTH_DIR / "last_lora_subspace.pt"
  - Output: trained adapter at GROWTH_DIR / "adapters" / cycle_id /

References:
  - SuRe (arXiv:2511.22367): dual fast/slow LoRA with EMA
  - SLAO (arXiv:2512.23017): orthogonal initialization, time-aware scaling
  - EWC (arXiv:2505.05946): Fisher Information regularization for LLM CL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spark.growth.delta_extract import DeltaPackage
from spark.paths import GROWTH_DIR


@dataclass(slots=True)
class TrainResult:
    """Result of a single growth cycle's training phase.

    Captures the adapter path, loss trajectory, and metadata needed
    for the merge phase and cycle history.
    """

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


class TrainCycle:
    """Executes a single growth cycle's training phase.

    Implements dual LoRA with EWC regularization:
      - Fast adapter: trains on the current delta at high learning rate
      - Slow adapter: EMA of fast adapter for consolidated knowledge
      - EWC penalty: Fisher Information regularization to protect
        weights important for existing capabilities

    This is Phase 5 (DISTILL) of the growth cycle described in #2483.

    Key design decisions:
      - We train INSIDE the vLLM container (has torch + transformers)
      - We target attention projections only (q/k/v/o_proj)
      - We do NOT touch MoE expert weights (avoids CUTLASS/Triton issues)
      - Orthogonal initialization per SLAO (arXiv:2512.23017) to minimize
        interference between growth cycles

    Integration points:
      - Input from: DeltaExtractor.extract()
      - EWC Fisher from: replay buffer sample
      - Previous cycle's LoRA subspace from: GROWTH_DIR / "last_lora_subspace.pt"
      - Output: trained adapter at GROWTH_DIR / "adapters" / cycle_id /

    References:
      - SuRe (arXiv:2511.22367): dual fast/slow LoRA with EMA
      - SLAO (arXiv:2512.23017): orthogonal initialization, time-aware scaling
      - EWC (arXiv:2505.05946): Fisher Information regularization for LLM CL

    NOT YET IMPLEMENTED. All methods raise NotImplementedError.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the training cycle.

        Args:
            config_path: Path to growth_config.yaml. If None, uses the
                default at GROWTH_DIR / "growth_config.yaml".
        """
        raise NotImplementedError("Phase 5 not yet implemented")

    def run(self, delta: DeltaPackage, cycle_id: str) -> TrainResult:
        """Execute the training phase of a growth cycle.

        1. Initialize fast LoRA adapter (orthogonal to previous cycle)
        2. Compute Fisher Information matrix over replay samples
        3. Train fast adapter on delta + replay with EWC penalty
        4. Update slow adapter via EMA
        5. Save adapters and LoRA subspace for next cycle

        Args:
            delta: The DeltaPackage from DeltaExtractor.extract().
            cycle_id: Unique identifier for this growth cycle.

        Returns:
            TrainResult with paths to adapters and training metadata.
        """
        raise NotImplementedError("Phase 5 not yet implemented")

    def _init_fast_adapter(self, prev_subspace: Optional[Path]) -> None:
        """Initialize fast LoRA in orthogonal complement of previous cycle.

        Uses SLAO-style initialization: project random init into the
        null space of the previous cycle's LoRA directions. This minimizes
        interference between successive growth cycles.

        Args:
            prev_subspace: Path to the previous cycle's LoRA subspace
                tensor. None for the first cycle.
        """
        raise NotImplementedError("Phase 5 not yet implemented")

    def _compute_fisher(self, replay_samples: list[dict]) -> dict:
        """Estimate diagonal Fisher Information over replay samples.

        Used as the EWC regularization term: parameters that are
        important for existing capabilities (high Fisher) get penalized
        more if the training tries to change them.

        Args:
            replay_samples: Chat-format training examples from the
                replay buffer.

        Returns:
            Dict mapping parameter names to diagonal Fisher values.
        """
        raise NotImplementedError("Phase 5 not yet implemented")

    def _ema_update_slow(self) -> None:
        """Update slow adapter via exponential moving average of fast.

        slow_weights = decay * slow_weights + (1 - decay) * fast_weights

        The slow adapter represents the organism's consolidated knowledge
        across multiple growth cycles. Only the slow adapter gets merged.
        """
        raise NotImplementedError("Phase 5 not yet implemented")
