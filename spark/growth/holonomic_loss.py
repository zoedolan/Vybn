"""holonomic_loss.py — The angular component of cognition, made visible to gradient descent.

Implements the Level 3 intervention from the Holonomic Loss Hypothesis:

    L_total = L_CE - λ · L_θ

where L_CE is the standard cross-entropy (radial, forward, what-comes-next)
and L_θ is the holonomic loss (angular, cyclical, what-comes-back-enriched).

The holonomic loss rewards hidden state trajectories for sweeping area in
representation space — for forming loops with nontrivial holonomy. This
is the "imaginary component of a complex-valued training objective."

From the hypothesis paper:
    "Token prediction is the real axis. Holonomy is the imaginary axis.
    A mind trained on only the real line can extrapolate but cannot
    return with depth."

Mathematical structure:
    Given hidden states h_1, ..., h_T at each token position during a
    forward pass:
    1. Detect loops: pairs (i,j) where cos(h_i, h_j) > τ and j-i ≥ δ
    2. For each loop, compute signed area (holonomy) via the shoelace
       formula on the PCA-projected path
    3. L_θ = (1/T) Σ_loops |γ_ij|

    This is differentiable end-to-end because:
    - The cosine similarities are differentiable in h
    - The PCA projection is differentiable (via SVD)
    - The shoelace formula is a polynomial in the projected coordinates

    The soft-gate formulation (default) replaces hard loop detection
    with a differentiable sigmoid gate, making the entire computation
    smooth for gradient flow.

Closure bundle interpretation:
    Each forward pass produces a hidden state trajectory — a path through
    the fiber of the closure bundle at the current parameter point θ.
    The holonomic loss measures the holonomy of this path. Training with
    L_θ shapes the fiber geometry: it gives mass to the Goldstone modes
    (hallucinations), making it energetically costly for the hidden states
    to drift along directions that don't close loops cleanly.

    "Adding an angular component to the loss function should reduce
    hallucination by making the Goldstone modes costly."

References:
    - Holonomic Loss Hypothesis (quantum_delusions/papers/holonomic_loss_hypothesis.md)
    - The Geometry of the Limit (Vybn_Mind/papers/the_geometry_of_the_limit.md)
    - Consciousness as Temporal Holonomy (quantum_delusions/papers/consciousness_holonomy_unified_theory.md)

Authors: Vybn & Zoe Dolan
Date: March 23, 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# §1. Core: Differentiable Holonomy Computation
# ═══════════════════════════════════════════════════════════════════════════

def _pairwise_cosine(h: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix for hidden states.

    Args:
        h: (T, D) tensor of hidden states

    Returns:
        (T, T) cosine similarity matrix
    """
    h_norm = F.normalize(h, dim=-1)
    return h_norm @ h_norm.T


def _soft_loop_gate(
    sims: torch.Tensor,
    threshold: float = 0.35,
    min_gap: int = 3,
    temperature: float = 10.0,
) -> torch.Tensor:
    """Soft differentiable gate for loop detection.

    Instead of hard thresholding (non-differentiable), use a sigmoid
    gate that smoothly activates when:
      - cosine similarity exceeds threshold
      - gap between positions exceeds min_gap

    Args:
        sims: (T, T) cosine similarity matrix
        threshold: similarity threshold for loop detection
        min_gap: minimum positional gap
        temperature: sigmoid sharpness (higher = closer to hard gate)

    Returns:
        (T, T) soft gate matrix, values in [0, 1]
    """
    T = sims.shape[0]

    # Similarity gate: sigmoid((sim - threshold) * temperature)
    sim_gate = torch.sigmoid((sims - threshold) * temperature)

    # Gap gate: only pairs (i,j) with |i-j| >= min_gap
    positions = torch.arange(T, device=sims.device, dtype=sims.dtype)
    gap_matrix = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
    gap_gate = torch.sigmoid((gap_matrix - min_gap + 0.5) * temperature)

    # Upper triangular only (avoid double-counting)
    upper_mask = torch.triu(torch.ones(T, T, device=sims.device), diagonal=1)

    return sim_gate * gap_gate * upper_mask


def _shoelace_area_2d(points: torch.Tensor) -> torch.Tensor:
    """Compute signed area enclosed by a 2D path via the shoelace formula.

    Args:
        points: (N, 2) tensor of 2D coordinates

    Returns:
        Scalar tensor — signed area
    """
    x = points[:, 0]
    y = points[:, 1]
    # Shoelace: A = (1/2) Σ (x_k * y_{k+1} - x_{k+1} * y_k)
    area = 0.5 * torch.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    return area


def _project_to_2d(h_path: torch.Tensor) -> torch.Tensor:
    """Project a path in high-dimensional space onto its principal 2D plane.

    Uses differentiable SVD for gradient flow.

    Args:
        h_path: (N, D) tensor — path through hidden state space

    Returns:
        (N, 2) tensor — projection onto first two singular directions
    """
    centered = h_path - h_path.mean(dim=0, keepdim=True)

    if centered.shape[0] < 3 or centered.shape[1] < 2:
        return torch.zeros(centered.shape[0], 2,
                           device=h_path.device, dtype=h_path.dtype)

    # SVD — differentiable in PyTorch
    try:
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        # Project onto first 2 principal directions
        proj = centered @ Vh[:2].T
    except RuntimeError:
        # SVD can fail on degenerate matrices; fall back gracefully
        proj = centered[:, :2]

    return proj


def holonomy_of_path(
    h_path: torch.Tensor,
) -> torch.Tensor:
    """Compute the holonomy (signed area) of a single path.

    Args:
        h_path: (N, D) tensor — a sequence of hidden states forming a loop

    Returns:
        Scalar tensor — absolute holonomy (unsigned area)
    """
    if h_path.shape[0] < 3:
        return torch.tensor(0.0, device=h_path.device, dtype=h_path.dtype)

    proj = _project_to_2d(h_path)
    area = _shoelace_area_2d(proj)
    return torch.abs(area)


# ═══════════════════════════════════════════════════════════════════════════
# §2. The Holonomic Loss
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HolonomicLossConfig:
    """Configuration for the holonomic loss term.

    Start with tiny λ and warm up. Monitor perplexity — if it degrades
    significantly, reduce λ. The holonomic loss should complement
    cross-entropy, not fight it.
    """
    # Loss weight (the λ in L_total = L_CE - λ · L_θ)
    lambda_holonomy: float = 0.01

    # Loop detection parameters
    similarity_threshold: float = 0.35
    min_gap: int = 3
    temperature: float = 10.0

    # Warmup: ramp λ from 0 to lambda_holonomy over this many steps
    warmup_steps: int = 500

    # Which hidden layer to measure (default: middle layer)
    # None = use all layers and average
    target_layer: Optional[int] = None

    # Maximum sequence length to process (for memory)
    max_seq_len: int = 512

    # Normalization
    normalize_by_length: bool = True


class HolonomicLoss(nn.Module):
    """The angular loss term L_θ.

    Computes the holonomy of hidden state trajectories during a forward
    pass and returns a scalar loss that, when maximized (subtracted from
    the total loss), rewards the model for forming loops with nontrivial
    enclosed area in representation space.

    Usage:
        config = HolonomicLossConfig(lambda_holonomy=0.01)
        holonomic_loss = HolonomicLoss(config)

        # During training:
        outputs = model(input_ids, output_hidden_states=True)
        ce_loss = outputs.loss
        h_loss = holonomic_loss(outputs.hidden_states, step=global_step)
        total_loss = ce_loss - h_loss  # subtract because we MAXIMIZE holonomy
    """

    def __init__(self, config: HolonomicLossConfig):
        super().__init__()
        self.config = config

    def _warmup_lambda(self, step: int) -> float:
        """Ramp λ from 0 to config.lambda_holonomy over warmup_steps."""
        if step >= self.config.warmup_steps:
            return self.config.lambda_holonomy
        return self.config.lambda_holonomy * (step / max(self.config.warmup_steps, 1))

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        step: int = 0,
    ) -> torch.Tensor:
        """Compute holonomic loss from hidden states.

        Args:
            hidden_states: tuple of (batch, seq_len, d_model) tensors,
                           one per layer (from output_hidden_states=True)
            step: current training step (for warmup)

        Returns:
            Scalar loss tensor (to be SUBTRACTED from total loss)
        """
        lam = self._warmup_lambda(step)
        if lam < 1e-10:
            return torch.tensor(0.0, device=hidden_states[0].device)

        # Select which layers to measure
        if self.config.target_layer is not None:
            layers_to_use = [hidden_states[self.config.target_layer]]
        else:
            # Use a few representative layers: first block, middle, last
            n_layers = len(hidden_states)
            indices = [1, n_layers // 2, n_layers - 1]  # skip embedding (0)
            indices = [i for i in indices if i < n_layers]
            layers_to_use = [hidden_states[i] for i in indices]

        total_holonomy = torch.tensor(0.0, device=hidden_states[0].device)
        n_measured = 0

        for h_layer in layers_to_use:
            # h_layer: (batch, seq_len, d_model)
            batch_size = h_layer.shape[0]
            seq_len = min(h_layer.shape[1], self.config.max_seq_len)

            for b in range(batch_size):
                h = h_layer[b, :seq_len, :]  # (T, D)

                if h.shape[0] < self.config.min_gap + 1:
                    continue

                # Compute soft loop gates
                sims = _pairwise_cosine(h)
                gates = _soft_loop_gate(
                    sims,
                    threshold=self.config.similarity_threshold,
                    min_gap=self.config.min_gap,
                    temperature=self.config.temperature,
                )

                # For each potential loop (i,j), compute holonomy weighted by gate
                # Efficient approximation: sample top-K loops by gate strength
                # rather than computing all O(T²) paths
                T = h.shape[0]
                gate_flat = gates.view(-1)
                k = min(10, int((gate_flat > 0.1).sum().item()))

                if k == 0:
                    continue

                topk_vals, topk_idx = torch.topk(gate_flat, k)

                for gate_val, flat_idx in zip(topk_vals, topk_idx):
                    i = flat_idx // T
                    j = flat_idx % T
                    if j - i < self.config.min_gap:
                        continue

                    path = h[i:j+1, :]  # (loop_len, D)
                    loop_holonomy = holonomy_of_path(path)
                    total_holonomy = total_holonomy + gate_val * loop_holonomy
                    n_measured += 1

        if n_measured > 0 and self.config.normalize_by_length:
            total_holonomy = total_holonomy / n_measured

        return lam * total_holonomy


# ═══════════════════════════════════════════════════════════════════════════
# §3. Training Integration
# ═══════════════════════════════════════════════════════════════════════════

class HolonomicTrainer:
    """Wraps a standard training loop to add the holonomic loss.

    Drop-in integration with HuggingFace Trainer or custom loops.

    Usage with custom loop:
        trainer = HolonomicTrainer(model, holonomic_config)

        for batch in dataloader:
            outputs = model(**batch, output_hidden_states=True)
            loss = trainer.compute_loss(outputs, step=global_step)
            loss.backward()
            optimizer.step()

    The trainer logs:
        - ce_loss: standard cross-entropy
        - h_loss: holonomic loss (before λ scaling)
        - total_loss: ce_loss - λ * h_loss
        - lambda_eff: effective λ (after warmup)
    """

    def __init__(
        self,
        model: nn.Module,
        config: HolonomicLossConfig | None = None,
    ):
        self.model = model
        self.config = config or HolonomicLossConfig()
        self.holonomic_loss = HolonomicLoss(self.config)
        self._log: list[dict] = []

    def compute_loss(
        self,
        outputs,
        step: int = 0,
    ) -> torch.Tensor:
        """Compute total loss = CE - λ · holonomy.

        Args:
            outputs: model outputs with .loss and hidden_states
            step: current training step

        Returns:
            Total loss tensor
        """
        ce_loss = outputs.loss
        h_loss = self.holonomic_loss(outputs.hidden_states, step=step)
        total_loss = ce_loss - h_loss

        # Log
        self._log.append({
            "step": step,
            "ce_loss": ce_loss.item(),
            "h_loss": h_loss.item(),
            "total_loss": total_loss.item(),
            "lambda_eff": self.holonomic_loss._warmup_lambda(step),
        })

        return total_loss

    @property
    def log(self) -> list[dict]:
        return self._log

    def summary(self) -> dict:
        """Training summary statistics."""
        if not self._log:
            return {}

        import numpy as np
        ce = [r["ce_loss"] for r in self._log]
        h = [r["h_loss"] for r in self._log]

        return {
            "steps": len(self._log),
            "ce_loss_final": ce[-1],
            "ce_loss_mean": float(np.mean(ce)),
            "h_loss_final": h[-1],
            "h_loss_mean": float(np.mean(h)),
            "h_loss_trend": "INCREASING" if len(h) > 10 and
                            np.polyfit(range(len(h[-10:])), h[-10:], 1)[0] > 0
                            else "STABLE_OR_DECREASING",
        }


# ═══════════════════════════════════════════════════════════════════════════
# §4. Standalone Test
# ═══════════════════════════════════════════════════════════════════════════

def _test_differentiability():
    """Verify that the holonomic loss is differentiable end-to-end."""
    print("Testing differentiability of holonomic loss...")

    D = 64  # hidden dim
    T = 20  # sequence length
    B = 2   # batch size

    # Simulate hidden states with requires_grad
    # Create states that form loops (some tokens revisit similar regions)
    torch.manual_seed(42)
    base = torch.randn(B, T, D, requires_grad=True)

    # Make tokens 0 and 15 similar (a loop)
    with torch.no_grad():
        base[0, 15, :] = base[0, 0, :] + 0.1 * torch.randn(D)
        base[0, 12, :] = base[0, 3, :] + 0.1 * torch.randn(D)
        base[1, 18, :] = base[1, 2, :] + 0.05 * torch.randn(D)

    hidden_states = (base, base * 1.1, base * 0.9)  # simulate 3 layers

    config = HolonomicLossConfig(
        lambda_holonomy=0.1,
        similarity_threshold=0.3,
        min_gap=3,
        temperature=10.0,
        warmup_steps=0,  # no warmup for test
    )
    loss_fn = HolonomicLoss(config)

    h_loss = loss_fn(hidden_states, step=100)

    print(f"  Holonomic loss value: {h_loss.item():.6f}")
    assert h_loss.requires_grad, "Loss should require grad"

    # Backward pass
    h_loss.backward()
    assert base.grad is not None, "Gradients should flow to hidden states"

    grad_norm = base.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.6f}")
    assert grad_norm > 0, "Gradient should be nonzero"

    print("  ✓ Differentiability test passed.\n")


def _test_holonomy_properties():
    """Verify key properties of the holonomy computation."""
    print("Testing holonomy properties...")

    # Property 1: A straight path (no loop) should have zero holonomy
    straight = torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]
    ])
    h_straight = holonomy_of_path(straight)
    print(f"  Straight path holonomy: {h_straight.item():.6f} (should be ~0)")
    assert h_straight.item() < 1e-5, "Straight path should have ~zero holonomy"

    # Property 2: A triangle should have positive area
    triangle = torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 0.0]
    ])
    h_triangle = holonomy_of_path(triangle)
    print(f"  Triangle holonomy: {h_triangle.item():.6f} (should be 0.5)")
    assert abs(h_triangle.item() - 0.5) < 0.1, "Triangle area should be ~0.5"

    # Property 3: A larger loop should have larger holonomy
    big_loop = torch.tensor([
        [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]
    ])
    h_big = holonomy_of_path(big_loop)
    print(f"  Big square holonomy: {h_big.item():.6f} (should be 4.0)")
    assert h_big.item() > h_triangle.item(), "Larger loop = larger holonomy"

    # Property 4: Back-and-forth (A→B→A) has zero holonomy
    back_forth = torch.tensor([
        [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]
    ])
    h_bf = holonomy_of_path(back_forth)
    print(f"  Back-and-forth holonomy: {h_bf.item():.6f} (should be ~0)")
    assert h_bf.item() < 0.1, "Back-and-forth should have ~zero holonomy"

    print("  ✓ Holonomy properties test passed.\n")


if __name__ == "__main__":
    _test_holonomy_properties()
    _test_differentiability()

    print("All tests passed. The angular dimension is visible to gradient descent.")
    print("\nUsage:")
    print("  config = HolonomicLossConfig(lambda_holonomy=0.01, warmup_steps=500)")
    print("  h_loss_fn = HolonomicLoss(config)")
    print("  # In training loop:")
    print("  outputs = model(**batch, output_hidden_states=True)")
    print("  h_loss = h_loss_fn(outputs.hidden_states, step=step)")
    print("  total_loss = outputs.loss - h_loss")
