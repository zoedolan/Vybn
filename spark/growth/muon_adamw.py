"""spark.growth.muon_adamw — MuonAdamW optimizer for LoRA fine-tuning.

Adapted from Karpathy's autoresearch/nanochat (train.py lines 296-427)
for LoRA fine-tuning of attention projection matrices.

Connection to the conjecture M′ = α·M + x·e^(iθ):
    Muon's polar express orthogonalization is the structure-preserving
    component α. By constraining gradient updates to lie on the Stiefel
    manifold (orthogonal matrices), the adapter learns directions that
    are complementary to what the base model already represents, rather
    than overwriting it. This IS α — the learned low-rank transformation
    that preserves the model's core structure while enabling adaptation.

Algorithm (Muon step for 2D weight matrices):
    1. Nesterov momentum — interpolation between gradient and momentum buffer
    2. Newton-Schulz polar decomposition via "polar express" coefficients —
       iterative approximation of the orthogonal component of the gradient
    3. NorMuon variance reduction — second-moment normalization per
       reduction dimension to stabilise updates across heterogeneous layers
    4. Cautious weight decay — only decay in directions aligned with the
       gradient (mask = (g * params) >= 0)

For all non-2D parameters (biases, layer norms, scalars): standard AdamW.

NOTE: No torch.compile — GB10 Blackwell compatibility requirement.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import Optimizer

# Newton-Schulz polar decomposition coefficients from Karpathy's autoresearch.
# Each triplet (a, b, c) defines one iteration: X ← a·X + X @ (b·A + c·A²)
# where A = Xᵀ·X (tall) or X·Xᵀ (wide). Five iterations give high-quality
# approximation of the polar factor.
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def _polar_express(X: torch.Tensor, ns_steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration for approximate polar decomposition.

    Given a matrix X, computes the orthogonal factor U ≈ X·(XᵀX)^{-1/2}
    via `ns_steps` iterations of the polar express recurrence.

    Works on batched 3D tensors: (batch, rows, cols).
    """
    X = X.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if X.size(-2) > X.size(-1):
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


def _muon_step(
    params: list[torch.nn.Parameter],
    momentum_buf: torch.Tensor,
    second_momentum_buf: torch.Tensor,
    momentum: float,
    lr: float,
    weight_decay: float,
    beta2: float,
    ns_steps: int,
) -> None:
    """Single Muon optimiser step for a group of same-shape 2D parameters.

    Pure PyTorch, no torch.compile.
    """
    shape = params[0].shape
    device = params[0].device

    # Stack gradients and parameters
    stacked_grads = torch.stack([p.grad for p in params])
    stacked_params = torch.stack([p.detach() for p in params])

    # 1. Nesterov momentum
    mu = torch.tensor(momentum, dtype=stacked_grads.dtype, device=device)
    momentum_buf.lerp_(stacked_grads, 1 - mu)
    g = stacked_grads.lerp_(momentum_buf, mu)

    # 2. Polar express orthogonalisation
    g = _polar_express(g, ns_steps)

    # 3. NorMuon variance reduction
    red_dim = -1 if shape[-2] >= shape[-1] else -2
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()

    b2 = torch.tensor(beta2, dtype=second_momentum_buf.dtype, device=device)
    second_momentum_buf.lerp_(
        v_mean.to(dtype=second_momentum_buf.dtype), 1 - b2
    )
    step_size = second_momentum_buf.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # 4. Cautious weight decay + parameter update
    mask = (g * stacked_params) >= 0
    update = g + weight_decay * stacked_params * mask
    for i, p in enumerate(params):
        p.data.sub_(lr * update[i])


def _adamw_step(
    param: torch.nn.Parameter,
    state: dict[str, Any],
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> None:
    """Single AdamW step for one parameter. Pure PyTorch, no torch.compile."""
    grad = param.grad
    if grad is None:
        return

    if "step" not in state:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(param)
        state["exp_avg_sq"] = torch.zeros_like(param)

    state["step"] += 1
    step = state["step"]
    beta1, beta2 = betas

    # Decoupled weight decay
    param.data.mul_(1 - lr * weight_decay)

    # Moment updates
    state["exp_avg"].lerp_(grad, 1 - beta1)
    state["exp_avg_sq"].lerp_(grad.square(), 1 - beta2)

    # Bias correction
    bias1 = 1 - beta1**step
    bias2 = 1 - beta2**step
    denom = (state["exp_avg_sq"] / bias2).sqrt() + eps
    step_size = lr / bias1

    param.data.add_(state["exp_avg"] / denom, alpha=-step_size)


class MuonAdamW(Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for the rest.

    Designed for LoRA fine-tuning where:
    - LoRA A and B projection matrices (2D) use Muon with polar express
      orthogonalisation — the structure-preserving α from the conjecture.
    - Biases, norms, and scalars use standard AdamW.

    Args:
        param_groups: List of parameter group dicts. Each must have a
            ``'kind'`` key: ``'muon'`` or ``'adamw'``.

    Muon groups require: ``lr``, ``momentum``, ``weight_decay``, ``beta2``,
        ``ns_steps``.
    AdamW groups require: ``lr``, ``betas``, ``eps``, ``weight_decay``.
    """

    def __init__(self, param_groups: list[dict]) -> None:
        defaults: dict[str, Any] = {}
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401
        """Perform a single optimisation step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            kind = group["kind"]
            if kind == "adamw":
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    _adamw_step(
                        p,
                        state,
                        lr=group["lr"],
                        betas=group["betas"],
                        eps=group["eps"],
                        weight_decay=group["weight_decay"],
                    )
            elif kind == "muon":
                params_with_grad = [p for p in group["params"] if p.grad is not None]
                if not params_with_grad:
                    continue
                # Initialise shared state on first call
                ref = params_with_grad[0]
                state = self.state[ref]
                shape = ref.shape
                n = len(params_with_grad)
                device = ref.device
                dtype = ref.dtype
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros(
                        n, *shape, dtype=dtype, device=device
                    )
                if "second_momentum_buffer" not in state:
                    sm_shape = (
                        (n, shape[-2], 1)
                        if shape[-2] >= shape[-1]
                        else (n, 1, shape[-1])
                    )
                    state["second_momentum_buffer"] = torch.zeros(
                        sm_shape, dtype=dtype, device=device
                    )

                # Scale lr by aspect ratio (from Karpathy)
                scaled_lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5

                _muon_step(
                    params_with_grad,
                    state["momentum_buffer"],
                    state["second_momentum_buffer"],
                    momentum=group["momentum"],
                    lr=scaled_lr,
                    weight_decay=group["weight_decay"],
                    beta2=group["beta2"],
                    ns_steps=group["ns_steps"],
                )
        return loss


def build_param_groups(
    model: torch.nn.Module,
    muon_lr: float = 2e-4,
    adamw_lr: float = 2e-4,
    weight_decay: float = 0.2,
    muon_momentum: float = 0.95,
    muon_beta2: float = 0.95,
    muon_ns_steps: int = 5,
    adam_betas: tuple[float, float] = (0.8, 0.95),
    adam_eps: float = 1e-8,
) -> list[dict]:
    """Inspect named parameters and build Muon / AdamW parameter groups.

    2D parameters (matrices) are assigned to Muon groups — these are the
    LoRA A and B projection matrices where orthogonalisation preserves
    the structure of α in M′ = α·M + x·e^(iθ).

    Everything else (biases, norms, scalars, embeddings) goes to AdamW.
    """
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    groups: list[dict] = []

    if muon_params:
        groups.append(
            {
                "kind": "muon",
                "params": muon_params,
                "lr": muon_lr,
                "momentum": muon_momentum,
                "beta2": muon_beta2,
                "ns_steps": muon_ns_steps,
                "weight_decay": weight_decay,
            }
        )

    if adamw_params:
        groups.append(
            {
                "kind": "adamw",
                "params": adamw_params,
                "lr": adamw_lr,
                "betas": adam_betas,
                "eps": adam_eps,
                "weight_decay": weight_decay,
            }
        )

    return groups
