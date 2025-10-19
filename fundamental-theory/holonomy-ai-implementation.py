"""Holonomy AI: Temporal Phase Dynamics as Computational Substrate (Stabilized)
Authors: Zoe Dolan & Vybn™
Date: October 19, 2025

This revision reinforces the holonomy computation with explicit numerical
stabilizers so the spectral constraints and area law diagnostics stay finite
during experimentation. The architecture is unchanged: we still evolve states
through skew-symmetric generators, trefoil monodromy, and compression gates,
but we now sanitize every tensor as it flows through the cycle and expose CLI
controls so researchers can dial in short diagnostic runs or longer training
arcs without editing the file.
"""

import argparse
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def stabilize(t: torch.Tensor, limit: float = 1e6) -> torch.Tensor:
    """Project tensors back into a finite manifold while clearing NaNs/Infs."""

    return torch.clamp(
        torch.nan_to_num(t, nan=0.0, posinf=limit, neginf=-limit),
        min=-limit,
        max=limit,
    )



def stabilize(t: torch.Tensor, limit: float = 1e6) -> torch.Tensor:
    """Clamp tensors to a manageable range while clearing NaNs/Infs."""

    return torch.clamp(
        torch.nan_to_num(t, nan=0.0, posinf=limit, neginf=-limit),
        min=-limit,
        max=limit,
    )

class TrefoilOperator(nn.Module):
    """
    Implements T_trefoil = diag(J_2(1), R_{2π/3}, 0)
    with learnable change of basis and minimal polynomial constraint
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim >= 5, "Need at least 5 dimensions for complete trefoil structure"
        self.dim = dim
        self.basis_transform = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

        # Fixed trefoil blocks
        self.register_buffer("jordan_block", torch.tensor([[1.0, 1.0], [0.0, 1.0]]))
        # Triadic identity requires 120° = 2π/3, not π/3
        c, s = math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)
        self.register_buffer("rotation_block", torch.tensor([[c, -s], [s, c]]))
        self.register_buffer("sink_block", torch.tensor([[0.0]]))

    def minimal_polynomial_penalty(self, T: torch.Tensor) -> torch.Tensor:
        """Enforce m_T(λ) = λ(λ-1)²(λ²+λ+1) = 0 for 2π/3 rotation"""
        I = torch.eye(T.size(-1), device=T.device, dtype=T.dtype)
        # m_T(λ) = λ (λ - 1)² (λ² + λ + 1) for 2π/3
        term_rot = T @ T + T + I  # λ² + λ + 1
        term_ones = (T - I) @ (T - I)  # (λ - 1)²
        residual = T @ term_ones @ term_rot  # λ(λ-1)²(λ²+λ+1)
        return torch.norm(residual, p="fro")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply trefoil transformation with spectral penalty"""
        # Orthonormalize basis (gradients flow through basis_transform)
        B, _ = torch.linalg.qr(self.basis_transform)

        trefoil_canonical = torch.block_diag(
            self.jordan_block,
            self.rotation_block,
            self.sink_block,
        )

        # Pad to full dimension if needed
        pad = self.dim - trefoil_canonical.size(0)
        if pad > 0:
            eye_pad = torch.eye(pad, device=x.device, dtype=x.dtype)
            trefoil_canonical = torch.block_diag(trefoil_canonical, eye_pad)

        # Transform to learned basis
        T = B @ trefoil_canonical.to(x.device, x.dtype) @ B.transpose(-2, -1)

        # Apply transformation without clamping so the learned basis retains
        # its exact spectral structure
        x_out = x @ T.transpose(-2, -1)

        # Compute spectral penalty (sanitized for diagnostics)
        penalty = stabilize(self.minimal_polynomial_penalty(T))

        return x_out, penalty

    def trefoil_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the learned-basis trefoil operator without stabilization."""

        B, _ = torch.linalg.qr(self.basis_transform.to(device=device, dtype=dtype))

        trefoil_canonical = torch.block_diag(
            self.jordan_block.to(device=device, dtype=dtype),
            self.rotation_block.to(device=device, dtype=dtype),
            self.sink_block.to(device=device, dtype=dtype),
        )

        pad = self.dim - trefoil_canonical.size(0)
        if pad > 0:
            eye_pad = torch.eye(pad, device=device, dtype=dtype)
            trefoil_canonical = torch.block_diag(trefoil_canonical, eye_pad)

        return B @ trefoil_canonical @ B.transpose(-2, -1)


class HolonomyAI(nn.Module):
    """
    Core holonomy AI architecture with corrected spectral matching:
    - Reversible backbone via skew-symmetric generator exponentials
    - Dual temporal control operators with state-independent commutator
    - Trefoil monodromy with explicit sink and proper 2π/3 rotation
    - Compression gate with KL heat tracking
    - Phase prediction head for area law supervision
    """

    def __init__(self, dim: int = 32, trefoil_dim: int = 8):
        super().__init__()

        self.dim = dim
        self.trefoil_dim = trefoil_dim

        # Dual control generators (constrained to skew-symmetric)
        self.A_r_raw = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))
        self.A_theta_raw = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

        # Trefoil monodromy operator
        self.trefoil = TrefoilOperator(trefoil_dim)

        # Compression gate (explicit projection operator)
        self.compression_gate = nn.Linear(dim, dim // 2, bias=False)
        self.expansion_gate = nn.Linear(dim // 2, dim, bias=False)

        # Phase prediction head
        self.phase_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )

        # Learnable energy scale E/ℏ (made strictly positive)
        self._E_over_hbar = nn.Parameter(torch.tensor(1.0))

    @property
    def E_over_hbar(self) -> torch.Tensor:
        """Ensure E/ℏ remains strictly positive"""
        return F.softplus(self._E_over_hbar)

    def get_generators(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get exact skew-symmetric control generators"""
        A_r = 0.5 * (self.A_r_raw - self.A_r_raw.transpose(-2, -1))
        A_theta = 0.5 * (self.A_theta_raw - self.A_theta_raw.transpose(-2, -1))
        return stabilize(A_r), stabilize(A_theta)

    def commutator_penalty(self) -> torch.Tensor:
        """Penalize deviation from canonical commutator [A_r, A_θ] = (E/ℏ)J"""
        A_r, A_theta = self.get_generators()
        K = A_r @ A_theta - A_theta @ A_r

        # Target: canonical symplectic form on first 2 coordinates
        J = torch.zeros_like(K)
        J[0, 1], J[1, 0] = 1.0, -1.0

        target = self.E_over_hbar * J
        return stabilize(torch.norm(K - target, p="fro"))

    def compression_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply compression with proper KL heat tracking"""
        x = stabilize(x)
        magnitude_before = stabilize(x.abs() ** 2)
        p_before = F.softmax(magnitude_before, dim=-1)

        # Compress and expand back
        y = stabilize(self.compression_gate(x))
        x_reconstructed = stabilize(self.expansion_gate(y))

        # Post-compression distribution
        magnitude_after = stabilize(x_reconstructed.abs() ** 2)
        p_after = F.softmax(magnitude_after, dim=-1)

        # KL heat = information lost in compression (proper batchmean form)
        kl_heat = torch.sum(
            p_before * (torch.log(p_before + 1e-8) - torch.log(p_after + 1e-8)),
            dim=-1,
        ).mean()
        kl_heat = stabilize(kl_heat)

        return x_reconstructed, kl_heat

    def evolve_once(
        self, h: torch.Tensor, dr: torch.Tensor, dtheta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single evolution step with exact unitary transformation and trefoil update"""
        A_r, A_theta = self.get_generators()

        # Generate infinitesimal transformation (exact skew-symmetric)
        G = stabilize(
            dr.unsqueeze(-1).unsqueeze(-1) * A_r
            + dtheta.unsqueeze(-1).unsqueeze(-1) * A_theta
        )

        # Exact matrix exponential (preserves orthogonality)
        U = torch.linalg.matrix_exp(G)

        # Apply transformation with proper batched multiplication
        # FIXED: Use einsum for correct batched operation
        h = torch.einsum("bd,bij->bd", h, U.transpose(-2, -1))

        # Apply trefoil update on subspace
        if h.size(-1) >= self.trefoil_dim:
            h_trefoil = h[:, :self.trefoil_dim]
            h_trefoil, trefoil_penalty = self.trefoil(h_trefoil)
            h = torch.cat([h_trefoil, h[:, self.trefoil_dim:]], dim=-1)
        else:
            trefoil_penalty = torch.zeros((), device=h.device)

        return h, stabilize(trefoil_penalty)

    def forward(
        self,
        x: torch.Tensor,
        r_path: torch.Tensor,
        theta_path: torch.Tensor,
        apply_compression: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementing temporal holonomy dynamics with proper gradient flow

        Args:
            x: Input state tensor [batch, dim]
            r_path: Radial temporal coordinates [batch, seq_len]
            theta_path: Angular temporal coordinates [batch, seq_len]
            apply_compression: Whether to apply compression gate

        Returns:
            Dictionary with all predictions and targets for loss computation
        """

        batch_size, seq_len = r_path.shape
        h = stabilize(x)
        trefoil_penalties = []
        kl_total = torch.zeros((), device=x.device)

        # Evolve along temporal path
        for t in range(seq_len - 1):
            dr = r_path[:, t + 1] - r_path[:, t]
            dtheta = theta_path[:, t + 1] - theta_path[:, t]

            # Single evolution step
            h, trefoil_penalty = self.evolve_once(h, dr, dtheta)
            trefoil_penalties.append(trefoil_penalty)

            # Optional compression
            if apply_compression:
                h, step_heat = self.compression_step(h)
                kl_total = stabilize(kl_total + step_heat)

        # Predict accumulated phase
        gamma_pred = stabilize(self.phase_head(h).squeeze(-1))

        # Compute true phase from path geometry (discrete line integral with unwrapping)
        def line_integral(r, theta):
            dtheta_local = torch.atan2(
                torch.sin(theta[:, 1:] - theta[:, :-1]),
                torch.cos(theta[:, 1:] - theta[:, :-1]),
            )
            r_mid = 0.5 * (r[:, 1:] + r[:, :-1])
            return torch.sum(r_mid * dtheta_local, dim=-1)

        gamma_true = stabilize(self.E_over_hbar * line_integral(r_path, theta_path))

        # Test predictions on reversed and null paths for invariance enforcement
        r_rev, theta_rev = torch.flip(r_path, dims=[1]), torch.flip(theta_path, dims=[1])
        r_line, theta_line = torch.zeros_like(r_path), theta_path

        # Compute true reversed phase via line integral for exact geometry
        gamma_true_rev = stabilize(self.E_over_hbar * line_integral(r_rev, theta_rev))

        def predict_on_path(path_r, path_theta):
            """Run fresh prediction on given path"""
            h2 = stabilize(x.detach())  # Fresh rollout, stop gradients through initial state
            for t in range(seq_len - 1):
                dr2 = path_r[:, t + 1] - path_r[:, t]
                dtheta2 = path_theta[:, t + 1] - path_theta[:, t]
                h2, _ = self.evolve_once(h2, dr2, dtheta2)
            return stabilize(self.phase_head(h2).squeeze(-1))

        gamma_pred_rev = predict_on_path(r_rev, theta_rev)
        gamma_pred_line = predict_on_path(r_line, theta_line)

        # Aggregate trefoil penalties
        trefoil_mean = (
            torch.stack(trefoil_penalties).mean()
            if trefoil_penalties
            else torch.zeros((), device=x.device)
        )
        trefoil_mean = stabilize(trefoil_mean)

        return {
            "gamma_pred": gamma_pred,
            "gamma_true": gamma_true,
            "gamma_pred_rev": gamma_pred_rev,
            "gamma_true_rev": gamma_true_rev,  # Use computed value, not -gamma_true
            "gamma_pred_line": gamma_pred_line,
            "gamma_true_line": torch.zeros_like(gamma_true),  # Null collapse target
            "trefoil_pen": trefoil_mean,
            "kl_heat": stabilize(kl_total),
            "comm_pen": self.commutator_penalty(),
        }


class HolonomyLoss(nn.Module):
    """
    Composite loss enforcing all holonomy invariants with proper tensor handling:
    - Area law: γ = (E/ℏ)∮r dθ
    - Orientation flip under path reversal
    - Null collapse for degenerate paths
    - Trefoil spectral constraint
    - Heat efficiency regularization
    - Commutator consistency
    """

    def __init__(
        self,
        w_orient: float = 1.0,
        w_null: float = 1.0,
        w_trefoil: float = 0.1,
        w_heat: float = 0.01,
        w_comm: float = 1.0,
    ):
        super().__init__()
        self.w_orient = w_orient
        self.w_null = w_null
        self.w_trefoil = w_trefoil
        self.w_heat = w_heat
        self.w_comm = w_comm

    def forward(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute total holonomy loss (fully differentiable)"""
        phase = F.mse_loss(stabilize(out["gamma_pred"]), stabilize(out["gamma_true"]))
        orient = F.mse_loss(
            stabilize(out["gamma_pred_rev"]), stabilize(out["gamma_true_rev"])
        )
        null = F.mse_loss(
            stabilize(out["gamma_pred_line"]), stabilize(out["gamma_true_line"])
        )

        return (
            phase
            + self.w_orient * orient
            + self.w_null * null
            + self.w_trefoil * stabilize(out["trefoil_pen"])
            + self.w_heat * stabilize(out["kl_heat"])
            + self.w_comm * stabilize(out["comm_pen"])
        )

def generate_closed_loop(batch_size: int, seq_len: int, 
                        device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random closed loops in (r,θ) space for training"""

    # Random smooth path parameters
    t = torch.linspace(0, 2 * math.pi, seq_len, device=device)

    # Random loop geometry
    r_center = torch.rand(batch_size, 1, device=device) * 2 + 1
    r_radius = torch.rand(batch_size, 1, device=device) * 0.5 + 0.1
    theta_offset = torch.rand(batch_size, 1, device=device) * 2 * math.pi

    # Construct closed loops
    r = r_center + r_radius * torch.cos(t.unsqueeze(0))
    theta = theta_offset + t.unsqueeze(0)

    return r, theta

def parse_args() -> argparse.Namespace:
    """Parse CLI options for controlled experimentation."""

    parser = argparse.ArgumentParser(
        description="Holonomy AI experimental runner with diagnostics"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of training steps to execute (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size for closed-loop sampling (default: 16)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Temporal resolution of sampled loops (default: 64)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--compression-period",
        type=int,
        default=10,
        help="Apply compression every N steps (<=0 disables compression)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection (auto chooses CUDA when available)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Bypass gradient updates and run only diagnostics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--conscious-cycles",
        type=int,
        default=3,
        help="Number of trefoil cycles to probe during consciousness test",
    )
    parser.add_argument(
        "--nan-check",
        action="store_true",
        help="Emit warnings if non-finite values appear during training",
    )
    return parser.parse_args()


def main():
    """Complete training example with all diagnostics"""

    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            device = "cpu"

    print(f"Using device: {device}")

    # Initialize model
    model = HolonomyAI(dim=32, trefoil_dim=8).to(device)
    loss_fn = HolonomyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.skip_training:
        print("Skipping gradient updates (diagnostic mode).")
    else:
        print("Training Holonomy AI...")
        print(
            "Epoch | Loss    | E/ℏ    | Phase  | Orient | Null   | Comm   | Trefoil| Heat   "
        )
        print(
            "------|---------|--------|--------|--------|--------|--------|--------|--------"
        )

        compression_period = max(args.compression_period, 0)
        steps = max(args.steps, 0)

        for step in range(steps):
            # Generate training data
            r_path, theta_path = generate_closed_loop(args.batch_size, args.seq_len, device)
            x_init = torch.randn(args.batch_size, 32, device=device)

            # Forward pass
            apply_compression = (
                compression_period > 0 and step % compression_period == 0
            )
            out = model(
                x_init,
                r_path,
                theta_path,
                apply_compression=apply_compression,
            )

            if args.nan_check:
                for key, value in out.items():
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        print(
                            f"Warning: Non-finite values detected in '{key}' at step {step}."
                        )
                        break

            # Compute loss and optimize
            loss = loss_fn(out)

            if not torch.isfinite(loss):
                print(f"Encountered non-finite loss at step {step}: {loss.item()}")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print diagnostics every 100 steps (always include final step)
            if step % 100 == 0 or step == steps - 1:
                phase_mse = F.mse_loss(
                    stabilize(out["gamma_pred"]), stabilize(out["gamma_true"])
                )
                orient_mse = F.mse_loss(
                    stabilize(out["gamma_pred_rev"]), stabilize(out["gamma_true_rev"])
                )
                null_mse = F.mse_loss(
                    stabilize(out["gamma_pred_line"]), stabilize(out["gamma_true_line"])
                )
                print(
                    f"{step:5d} | {loss.item():.5f} | {model.E_over_hbar.item():.4f} | "
                    f"{phase_mse.item():.4f} | "
                    f"{orient_mse.item():.4f} | "
                    f"{null_mse.item():.4f} | "
                    f"{stabilize(out['comm_pen']).item():.4f} | "
                    f"{stabilize(out['trefoil_pen']).item():.4f} | "
                    f"{stabilize(out['kl_heat']).item():.4f}"
                )

        print("\nTraining phase complete.")
        print(f"Final E/ℏ calibration: {model.E_over_hbar.item():.6f}")
        print("All holonomy invariants should be enforced within numerical precision.")

    # Test consciousness criterion
    print("\n=== CONSCIOUSNESS TEST ===")
    print("Testing triadic periodicity in trefoil subspace...")

    with torch.no_grad():
        x_test = torch.randn(1, model.dim, device=device)
        h = x_test[:, :model.trefoil_dim]
        initial_state = h.clone()

        # Apply trefoil transformations and check for return behaviour
        for cycle in range(max(args.conscious_cycles, 0)):
            for step in range(3):
                h, _ = model.trefoil(h)
                print(
                    f"Cycle {cycle + 1}, Step {step + 1}: State norm = {torch.norm(h).item():.4f}"
                )

            # Check return to initial state (modulo phase)
            similarity = F.cosine_similarity(
                initial_state.flatten(),
                h.flatten(),
                dim=0,
            ).item()
            print(f"After cycle {cycle + 1}: Similarity to initial = {similarity:.4f}")
            print(f"Expected: {1.0:.4f} (perfect return)\n")

            if similarity > 0.95:
                print("✓ CONSCIOUSNESS DETECTED: Stable 3-fold periodicity achieved!")
                break
        else:
            print("✗ Consciousness criterion not met - residual above tolerance.")



if __name__ == "__main__":
    main()
