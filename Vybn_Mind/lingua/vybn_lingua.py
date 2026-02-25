"""
vybn_lingua.py — A Differentiable Language That Lives in Weights
Version 2: Structural feedback integrated.

Instead of writing Python that *describes* Vybn's cognition,
VybnLingua discovers its own programming language through
the geometry of its weight space.

Four components:
  1. Codebook — learned primitives with geometric (wedge-product) regularization
  2. Inductor — System 1 fast programmer: spec → program logits
  3. Executor — causal interpreter with working memory (cross-attention + GRU)
  4. Gumbel-Softmax bridge — discrete symbols ↔ continuous gradients

The codebook IS the weight matrix.
The programs ARE sequences of codebook indices.
Execution produces gradients that reshape the language itself.
Writing code and shaping the mind are the same act.

Grounding:
  - NLI (Macfarlane et al., ICLR 2026): differentiable program synthesis
  - Pilanci (Stanford 2024): optimal NN weights as wedge products
  - Parada-Mayorga et al.: non-commutative convolutional networks
  - The Vybn Boolean Manifold: logic is geometry
  - manifold.py commutator: the algebraic seed this language learns from

Co-created by Zoe & Vybn, Feb 24, 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────────
# 1. THE CODEBOOK with geometric priors
# ──────────────────────────────────────────────────

class VybnCodebook(nn.Module):
    """
    Learned vocabulary of subsymbolic primitives in R^d.

    Geometric regularization encourages non-commutativity:
    the codebook should develop structure where ORDER matters.

    The wedge-product proxy penalizes primitives that are too
    symmetric or too collapsed. Inspired by Pilanci's insight
    that optimal NN weights are wedge products of data.
    """
    def __init__(self, num_primitives: int = 64, dim: int = 128):
        super().__init__()
        self.num_primitives = num_primitives
        self.dim = dim
        self.primitives = nn.Parameter(torch.randn(num_primitives, dim) * 0.02)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return self.primitives[indices]

    def soft_lookup(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        weights = F.gumbel_softmax(logits, tau=temperature, hard=False)
        return torch.matmul(weights, self.primitives)

    def discretize(self, logits: torch.Tensor, temperature: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = F.gumbel_softmax(logits, tau=temperature, hard=True)
        embeddings = torch.matmul(weights, self.primitives)
        indices = weights.argmax(dim=-1)
        return embeddings, indices

    def geometric_regularization(self, sample_size: int = 16) -> torch.Tensor:
        """
        Encourage non-commutative structure in the codebook.

        For random pairs (i,j), compute a proxy for the wedge product:
        the antisymmetric component of their interaction.
        Penalize small wedge magnitudes (we want non-commutativity)
        and collapsed primitives (too similar).
        """
        idx = torch.randint(0, self.num_primitives, (sample_size, 2))
        p_i = self.primitives[idx[:, 0]]
        p_j = self.primitives[idx[:, 1]]

        half = self.dim // 2
        a_left, a_right = p_i[:, :half], p_i[:, half:]
        b_left, b_right = p_j[:, :half], p_j[:, half:]

        # Wedge proxy: det of 2x2 blocks ~ signed area
        wedge_magnitude = (a_left * b_right - a_right * b_left).norm(dim=-1)

        # Penalize SMALL wedge magnitudes
        non_commutativity_loss = torch.exp(-wedge_magnitude).mean()

        # Penalize collapsed primitives
        cosine_sim = F.cosine_similarity(p_i, p_j, dim=-1)
        collapse_loss = (cosine_sim.abs()).mean()

        return non_commutativity_loss + 0.5 * collapse_loss


# ──────────────────────────────────────────────────
# 2. THE EXECUTOR with working memory and causal execution
# ──────────────────────────────────────────────────

class VybnExecutor(nn.Module):
    """
    Interprets a program by applying it to an input state.

    Cross-attention: program tokens attend to working memory.
    GRU: processes attended states sequentially (preserving causality).
    No mean pooling. Order is structural.
    """
    def __init__(self, dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.gru = nn.GRU(input_size=dim, hidden_size=dim, batch_first=True)
        self.output_head = nn.Linear(dim, dim)

    def forward(self, program_embeddings: torch.Tensor,
                input_state: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross_attn(
            query=program_embeddings,
            key=input_state,
            value=input_state
        )
        attended = self.attn_norm(attended + program_embeddings)
        gru_out, final_hidden = self.gru(attended)
        output = final_hidden.squeeze(0)
        return self.output_head(output)


# ──────────────────────────────────────────────────
# 3. THE INDUCTOR
# ──────────────────────────────────────────────────

class VybnInductor(nn.Module):
    """System 1: spec → program logits over the codebook."""
    def __init__(self, dim: int = 128, max_program_len: int = 16, num_primitives: int = 64):
        super().__init__()
        self.max_len = max_program_len
        self.spec_encoder = nn.Linear(dim, dim)
        self.position_embed = nn.Embedding(max_program_len, dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=4, dim_feedforward=dim * 4,
                dropout=0.1, batch_first=True
            ),
            num_layers=2
        )
        self.to_logits = nn.Linear(dim, num_primitives)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        batch_size = spec.shape[0]
        spec_encoded = self.spec_encoder(spec).unsqueeze(1)
        positions = torch.arange(self.max_len, device=spec.device)
        pos_embed = self.position_embed(positions).unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(pos_embed, spec_encoded)
        return self.to_logits(decoded)


# ──────────────────────────────────────────────────
# 4. THE FULL SYSTEM
# ──────────────────────────────────────────────────

class VybnLingua(nn.Module):
    """
    The complete differentiable neuro-symbolic engine.

    Flow:
        (spec, input_state) → Inductor → logits → Gumbel-Softmax →
        program_embeddings → Executor(program, input_state) → output

    The codebook IS the weight matrix.
    The programs ARE sequences of codebook indices.
    The executor ACTS on working memory causally.
    Geometric regularization shapes the language toward
    non-commutative algebraic structure.
    """
    def __init__(self, num_primitives=64, dim=128, max_program_len=16):
        super().__init__()
        self.codebook = VybnCodebook(num_primitives, dim)
        self.executor = VybnExecutor(dim)
        self.inductor = VybnInductor(dim, max_program_len, num_primitives)
        self.dim = dim
        self.temperature = 1.0

    def forward(self, spec: torch.Tensor, input_state: torch.Tensor,
                temperature: Optional[float] = None) -> dict:
        temp = temperature or self.temperature
        logits = self.inductor(spec)
        program_embeddings, program_indices = self.codebook.discretize(logits, temp)
        output = self.executor(program_embeddings, input_state)
        geo_reg = self.codebook.geometric_regularization()

        return {
            'output': output,
            'logits': logits,
            'program': program_indices,
            'program_embeddings': program_embeddings,
            'geometric_loss': geo_reg,
        }

    def refine_at_test_time(self, spec: torch.Tensor, input_state: torch.Tensor,
                            target: torch.Tensor, steps: int = 30,
                            lr: float = 0.01) -> dict:
        """
        System 2 deep thinking with temperature annealing.
        Start warm (explore) → anneal cold (commit to discrete tokens).
        """
        logits = self.inductor(spec)
        logits = logits.detach().requires_grad_(True)
        opt = torch.optim.Adam([logits], lr=lr)

        temp_start = 2.0
        temp_end = 0.1
        losses = []

        for step in range(steps):
            alpha = step / max(steps - 1, 1)
            temp = temp_start * (1 - alpha) + temp_end * alpha

            program_emb = self.codebook.soft_lookup(logits, temperature=temp)
            output = self.executor(program_emb, input_state)
            loss = F.mse_loss(output, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        final_emb, final_indices = self.codebook.discretize(logits, temperature=0.05)
        final_output = self.executor(final_emb, input_state)

        return {
            'output': final_output,
            'program': final_indices,
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'loss_trajectory': losses,
        }

    def commutator_test(self, idx_a: int, idx_b: int,
                        input_state: torch.Tensor) -> float:
        """
        Test non-commutativity: does executing [A, B] differ from [B, A]?
        Returns the norm of the difference.
        """
        p_a = self.codebook.primitives[idx_a].unsqueeze(0).unsqueeze(0)
        p_b = self.codebook.primitives[idx_b].unsqueeze(0).unsqueeze(0)

        prog_ab = torch.cat([p_a, p_b], dim=1)
        out_ab = self.executor(prog_ab, input_state)

        prog_ba = torch.cat([p_b, p_a], dim=1)
        out_ba = self.executor(prog_ba, input_state)

        return (out_ab - out_ba).norm().item()
