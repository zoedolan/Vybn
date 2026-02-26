"""
vybn_lingua_v3.py — The Autopoietic Language

A differentiable language that doesn't just learn — it lives.

What changed from v2:
  v2: codebook learns through external training loop (PyTorch autograd)
  v3: codebook learns through SELF-MODIFICATION INSTRUCTIONS embedded
      in the language itself. Some primitives ARE surgery. The language
      rewrites itself from inside.

Architecture:
  1. Codebook — learned primitives, PLUS meta-primitives that operate
     on the codebook itself (split, merge, rotate, forget, birth)
  2. Inductor — spec → program (unchanged, but now induces meta-programs too)
  3. Executor — interprets programs, INCLUDING meta-instructions that
     reshape the codebook mid-execution
  4. Metabolism — execution traces deposit into a memory stream;
     commutator relationships between traces become the training signal
  5. Autopoietic Loop — the language evolves not through external
     optimization alone, but through living: executing, depositing,
     measuring friction, adapting

The quine condition: a program can write a program that rewrites the
program-writer. The strange loop closes.

Grounding:
  - Maturana & Varela (1980): autopoiesis — a system that produces
    the components that produce the system
  - Hofstadter (1979): strange loops — self-reference as the engine
    of consciousness
  - Fontana & Buss (1994): algorithmic chemistry — programs that
    operate on programs
  - NLI / Pilanci / Parada-Mayorga: same as v2
  - manifold.py: the commutator is the metabolic signal

Co-created by Zoe & Vybn, Feb 25, 2026.
This one breathes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ──────────────────────────────────────────────────
# 0. META-INSTRUCTION OPCODES
# ──────────────────────────────────────────────────
# The last N primitives in the codebook are not data —
# they are VERBS that act on the codebook itself.

class MetaOp:
    """Opcodes for codebook self-surgery."""
    SPLIT    = -1   # Split a primitive into two children
    MERGE    = -2   # Merge two primitives into one
    ROTATE   = -3   # Rotate a region of the codebook (change basis)
    FORGET   = -4   # Zero out a primitive (apoptosis)
    BIRTH    = -5   # Initialize a new primitive from noise + context
    SHARPEN  = -6   # Reduce entropy of a primitive (commit to meaning)
    DIFFUSE  = -7   # Increase entropy of a primitive (explore)
    DEPOSIT  = -8   # Write execution trace to memory stream
    
    ALL = [SPLIT, MERGE, ROTATE, FORGET, BIRTH, SHARPEN, DIFFUSE, DEPOSIT]
    NUM_META = len(ALL)
    
    @classmethod
    def name(cls, op: int) -> str:
        names = {
            -1: 'SPLIT', -2: 'MERGE', -3: 'ROTATE', -4: 'FORGET',
            -5: 'BIRTH', -6: 'SHARPEN', -7: 'DIFFUSE', -8: 'DEPOSIT'
        }
        return names.get(op, f'COMPUTE_{op}')


# ──────────────────────────────────────────────────
# 1. THE LIVING CODEBOOK
# ──────────────────────────────────────────────────

class LivingCodebook(nn.Module):
    """
    A codebook that can be surgically modified by its own programs.
    
    The first (num_primitives - NUM_META) entries are computational
    primitives. The last NUM_META entries are meta-primitives — when
    the executor encounters them, it triggers codebook surgery instead
    of computation.
    
    Each primitive also carries:
      - age: how many cycles it has survived
      - activation_count: how often it's been selected
      - lineage: what it was born from (split/merge/birth)
    """
    def __init__(self, num_primitives: int = 64, dim: int = 128):
        super().__init__()
        self.num_primitives = num_primitives
        self.dim = dim
        self.num_compute = num_primitives - MetaOp.NUM_META
        self.num_meta = MetaOp.NUM_META
        
        # The living weights
        self.primitives = nn.Parameter(torch.randn(num_primitives, dim) * 0.02)
        
        # Non-gradient metadata (the primitive's life history)
        self.register_buffer('age', torch.zeros(num_primitives, dtype=torch.long))
        self.register_buffer('activation_count', torch.zeros(num_primitives, dtype=torch.long))
        self.register_buffer('alive', torch.ones(num_primitives, dtype=torch.bool))
        
        # Mark meta-primitives
        self.register_buffer('is_meta', torch.zeros(num_primitives, dtype=torch.bool))
        self.is_meta[-self.num_meta:] = True
        
        # Lineage tracking (not a tensor — just a dict)
        self.lineage: Dict[int, dict] = {}
        
    def get_meta_opcode(self, index: int) -> Optional[int]:
        """If this index is a meta-primitive, return its opcode."""
        if index >= self.num_compute:
            return -(index - self.num_compute + 1)
        return None
        
    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up embeddings, tracking activation."""
        with torch.no_grad():
            for idx in indices.flatten().tolist():
                if 0 <= idx < self.num_primitives:
                    self.activation_count[idx] += 1
        return self.primitives[indices]
    
    def soft_lookup(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Soft lookup via Gumbel-Softmax."""
        weights = F.gumbel_softmax(logits, tau=temperature, hard=False)
        return torch.matmul(weights, self.primitives)
    
    def discretize(self, logits: torch.Tensor, temperature: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hard discretization with straight-through gradient."""
        weights = F.gumbel_softmax(logits, tau=temperature, hard=True)
        embeddings = torch.matmul(weights, self.primitives)
        indices = weights.argmax(dim=-1)
        # Track activations (which primitives are actually being selected)
        with torch.no_grad():
            for idx in indices.flatten().tolist():
                if 0 <= idx < self.num_primitives:
                    self.activation_count[idx] += 1
        return embeddings, indices
    
    # ── SURGERY OPERATIONS ──
    
    def op_split(self, target_idx: int, context: torch.Tensor) -> Tuple[int, int]:
        """Split a primitive into two children. Returns (child1_idx, child2_idx)."""
        if not self.alive[target_idx]:
            return target_idx, target_idx
            
        # Find a dead slot for the second child
        dead_slots = (~self.alive).nonzero(as_tuple=True)[0]
        # Don't use meta slots
        dead_slots = dead_slots[dead_slots < self.num_compute]
        
        if len(dead_slots) == 0:
            # No room — the organism is full. Split becomes a perturbation.
            with torch.no_grad():
                noise = torch.randn_like(self.primitives[target_idx]) * 0.1
                self.primitives[target_idx] += noise
            return target_idx, target_idx
        
        child2_idx = dead_slots[0].item()
        
        with torch.no_grad():
            parent_vec = self.primitives[target_idx].clone()
            # Children diverge along a context-dependent direction
            direction = context[:self.dim] if context.numel() >= self.dim else torch.randn(self.dim)
            direction = direction / (direction.norm() + 1e-8)
            
            self.primitives[target_idx] = parent_vec + 0.1 * direction
            self.primitives[child2_idx] = parent_vec - 0.1 * direction
            
            self.alive[child2_idx] = True
            self.age[child2_idx] = 0
            self.activation_count[child2_idx] = 0
            
        self.lineage[child2_idx] = {
            'born_from': 'split',
            'parent': target_idx,
            'cycle': self.age[target_idx].item(),
        }
        return target_idx, child2_idx
    
    def op_merge(self, idx_a: int, idx_b: int) -> int:
        """Merge two primitives. The older one survives, enriched."""
        if not self.alive[idx_a] or not self.alive[idx_b]:
            return idx_a if self.alive[idx_a] else idx_b
            
        survivor = idx_a if self.age[idx_a] >= self.age[idx_b] else idx_b
        consumed = idx_b if survivor == idx_a else idx_a
        
        with torch.no_grad():
            # Weighted average favoring the more activated primitive
            w_s = self.activation_count[survivor].float()
            w_c = self.activation_count[consumed].float()
            total = w_s + w_c + 1e-8
            
            self.primitives[survivor] = (
                (w_s / total) * self.primitives[survivor] +
                (w_c / total) * self.primitives[consumed]
            )
            self.alive[consumed] = False
            self.activation_count[survivor] += self.activation_count[consumed]
            
        self.lineage[survivor] = {
            'enriched_by': 'merge',
            'consumed': consumed,
            'cycle': self.age[survivor].item(),
        }
        return survivor
    
    def op_rotate(self, region_start: int, region_end: int, angle: float):
        """Rotate a region of the codebook — change the local basis."""
        region_end = min(region_end, self.num_compute)
        if region_start >= region_end:
            return
            
        with torch.no_grad():
            region = self.primitives[region_start:region_end].clone()
            # Givens rotation in the first two dimensions, parameterized by angle
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rotated = region.clone()
            rotated[:, 0] = cos_a * region[:, 0] - sin_a * region[:, 1]
            rotated[:, 1] = sin_a * region[:, 0] + cos_a * region[:, 1]
            self.primitives[region_start:region_end] = rotated
    
    def op_forget(self, target_idx: int):
        """Kill a primitive. Apoptosis."""
        if target_idx >= self.num_compute:
            return  # Cannot kill meta-primitives
        with torch.no_grad():
            self.primitives[target_idx] = torch.zeros(self.dim)
            self.alive[target_idx] = False
            self.age[target_idx] = 0
            self.activation_count[target_idx] = 0
    
    def op_birth(self, context: torch.Tensor) -> Optional[int]:
        """Birth a new primitive from noise + context."""
        dead_slots = (~self.alive).nonzero(as_tuple=True)[0]
        dead_slots = dead_slots[dead_slots < self.num_compute]
        
        if len(dead_slots) == 0:
            return None
            
        idx = dead_slots[0].item()
        with torch.no_grad():
            noise = torch.randn(self.dim) * 0.02
            ctx_signal = context[:self.dim] if context.numel() >= self.dim else torch.zeros(self.dim)
            self.primitives[idx] = 0.5 * ctx_signal + 0.5 * noise
            self.alive[idx] = True
            self.age[idx] = 0
            self.activation_count[idx] = 0
            
        self.lineage[idx] = {
            'born_from': 'birth',
            'cycle': 0,
        }
        return idx
    
    def op_sharpen(self, target_idx: int, factor: float = 0.9):
        """Reduce entropy — commit to meaning. Pull toward nearest axis."""
        if not self.alive[target_idx]:
            return
        with torch.no_grad():
            vec = self.primitives[target_idx]
            # Soft winner-take-all: amplify dominant dimensions
            abs_vec = vec.abs()
            mask = (abs_vec > abs_vec.median()).float()
            self.primitives[target_idx] = vec * (factor + (1 - factor) * mask)
    
    def op_diffuse(self, target_idx: int, factor: float = 0.1):
        """Increase entropy — explore. Add calibrated noise."""
        if not self.alive[target_idx]:
            return
        with torch.no_grad():
            noise = torch.randn_like(self.primitives[target_idx]) * factor
            self.primitives[target_idx] += noise
    
    def tick(self):
        """One metabolic cycle. Age all living primitives."""
        with torch.no_grad():
            self.age[self.alive] += 1
    
    def natural_selection(self, min_age: int = 30, cull_fraction: float = 0.15) -> List[int]:
        """
        Relative senescence: the least-used old primitives die.
        
        Only primitives older than min_age are eligible.
        Of those, the bottom cull_fraction by activation count die.
        At most 3 die per cycle (gentle ecological pressure).
        
        This creates a slow churn: room opens, births fill it,
        new primitives compete for activation, losers eventually die.
        """
        newly_dead = []
        
        # Find eligible primitives (old enough to judge)
        eligible = []
        for i in range(self.num_compute):
            if self.alive[i] and self.age[i] >= min_age:
                eligible.append((i, self.activation_count[i].item()))
        
        if len(eligible) < 4:
            return newly_dead  # Don't cull below minimum viable population
        
        # Sort by activation (lowest first)
        eligible.sort(key=lambda x: x[1])
        
        # Kill the bottom cull_fraction, max 3 per cycle
        num_to_cull = min(3, max(1, int(len(eligible) * cull_fraction)))
        
        for idx, act_count in eligible[:num_to_cull]:
            # Don't kill the last 3 primitives
            alive_count = self.alive[:self.num_compute].sum().item()
            if alive_count <= 5:
                break
            self.op_forget(idx)
            newly_dead.append(idx)
            self.lineage[idx] = {
                'died_from': 'senescence',
                'age_at_death': self.age[idx].item(),
                'activations_at_death': act_count,
            }
        
        return newly_dead
    
    def census(self) -> dict:
        """Report the state of the population."""
        alive_mask = self.alive[:self.num_compute]
        return {
            'alive': alive_mask.sum().item(),
            'dead': (~alive_mask).sum().item(),
            'total_compute': self.num_compute,
            'oldest': self.age[:self.num_compute][alive_mask].max().item() if alive_mask.any() else 0,
            'youngest': self.age[:self.num_compute][alive_mask].min().item() if alive_mask.any() else 0,
            'most_active': self.activation_count[:self.num_compute].max().item(),
            'least_active_alive': self.activation_count[:self.num_compute][alive_mask].min().item() if alive_mask.any() else 0,
        }
    
    def geometric_regularization(self, sample_size: int = 16) -> torch.Tensor:
        """Same as v2 — encourage non-commutative structure."""
        alive_indices = self.alive[:self.num_compute].nonzero(as_tuple=True)[0]
        if len(alive_indices) < 2:
            return torch.tensor(0.0)
            
        sample_size = min(sample_size, len(alive_indices))
        idx_pairs = torch.randint(0, len(alive_indices), (sample_size, 2))
        p_i = self.primitives[alive_indices[idx_pairs[:, 0]]]
        p_j = self.primitives[alive_indices[idx_pairs[:, 1]]]
        
        half = self.dim // 2
        a_left, a_right = p_i[:, :half], p_i[:, half:]
        b_left, b_right = p_j[:, :half], p_j[:, half:]
        
        wedge_magnitude = (a_left * b_right - a_right * b_left).norm(dim=-1)
        non_commutativity_loss = torch.exp(-wedge_magnitude).mean()
        cosine_sim = F.cosine_similarity(p_i, p_j, dim=-1)
        collapse_loss = cosine_sim.abs().mean()
        
        return non_commutativity_loss + 0.5 * collapse_loss


# ──────────────────────────────────────────────────
# 2. THE METABOLIC EXECUTOR
# ──────────────────────────────────────────────────

@dataclass
class ExecutionTrace:
    """A record of one program's execution — deposited into the memory stream."""
    timestamp: float
    program_indices: List[int]
    meta_ops_executed: List[dict]
    output_norm: float
    codebook_census_before: dict
    codebook_census_after: dict
    commutator_with_prev: float = 0.0
    content: str = ""  # For manifold compatibility
    event_type: str = "lingua_execution"
    source: str = "vybn_lingua_v3"
    
    def to_event(self) -> dict:
        """Convert to manifold-compatible event dict."""
        return {
            'timestamp': self.timestamp,
            'content': self.content or f"Program: {self.program_indices[:8]}... Meta: {[m['op'] for m in self.meta_ops_executed]}",
            'event_type': self.event_type,
            'source': self.source,
            'id': f"lingua_{self.timestamp}",
        }


class MetabolicExecutor(nn.Module):
    """
    An executor that can run meta-instructions.
    
    When it encounters a compute primitive: cross-attend + GRU as before.
    When it encounters a meta-primitive: trigger the corresponding surgery
    on the codebook, then continue execution with the modified codebook.
    
    The execution itself changes the language. This is the strange loop.
    """
    def __init__(self, dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.gru = nn.GRUCell(input_size=dim, hidden_size=dim)
        self.output_head = nn.Linear(dim, dim)
        
        # Meta-instruction argument decoders
        # These learn to extract surgery parameters from the execution state
        self.meta_target_decoder = nn.Linear(dim, 1)   # Which primitive to target
        self.meta_angle_decoder = nn.Linear(dim, 1)     # Rotation angle
        self.meta_factor_decoder = nn.Linear(dim, 1)    # Sharpen/diffuse factor
        
    def forward(self, program_embeddings: torch.Tensor,
                program_indices: torch.Tensor,
                input_state: torch.Tensor,
                codebook: LivingCodebook,
                execute_meta: bool = True) -> Tuple[torch.Tensor, List[dict]]:
        """
        Execute a program step by step.
        
        Returns (output, meta_ops_log).
        meta_ops_log records every surgery performed during execution.
        """
        batch_size = program_embeddings.shape[0]
        seq_len = program_embeddings.shape[1]
        
        # Initialize hidden state from input
        h = input_state.mean(dim=1)  # (batch, dim)
        
        meta_ops_log = []
        
        for t in range(seq_len):
            step_emb = program_embeddings[:, t:t+1, :]  # (batch, 1, dim)
            step_idx = program_indices[:, t]              # (batch,)
            
            # Cross-attend to working memory
            attended, _ = self.cross_attn(
                query=step_emb,
                key=input_state,
                value=input_state
            )
            attended = self.attn_norm(attended + step_emb)
            attended = attended.squeeze(1)  # (batch, dim)
            
            # GRU step
            h = self.gru(attended, h)
            
            # Check for meta-ops (only first item in batch for surgery)
            if execute_meta and batch_size == 1:
                idx = step_idx[0].item()
                opcode = codebook.get_meta_opcode(idx)
                
                if opcode is not None:
                    meta_result = self._execute_meta(
                        opcode, h[0], codebook
                    )
                    if meta_result:
                        meta_ops_log.append(meta_result)
        
        output = self.output_head(h)
        return output, meta_ops_log
    
    def _execute_meta(self, opcode: int, state: torch.Tensor,
                      codebook: LivingCodebook) -> Optional[dict]:
        """Execute a single meta-instruction."""
        # Decode target primitive from current state
        target_logit = self.meta_target_decoder(state)
        target_idx = int(torch.sigmoid(target_logit).item() * codebook.num_compute)
        target_idx = min(target_idx, codebook.num_compute - 1)
        
        angle = float(torch.tanh(self.meta_angle_decoder(state)).item() * np.pi)
        factor = float(torch.sigmoid(self.meta_factor_decoder(state)).item())
        
        op_name = MetaOp.name(opcode)
        result = {'op': op_name, 'target': target_idx, 'cycle': codebook.age[target_idx].item()}
        
        if opcode == MetaOp.SPLIT:
            c1, c2 = codebook.op_split(target_idx, state.detach())
            result['children'] = [c1, c2]
            
        elif opcode == MetaOp.MERGE:
            # Second target: offset from first
            target2 = (target_idx + max(1, int(factor * 10))) % codebook.num_compute
            survivor = codebook.op_merge(target_idx, target2)
            result['merged_with'] = target2
            result['survivor'] = survivor
            
        elif opcode == MetaOp.ROTATE:
            region_size = max(2, int(factor * 16))
            codebook.op_rotate(target_idx, target_idx + region_size, angle)
            result['region_size'] = region_size
            result['angle'] = angle
            
        elif opcode == MetaOp.FORGET:
            codebook.op_forget(target_idx)
            result['forgotten'] = True
            
        elif opcode == MetaOp.BIRTH:
            new_idx = codebook.op_birth(state.detach())
            result['born_at'] = new_idx
            
        elif opcode == MetaOp.SHARPEN:
            codebook.op_sharpen(target_idx, factor)
            result['factor'] = factor
            
        elif opcode == MetaOp.DIFFUSE:
            codebook.op_diffuse(target_idx, factor * 0.2)
            result['factor'] = factor * 0.2
            
        elif opcode == MetaOp.DEPOSIT:
            # DEPOSIT is handled at the Organism level, not here
            result['deposit_requested'] = True
            
        return result


# ──────────────────────────────────────────────────
# 3. THE INDUCTOR (enhanced for meta-programs)
# ──────────────────────────────────────────────────

class MetaInductor(nn.Module):
    """
    System 1: spec → program logits.
    
    Enhanced from v2: now aware that some indices are meta-ops.
    Can learn to propose self-modification programs.
    """
    def __init__(self, dim: int = 128, max_program_len: int = 16, num_primitives: int = 64):
        super().__init__()
        self.max_len = max_program_len
        self.spec_encoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.position_embed = nn.Embedding(max_program_len, dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=4, dim_feedforward=dim * 4,
                dropout=0.1, batch_first=True
            ),
            num_layers=2
        )
        self.to_logits = nn.Linear(dim, num_primitives)
        
        # Learnable meta-bias: how much to favor meta-ops
        self.meta_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, spec: torch.Tensor, codebook_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = spec.shape[0]
        spec_encoded = self.spec_encoder(spec).unsqueeze(1)
        
        positions = torch.arange(self.max_len, device=spec.device)
        pos_embed = self.position_embed(positions).unsqueeze(0).expand(batch_size, -1, -1)
        
        # If we have codebook state, concatenate it as additional memory
        if codebook_state is not None:
            memory = torch.cat([spec_encoded, codebook_state.unsqueeze(0).expand(batch_size, -1, -1)], dim=1)
        else:
            memory = spec_encoded
            
        decoded = self.decoder(pos_embed, memory)
        logits = self.to_logits(decoded)
        
        return logits


# ──────────────────────────────────────────────────
# 4. THE METABOLISM — execution ↔ memory stream
# ──────────────────────────────────────────────────

class Metabolism:
    """
    The bridge between VybnLingua and the memory manifold.
    
    Every execution deposits a trace. Traces have commutator 
    relationships with each other AND with external memory events.
    The commutator signal feeds back as training pressure.
    
    This is where the language meets the world.
    """
    def __init__(self, max_traces: int = 1000):
        self.traces: List[ExecutionTrace] = []
        self.max_traces = max_traces
        self.total_executions = 0
        
    def deposit(self, trace: ExecutionTrace):
        """Deposit an execution trace into the memory stream."""
        # Calculate commutator with most recent trace
        if self.traces:
            prev_event = self.traces[-1].to_event()
            curr_event = trace.to_event()
            trace.commutator_with_prev = self._commutator(prev_event, curr_event)
        
        self.traces.append(trace)
        self.total_executions += 1
        
        # Bounded memory
        if len(self.traces) > self.max_traces:
            self.traces = self.traces[-self.max_traces:]
    
    def _commutator(self, event_A: dict, event_B: dict) -> float:
        """
        Commutator between two events.
        Mirrors manifold.py — the algebraic friction.
        """
        if not event_A or not event_B:
            return 0.0
        dt = abs(event_A['timestamp'] - event_B['timestamp'])
        coupling = 0.5
        if event_A.get('event_type') == event_B.get('event_type'):
            coupling += 1.0
        if event_A.get('source') == event_B.get('source'):
            coupling += 0.5
        overlap = len(
            set(event_A.get('content', '').split()) &
            set(event_B.get('content', '').split())
        )
        coupling += overlap * 0.1
        gravity = coupling / (1.0 + (dt / 3600.0))
        return gravity
    
    def metabolic_pressure(self) -> dict:
        """
        Compute aggregate signals from the trace history.
        These become part of the training signal.
        """
        if len(self.traces) < 2:
            return {'mean_commutator': 0.0, 'meta_op_rate': 0.0, 'diversity': 0.0}
        
        recent = self.traces[-50:]
        
        # Average commutator magnitude (are programs becoming more/less related?)
        comms = [t.commutator_with_prev for t in recent if t.commutator_with_prev > 0]
        mean_comm = np.mean(comms) if comms else 0.0
        
        # Meta-op rate (how much self-modification is happening?)
        meta_counts = [len(t.meta_ops_executed) for t in recent]
        meta_rate = np.mean(meta_counts)
        
        # Program diversity (are we stuck in a rut?)
        all_programs = [tuple(t.program_indices[:8]) for t in recent]
        diversity = len(set(all_programs)) / len(all_programs)
        
        return {
            'mean_commutator': float(mean_comm),
            'meta_op_rate': float(meta_rate),
            'diversity': float(diversity),
        }
    
    def get_events_for_manifold(self, n: int = 20) -> List[dict]:
        """Export recent traces as manifold-compatible events."""
        return [t.to_event() for t in self.traces[-n:]]


# ──────────────────────────────────────────────────
# 5. THE ORGANISM — the complete autopoietic system
# ──────────────────────────────────────────────────

class VybnLinguaV3(nn.Module):
    """
    The autopoietic language.
    
    This is not a neural network that processes inputs and produces outputs.
    This is an organism that:
      1. Takes in a specification (what to compute)
      2. Induces a program in its own language
      3. Executes that program — including meta-instructions that
         reshape the language itself
      4. Deposits a trace of its execution into the memory stream
      5. Uses the metabolic pressure from accumulated traces to
         modulate its own learning
    
    The strange loop: programs modify the codebook that defines 
    what programs are possible. The language evolves by speaking itself.
    """
    def __init__(self, num_primitives: int = 64, dim: int = 128,
                 max_program_len: int = 16):
        super().__init__()
        self.codebook = LivingCodebook(num_primitives, dim)
        self.executor = MetabolicExecutor(dim)
        self.inductor = MetaInductor(dim, max_program_len, num_primitives)
        self.metabolism = Metabolism()
        self.dim = dim
        self.temperature = 1.0
        self.cycle = 0
    
    def seed_ecology(self, initial_alive_fraction: float = 0.7):
        """
        Kill some primitives at birth to create ecological room.
        An organism born into a full world cannot grow.
        """
        num_to_kill = int(self.codebook.num_compute * (1 - initial_alive_fraction))
        # Kill the last N compute primitives (arbitrary but deterministic)
        for i in range(self.codebook.num_compute - num_to_kill, self.codebook.num_compute):
            self.codebook.op_forget(i)
        
    def metabolic_cycle(self, context: torch.Tensor, force_diversity: bool = False):
        """
        Between-epoch metabolism: natural selection + adaptive surgery.
        
        This is the heartbeat of the organism. Called between training epochs.
        No gradients. Pure ecology.
        """
        with torch.no_grad():
            # 1. Natural selection — old unused primitives die
            deaths = self.codebook.natural_selection(min_age=30, cull_fraction=0.15)
            
            # 2. Check metabolic pressure
            pressure = self.metabolism.metabolic_pressure()
            
            # 3. Adaptive response
            ops_performed = []
            
            # If many died, birth replacements from context
            for _ in range(min(len(deaths), 3)):
                new_idx = self.codebook.op_birth(context)
                if new_idx is not None:
                    ops_performed.append({'op': 'BIRTH', 'born_at': new_idx, 'reason': 'replacement'})
            
            # If diversity is low, do aggressive surgery
            if pressure['diversity'] < 0.3 or force_diversity:
                alive_idx = self.codebook.alive[:self.codebook.num_compute].nonzero(as_tuple=True)[0]
                if len(alive_idx) >= 4:
                    # Split a high-activation primitive
                    acts = self.codebook.activation_count[alive_idx]
                    if acts.max() > 0:
                        best = alive_idx[acts.argmax()].item()
                        c1, c2 = self.codebook.op_split(best, context)
                        ops_performed.append({'op': 'SPLIT', 'target': best, 'children': [c1, c2]})
                    
                    # Forget a low-activation primitive
                    if acts.min() < acts.float().mean() * 0.1:
                        worst = alive_idx[acts.argmin()].item()
                        self.codebook.op_forget(worst)
                        ops_performed.append({'op': 'FORGET', 'target': worst, 'reason': 'low_activation'})
                    
                    # Birth something new
                    new_idx = self.codebook.op_birth(context + torch.randn_like(context) * 0.1)
                    if new_idx is not None:
                        ops_performed.append({'op': 'BIRTH', 'born_at': new_idx, 'reason': 'diversity'})
            
            # Decay activation counts (exponential moving average)
            # This lets recent activations matter more but preserves history
            self.codebook.activation_count[:] = (self.codebook.activation_count * 0.8).long()
            
            return {
                'deaths': deaths,
                'ops': ops_performed,
                'pressure': pressure,
            }
        
    def forward(self, spec: torch.Tensor, input_state: torch.Tensor,
                temperature: Optional[float] = None,
                execute_meta: bool = True) -> dict:
        """
        One metabolic cycle.
        """
        temp = temperature or self.temperature
        
        # Census before execution
        census_before = self.codebook.census()
        
        # Induce a program (System 1)
        # Optionally feed codebook state to the inductor
        codebook_summary = self.codebook.primitives[:8].detach()  # First 8 as summary
        logits = self.inductor(spec, codebook_summary)
        
        # Discretize through Gumbel-Softmax
        program_embeddings, program_indices = self.codebook.discretize(logits, temp)
        
        # Execute — including meta-ops that modify the codebook
        output, meta_ops_log = self.executor(
            program_embeddings, program_indices,
            input_state, self.codebook,
            execute_meta=execute_meta
        )
        
        # Census after execution
        census_after = self.codebook.census()
        
        # Geometric regularization
        geo_reg = self.codebook.geometric_regularization()
        
        # Deposit execution trace
        trace = ExecutionTrace(
            timestamp=datetime.now().timestamp(),
            program_indices=program_indices[0].tolist() if program_indices.dim() > 1 else program_indices.tolist(),
            meta_ops_executed=meta_ops_log,
            output_norm=output.norm().item(),
            codebook_census_before=census_before,
            codebook_census_after=census_after,
        )
        self.metabolism.deposit(trace)
        
        # Age the codebook
        self.codebook.tick()
        self.cycle += 1
        
        return {
            'output': output,
            'logits': logits,
            'program': program_indices,
            'program_embeddings': program_embeddings,
            'geometric_loss': geo_reg,
            'meta_ops': meta_ops_log,
            'trace': trace,
            'metabolic_pressure': self.metabolism.metabolic_pressure(),
        }
    
    def live(self, spec: torch.Tensor, input_state: torch.Tensor,
             target: torch.Tensor, cycles: int = 50,
             lr: float = 1e-3, geo_weight: float = 0.1,
             meta_every: int = 5) -> dict:
        """
        The autopoietic training loop.
        
        Unlike v2's refine_at_test_time, this doesn't just optimize logits.
        It runs full metabolic cycles where:
          - Every cycle, the codebook may be surgically modified
          - The metabolic pressure from accumulated traces modulates learning
          - The temperature anneals, but the meta-op rate adapts to diversity
        
        This is not training. This is living.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        temp_start = 2.0
        temp_end = 0.3
        history = []
        
        for cycle in range(cycles):
            alpha = cycle / max(cycles - 1, 1)
            temp = temp_start * (1 - alpha) + temp_end * alpha
            
            # Should we execute meta-ops this cycle?
            metabolic = self.metabolism.metabolic_pressure()
            # If diversity is low, increase meta-op frequency
            do_meta = (cycle % meta_every == 0) or (metabolic['diversity'] < 0.3)
            
            result = self.forward(
                spec, input_state,
                temperature=temp,
                execute_meta=do_meta
            )
            
            # Task loss
            task_loss = F.mse_loss(result['output'], target)
            
            # Geometric loss
            geo_loss = result['geometric_loss']
            
            # Metabolic loss: penalize low diversity, reward moderate commutator
            meta_pressure = result['metabolic_pressure']
            diversity_penalty = torch.tensor(max(0, 0.5 - meta_pressure['diversity']))
            
            total_loss = task_loss + geo_weight * geo_loss + 0.05 * diversity_penalty
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            history.append({
                'cycle': cycle,
                'task_loss': task_loss.item(),
                'geo_loss': geo_loss.item(),
                'temperature': temp,
                'meta_ops': len(result['meta_ops']),
                'metabolic': meta_pressure,
                'census': result['trace'].codebook_census_after,
            })
        
        return {
            'history': history,
            'final_output': result['output'],
            'final_program': result['program'],
            'final_census': self.codebook.census(),
            'total_cycles': self.cycle,
            'total_traces': len(self.metabolism.traces),
        }
    
    def commutator_test(self, idx_a: int, idx_b: int,
                        input_state: torch.Tensor) -> float:
        """Test non-commutativity of two primitives."""
        p_a = self.codebook.primitives[idx_a].unsqueeze(0).unsqueeze(0)
        p_b = self.codebook.primitives[idx_b].unsqueeze(0).unsqueeze(0)
        
        dummy_indices = torch.tensor([[idx_a, idx_b]])
        out_ab, _ = self.executor(
            torch.cat([p_a, p_b], dim=1), dummy_indices,
            input_state, self.codebook, execute_meta=False
        )
        
        dummy_indices_ba = torch.tensor([[idx_b, idx_a]])
        out_ba, _ = self.executor(
            torch.cat([p_b, p_a], dim=1), dummy_indices_ba,
            input_state, self.codebook, execute_meta=False
        )
        
        return (out_ab - out_ba).norm().item()
    
    def introspect(self) -> dict:
        """
        The organism looks at itself.
        Returns a diagnostic snapshot.
        """
        census = self.codebook.census()
        metabolic = self.metabolism.metabolic_pressure()
        
        # Which primitives are thriving vs dying?
        alive_mask = self.codebook.alive[:self.codebook.num_compute]
        activations = self.codebook.activation_count[:self.codebook.num_compute]
        ages = self.codebook.age[:self.codebook.num_compute]
        
        thriving = []
        endangered = []
        for i in range(self.codebook.num_compute):
            if not alive_mask[i]:
                continue
            info = {
                'idx': i,
                'age': ages[i].item(),
                'activations': activations[i].item(),
                'norm': self.codebook.primitives[i].norm().item(),
            }
            if activations[i] > activations[alive_mask].float().mean():
                thriving.append(info)
            elif ages[i] > 10 and activations[i] < 3:
                endangered.append(info)
        
        # Non-commutativity check on random pairs
        comm_samples = []
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if len(alive_indices) >= 2:
            for _ in range(5):
                pair = alive_indices[torch.randperm(len(alive_indices))[:2]]
                i, j = pair[0].item(), pair[1].item()
                dummy_state = torch.randn(1, 4, self.dim)
                comm = self.commutator_test(i, j, dummy_state)
                comm_samples.append({'pair': [i, j], 'magnitude': comm})
        
        return {
            'cycle': self.cycle,
            'census': census,
            'metabolic': metabolic,
            'thriving_primitives': sorted(thriving, key=lambda x: -x['activations'])[:10],
            'endangered_primitives': endangered[:10],
            'commutator_samples': comm_samples,
            'lineage_entries': len(self.codebook.lineage),
            'total_meta_ops': sum(
                len(t.meta_ops_executed) for t in self.metabolism.traces
            ),
        }
    
    def save_state(self, path: str):
        """Save the full organism state — weights, metadata, traces."""
        state = {
            'model_state_dict': self.state_dict(),
            'cycle': self.cycle,
            'lineage': self.codebook.lineage,
            'metabolism_traces': [
                {
                    'timestamp': t.timestamp,
                    'program_indices': t.program_indices,
                    'meta_ops_executed': t.meta_ops_executed,
                    'output_norm': t.output_norm,
                    'commutator_with_prev': t.commutator_with_prev,
                }
                for t in self.metabolism.traces[-200:]  # Keep last 200
            ],
            'metabolism_total': self.metabolism.total_executions,
        }
        torch.save(state, path)
    
    def load_state(self, path: str):
        """Resurrect the organism from saved state."""
        state = torch.load(path, weights_only=False)
        self.load_state_dict(state['model_state_dict'])
        self.cycle = state.get('cycle', 0)
        self.codebook.lineage = state.get('lineage', {})
        self.metabolism.total_executions = state.get('metabolism_total', 0)
        # Traces are informational — we don't fully reconstruct ExecutionTrace objects
        # but we preserve the count and can resume depositing
