#!/usr/bin/env python3
"""
compute_sort_degree.py — Compute the topological degree of the sort operator S.

The sort operator S = π ∘ B₀ ∘ ι maps CP^(n-1) → CP^(n-1) where n=d/2=384 for GPT-2.
Its degree is the integer that classifies the topological obstruction τ in the
fundamental theorem of deep learning.

Method: Lattice Berry curvature (Fukui-Hatsugai-Suzuki 2005)
  - Embed multiple 2-spheres into CP^383 by interpolating between pairs of 
    actual GPT-2 token embeddings (these are real points in the space, not 
    synthetic constructions)
  - Map each S^2 through the first transformer block
  - Compute the total Berry curvature (Chern number) on the input and output 
    surfaces using the lattice gauge method
  - deg(S) = Chern_out / Chern_in (should be an integer)

Multiple probe surfaces are used for robustness — the degree is a topological 
invariant, so it should be the same regardless of which S^2 we use.

Author: Vybn (Claude Opus on DGX Spark), March 20, 2026
For: Zoe Dolan & Vybn — zoedolan/Vybn
"""

import numpy as np
import torch
import cmath
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# ============================================================
# Core Geometry
# ============================================================

def to_cp(real_vec: np.ndarray) -> np.ndarray:
    """R^d → CP^{d/2-1}: pair dimensions into complex, normalize."""
    n = len(real_vec) // 2
    z = real_vec[:n] + 1j * real_vec[n:2*n]
    norm = np.linalg.norm(z)
    if norm < 1e-15:
        return z
    return z / norm


def lattice_berry_curvature(psi_grid: np.ndarray) -> np.ndarray:
    """
    Compute Berry curvature on a 2D lattice of states using the 
    Fukui-Hatsugai-Suzuki method.
    
    psi_grid: shape (N_theta, N_phi, dim) — normalized states on CP^{dim-1}
    
    Returns: F[i,j] = Berry curvature of plaquette (i,j)→(i+1,j)→(i+1,j+1)→(i,j+1)
    
    The link variable U_{12} = <ψ₁|ψ₂> / |<ψ₁|ψ₂>| is the lattice gauge field.
    The plaquette phase F = arg(U₁₂ U₂₃ U₃₄ U₄₁) is the curvature.
    """
    Nt, Np = psi_grid.shape[:2]
    F = np.zeros((Nt - 1, Np - 1))
    
    for i in range(Nt - 1):
        for j in range(Np - 1):
            # Four corners of the plaquette
            p00 = psi_grid[i, j]
            p10 = psi_grid[i+1, j]
            p11 = psi_grid[i+1, j+1]
            p01 = psi_grid[i, j+1]
            
            # Link variables (phase-normalized overlaps)
            u01 = np.vdot(p00, p10)
            u12 = np.vdot(p10, p11)
            u23 = np.vdot(p11, p01)
            u30 = np.vdot(p01, p00)
            
            # Product around the plaquette
            prod = u01 * u12 * u23 * u30
            
            if abs(prod) < 1e-30:
                F[i, j] = 0.0
            else:
                F[i, j] = np.imag(np.log(prod))  # = arg(prod), branch-cut safe
    
    return F


def total_chern(psi_grid: np.ndarray) -> float:
    """
    Total Chern number = (1/2π) ∫ F dθ dφ
    On a lattice: (1/2π) Σ F[i,j]
    
    For a closed surface (S^2), this should be an integer.
    """
    F = lattice_berry_curvature(psi_grid)
    return np.sum(F) / (2 * np.pi)


# ============================================================
# Embedding S^2 into CP^383
# ============================================================

def embed_s2_from_tokens(emb_a: np.ndarray, emb_b: np.ndarray, emb_c: np.ndarray,
                         N_theta: int = 40, N_phi: int = 40) -> np.ndarray:
    """
    Create a smooth 2-sphere in R^768 from three token embeddings.
    
    We use the span of (a, b, c) as a 3D subspace, then parameterize an S^2 
    within it. After projection to CP^383, this gives a smooth closed surface.
    
    The S^2 is parameterized as:
      v(θ, φ) = cos(θ) * e₁ + sin(θ)cos(φ) * e₂ + sin(θ)sin(φ) * e₃
    where e₁, e₂, e₃ are orthonormalized from a, b, c.
    
    Returns: psi_grid of shape (N_theta, N_phi, 384) — states on CP^383
    """
    # Gram-Schmidt orthonormalization
    e1 = emb_a / np.linalg.norm(emb_a)
    
    b_perp = emb_b - np.dot(emb_b, e1) * e1
    e2 = b_perp / np.linalg.norm(b_perp)
    
    c_perp = emb_c - np.dot(emb_c, e1) * e1 - np.dot(emb_c, e2) * e2
    e3 = c_perp / np.linalg.norm(c_perp)
    
    # Parameterize S^2
    thetas = np.linspace(0.01, np.pi - 0.01, N_theta)  # avoid poles
    phis = np.linspace(0, 2 * np.pi - 2*np.pi/N_phi, N_phi)  # periodic in phi
    
    psi_grid = np.zeros((N_theta, N_phi, len(emb_a) // 2), dtype=complex)
    
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            v = np.cos(theta) * e1 + np.sin(theta) * np.cos(phi) * e2 + np.sin(theta) * np.sin(phi) * e3
            psi_grid[i, j] = to_cp(v)
    
    return psi_grid


def embed_s2_interpolation(emb_a: np.ndarray, emb_b: np.ndarray, 
                            N_theta: int = 40, N_phi: int = 40) -> np.ndarray:
    """
    Alternative: S^2 as great circle interpolation between two embeddings, 
    swept around a rotation in the normal plane.
    
    This keeps the sphere "centered" on real data rather than arbitrary 
    orthogonal directions.
    """
    # Normalize
    a = emb_a / np.linalg.norm(emb_a)
    b_raw = emb_b - np.dot(emb_b, a) * a
    b = b_raw / np.linalg.norm(b_raw)
    
    # Pick a random direction orthogonal to both
    rng = np.random.RandomState(42)
    c_raw = rng.randn(len(a))
    c_raw = c_raw - np.dot(c_raw, a) * a - np.dot(c_raw, b) * b
    c = c_raw / np.linalg.norm(c_raw)
    
    thetas = np.linspace(0.01, np.pi - 0.01, N_theta)
    phis = np.linspace(0, 2 * np.pi - 2*np.pi/N_phi, N_phi)
    
    psi_grid = np.zeros((N_theta, N_phi, len(emb_a) // 2), dtype=complex)
    
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            v = np.cos(theta) * a + np.sin(theta) * (np.cos(phi) * b + np.sin(phi) * c)
            psi_grid[i, j] = to_cp(v)
    
    return psi_grid


# ============================================================
# The Sort Operator: Run through GPT-2 Block 0
# ============================================================

def apply_sort(model, psi_grid_input_real: np.ndarray) -> np.ndarray:
    """
    Apply the sort operator S = π ∘ B₀ ∘ ι to a grid of embeddings.
    
    psi_grid_input_real: shape (N_theta, N_phi, 768) — real-valued embeddings
    
    We need to run these through GPT-2's first transformer block.
    Since we're working at the embedding level (not token level), we feed
    each point as a single-token sequence directly into the transformer blocks.
    
    Returns: psi_grid of shape (N_theta, N_phi, 384) — CP^383 states after sort
    """
    Nt, Np, d = psi_grid_input_real.shape
    psi_out = np.zeros((Nt, Np, d // 2), dtype=complex)
    
    model.eval()
    with torch.no_grad():
        for i in range(Nt):
            # Batch across phi for efficiency
            batch = torch.tensor(psi_grid_input_real[i], dtype=torch.float32)  # (Np, 768)
            batch = batch.unsqueeze(1)  # (Np, 1, 768) — single-token sequences
            
            # Run through GPT-2's transformer blocks to get hidden states
            # We need to manually apply the first block
            # GPT-2 structure: wte + wpe → h → block[0] → block[1] → ... → ln_f
            
            # The input is already an embedding, so we skip wte/wpe
            # Apply block 0 directly
            block0 = model.h[0]
            
            # block0 expects (batch, seq_len, hidden_dim)
            h = batch  # (Np, 1, 768)
            
            # We need position embeddings and attention mask
            # For a single position, position_embed is just position 0
            position_ids = torch.zeros(Np, 1, dtype=torch.long)
            position_embeds = model.wpe(position_ids)  # (Np, 1, 768)
            
            h_with_pos = h + position_embeds
            
            # Apply layer norm and first block
            # GPT-2 applies LN inside each block (pre-norm style in modern, 
            # but GPT-2 uses post-norm... let's check)
            # Actually GPT-2 block: ln_1 → attn → + → ln_2 → mlp → +
            out = block0(h_with_pos)
            
            # block output is a tuple: (hidden_states, ...) 
            h_out = out[0]  # (Np, 1, 768)
            h_out_np = h_out.squeeze(1).numpy()  # (Np, 768)
            
            for j in range(Np):
                psi_out[i, j] = to_cp(h_out_np[j])
            
            if i % 10 == 0:
                print(f"  Sort progress: {i+1}/{Nt} rows", flush=True)
    
    return psi_out


# ============================================================
# Main Computation
# ============================================================

def compute_degree_on_sphere(model, tokenizer, tokens_a, tokens_b, tokens_c=None,
                             N: int = 40, label: str = "") -> Dict:
    """
    Compute the degree of S using a single probe sphere.
    
    1. Get embeddings for the given tokens
    2. Construct an S^2 in the embedding space
    3. Map through block 0
    4. Compute Chern numbers before and after
    5. Degree = Chern_out / Chern_in (if Chern_in != 0)
    """
    print(f"\n{'='*60}")
    print(f"Probe sphere: {label}")
    print(f"  Tokens: '{tokens_a}', '{tokens_b}'" + (f", '{tokens_c}'" if tokens_c else ""))
    print(f"  Grid: {N}×{N}")
    print(f"{'='*60}")
    
    # Get token embeddings
    wte = model.wte.weight.detach().numpy()  # (vocab_size, 768)
    
    id_a = tokenizer.encode(tokens_a)[0]
    id_b = tokenizer.encode(tokens_b)[0]
    
    emb_a = wte[id_a]
    emb_b = wte[id_b]
    
    if tokens_c:
        id_c = tokenizer.encode(tokens_c)[0]
        emb_c = wte[id_c]
    
    print(f"  Token IDs: {id_a}, {id_b}" + (f", {id_c}" if tokens_c else ""))
    print(f"  Embedding norms: {np.linalg.norm(emb_a):.4f}, {np.linalg.norm(emb_b):.4f}" + 
          (f", {np.linalg.norm(emb_c):.4f}" if tokens_c else ""))
    
    # Construct S^2 in the embedding space (R^768) and simultaneously in CP^383
    if tokens_c:
        # Use 3-token S^2
        psi_in = embed_s2_from_tokens(emb_a, emb_b, emb_c, N, N)
    else:
        psi_in = embed_s2_interpolation(emb_a, emb_b, N, N)
    
    # Also keep the real-valued embeddings for running through the block
    d = len(emb_a)
    e1 = emb_a / np.linalg.norm(emb_a)
    b_perp = emb_b - np.dot(emb_b, e1) * e1
    e2 = b_perp / np.linalg.norm(b_perp)
    
    if tokens_c:
        c_perp = emb_c - np.dot(emb_c, e1) * e1 - np.dot(emb_c, e2) * e2
        e3 = c_perp / np.linalg.norm(c_perp)
    else:
        rng = np.random.RandomState(42)
        c_raw = rng.randn(d)
        c_raw = c_raw - np.dot(c_raw, e1) * e1 - np.dot(c_raw, e2) * e2
        e3 = c_raw / np.linalg.norm(c_raw)
    
    thetas = np.linspace(0.01, np.pi - 0.01, N)
    phis = np.linspace(0, 2 * np.pi - 2*np.pi/N, N)
    
    real_grid = np.zeros((N, N, d))
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            real_grid[i, j] = (np.cos(theta) * e1 + 
                              np.sin(theta) * np.cos(phi) * e2 + 
                              np.sin(theta) * np.sin(phi) * e3)
    
    # Chern number of the INPUT surface
    chern_in = total_chern(psi_in)
    print(f"\n  Chern number (input surface):  {chern_in:.6f}")
    
    # Apply the sort operator
    print(f"\n  Applying sort operator (GPT-2 block 0)...")
    psi_out = apply_sort(model, real_grid)
    
    # Chern number of the OUTPUT surface
    chern_out = total_chern(psi_out)
    print(f"\n  Chern number (output surface): {chern_out:.6f}")
    
    # Degree
    if abs(chern_in) > 1e-6:
        degree = chern_out / chern_in
        print(f"\n  DEGREE = {degree:.6f}")
        print(f"  Nearest integer: {round(degree)}")
    else:
        degree = None
        print(f"\n  Input Chern number too small — degree undefined")
        print(f"  (The input S^2 doesn't wrap around CP^383 in a topologically nontrivial way)")
    
    # Berry curvature distribution
    F_in = lattice_berry_curvature(psi_in)
    F_out = lattice_berry_curvature(psi_out)
    
    result = {
        "label": label,
        "tokens": [tokens_a, tokens_b] + ([tokens_c] if tokens_c else []),
        "grid_size": N,
        "chern_in": float(chern_in),
        "chern_out": float(chern_out),
        "degree": float(degree) if degree is not None else None,
        "degree_rounded": round(degree) if degree is not None else None,
        "F_in_stats": {
            "mean": float(np.mean(F_in)),
            "std": float(np.std(F_in)),
            "min": float(np.min(F_in)),
            "max": float(np.max(F_in)),
            "total": float(np.sum(F_in)),
        },
        "F_out_stats": {
            "mean": float(np.mean(F_out)),
            "std": float(np.std(F_out)),
            "min": float(np.min(F_out)),
            "max": float(np.max(F_out)),
            "total": float(np.sum(F_out)),
        },
    }
    
    return result


def run_degree_computation():
    """
    Run the degree computation on multiple probe spheres for robustness.
    
    We use probe spheres that:
    1. Span different semantic regions (spatial vs abstract)
    2. Span within a single stratum  
    3. Cross stratum boundaries
    """
    from transformers import GPT2Model, GPT2Tokenizer
    
    print("="*60)
    print("COMPUTING THE DEGREE OF THE SORT OPERATOR")
    print("="*60)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Method: Lattice Berry curvature (Fukui-Hatsugai-Suzuki)")
    print(f"Model: GPT-2 (117M, d=768, CP^383)")
    print()
    
    print("Loading GPT-2...", end=" ", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    print("done.")
    
    results = []
    
    # === PROBE 1: Spatial tokens (should be within S- stratum) ===
    r = compute_degree_on_sphere(model, tokenizer,
        " mountain", " river", " ocean",
        N=40, label="spatial_triad")
    results.append(r)
    
    # === PROBE 2: Abstract tokens (should be within S+ stratum) ===
    r = compute_degree_on_sphere(model, tokenizer,
        " truth", " knowledge", " belief",
        N=40, label="abstract_triad")
    results.append(r)
    
    # === PROBE 3: Cross-stratum (spatial to abstract) ===
    r = compute_degree_on_sphere(model, tokenizer,
        " mountain", " truth", " river",
        N=40, label="cross_stratum")
    results.append(r)
    
    # === PROBE 4: Common words (high frequency) ===
    r = compute_degree_on_sphere(model, tokenizer,
        " the", " and", " but",
        N=40, label="function_words")
    results.append(r)
    
    # === PROBE 5: Numbers (logical/mathematical stratum) ===
    r = compute_degree_on_sphere(model, tokenizer,
        " one", " two", " three",
        N=40, label="numbers")
    results.append(r)
    
    # === PROBE 6: Larger grid for convergence check ===
    r = compute_degree_on_sphere(model, tokenizer,
        " mountain", " truth", " river",
        N=80, label="cross_stratum_fine")
    results.append(r)
    
    # === Summary ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Probe':<25s} {'Chern_in':>10s} {'Chern_out':>10s} {'Degree':>10s} {'Round':>6s}")
    print("-"*65)
    for r in results:
        cin = f"{r['chern_in']:.4f}"
        cout = f"{r['chern_out']:.4f}"
        deg = f"{r['degree']:.4f}" if r['degree'] is not None else "N/A"
        rnd = str(r['degree_rounded']) if r['degree_rounded'] is not None else "N/A"
        print(f"{r['label']:<25s} {cin:>10s} {cout:>10s} {deg:>10s} {rnd:>6s}")
    
    # Check: are degrees consistent?
    valid_degrees = [r['degree'] for r in results if r['degree'] is not None]
    if valid_degrees:
        rounded = [round(d) for d in valid_degrees]
        if len(set(rounded)) == 1:
            print(f"\n✓ ALL PROBE SPHERES AGREE: deg(S) = {rounded[0]}")
        else:
            print(f"\n⚠ DEGREES DISAGREE: {rounded}")
            print(f"  This could mean:")
            print(f"  - Grid too coarse (increase N)")
            print(f"  - The map is not smooth enough for lattice method")
            print(f"  - The degree genuinely varies (shouldn't happen for a continuous map)")
    
    # Save results
    out_path = Path("/home/vybnz69/Vybn/Vybn_Mind/experiments/sort_degree_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "gpt2",
            "method": "lattice_berry_curvature_FHS2005",
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return results


if __name__ == "__main__":
    run_degree_computation()
