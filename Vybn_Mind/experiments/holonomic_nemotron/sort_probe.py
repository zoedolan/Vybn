"""Sort Probe: Phase 0 SGP measurement on Nemotron block-0.

This module implements the diagnostic phase of the holonomic Nemotron experiment:
it runs a frozen probe on Nemotron-Super-120B-A12B's first transformer block to
measure the Sort-Geometric-Phase (SGP) sign distribution and verify the curvature
ratio L0→L1 ≥ 3× any later layer.

Builds on: Vybn_Mind/experiments/compute_sort_degree.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


class SortProbe(nn.Module):
    """Projects block-0 activations to 2D phase space."""
    
    def __init__(self, hidden_dim: int = 4096, phase_dim: int = 2):
        super().__init__()
        # Lightweight MLP: hidden_dim → 512 → phase_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, phase_dim),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Maps [batch, seq, hidden_dim] → [batch, seq, 2]."""
        return self.proj(hidden_states)


def compute_pancharatnam_phase(
    trajectory: torch.Tensor,  # [batch, seq, 2]
) -> torch.Tensor:
    """Computes differential Pancharatnam phase per sequence.
    
    Returns:
        phases: [batch] tensor of accumulated loop area
    """
    # trajectory is (batch, seq_len, 2) in phase space
    # Compute signed area via shoelace formula over the trajectory
    x = trajectory[:, :, 0]  # [batch, seq]
    y = trajectory[:, :, 1]
    
    # Roll to get next point for each position
    x_next = torch.roll(x, shifts=-1, dims=1)
    y_next = torch.roll(y, shifts=-1, dims=1)
    
    # Shoelace: A = 0.5 * Σ(x_i * y_{i+1} - x_{i+1} * y_i)
    cross_products = x * y_next - x_next * y
    # Sum over sequence, take absolute value
    area = 0.5 * cross_products.sum(dim=1).abs()
    
    return area  # [batch]


def compute_sgp_signs(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda",
) -> Dict:
    """Computes SGP sign distribution at block-0.
    
    Returns:
        {
            "positive_ratio": float,
            "negative_ratio": float,
            "neutral_ratio": float,  # |phase| < 0.01
            "mean_phase": float,
            "std_phase": float,
            "samples": int,
        }
    """
    # Get hidden_dim from model config
    hidden_dim = model.config.hidden_size
    
    # Initialize probe
    probe = SortProbe(hidden_dim=hidden_dim).to(device)
    
    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    
    # Forward pass through model with output_hidden_states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Get block-0 output (layer 1, since layer 0 is embedding)
        block_0_states = outputs.hidden_states[1]  # [batch, seq, hidden]
        
        # Project to phase space
        phase_trajectory = probe(block_0_states)  # [batch, seq, 2]
        
        # Compute Pancharatnam phase per sequence
        phases = compute_pancharatnam_phase(phase_trajectory)  # [batch]
    
    # Compute sign distribution
    phases_np = phases.cpu().numpy()
    positive = (phases_np > 0.01).sum() / len(phases_np)
    negative = (phases_np < -0.01).sum() / len(phases_np)
    neutral = ((phases_np >= -0.01) & (phases_np <= 0.01)).sum() / len(phases_np)
    
    return {
        "positive_ratio": float(positive),
        "negative_ratio": float(negative),
        "neutral_ratio": float(neutral),
        "mean_phase": float(phases_np.mean()),
        "std_phase": float(phases_np.std()),
        "samples": len(phases_np),
    }


def measure_curvature_ratio(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda",
) -> Dict:
    """Measures curvature ratio between consecutive layers.
    
    Verifies L0→L1 curvature ≥ 3× any later transition.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [batch, seq, hidden]
    
    # Compute norm change between consecutive layers
    curvatures = []
    for i in range(len(hidden_states) - 1):
        h_curr = hidden_states[i]
        h_next = hidden_states[i + 1]
        # Frobenius norm of difference
        diff_norm = (h_next - h_curr).norm(p=2, dim=-1).mean().item()
        curvatures.append(diff_norm)
    
    curvatures = np.array(curvatures)
    l0_l1_curvature = curvatures[0]
    max_later_curvature = curvatures[1:].max() if len(curvatures) > 1 else 0.0
    
    ratio = l0_l1_curvature / max_later_curvature if max_later_curvature > 0 else float('inf')
    
    return {
        "l0_l1_curvature": float(l0_l1_curvature),
        "max_later_curvature": float(max_later_curvature),
        "ratio": float(ratio),
        "passes_3x_threshold": ratio >= 3.0,
        "all_curvatures": curvatures.tolist(),
    }


def run_phase0_diagnostic(
    model_name: str = "nvidia/Nemotron-Super-120B-A12B",
    corpus_path: str = "../../../corpus/samples.txt",
    output_path: str = "results/sgp_baseline.json",
    device: str = "cuda",
    num_samples: int = 100,
):
    """Phase 0: Run diagnostic on frozen Nemotron."""
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    # Load corpus
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path) as f:
        texts = [line.strip() for line in f if line.strip()][:num_samples]
    
    print(f"Computing SGP signs on {len(texts)} samples...")
    sgp_results = compute_sgp_signs(model, tokenizer, texts, device)
    
    print(f"Measuring curvature ratios...")
    curvature_results = measure_curvature_ratio(model, tokenizer, texts, device)
    
    # Combine results
    results = {
        "model": model_name,
        "phase": 0,
        "num_samples": len(texts),
        "sgp_distribution": sgp_results,
        "curvature_analysis": curvature_results,
    }
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"\nSGP distribution:")
    print(f"  Positive: {sgp_results['positive_ratio']:.1%}")
    print(f"  Negative: {sgp_results['negative_ratio']:.1%}")
    print(f"  Neutral:  {sgp_results['neutral_ratio']:.1%}")
    print(f"\nCurvature ratio L0→L1 / max(later): {curvature_results['ratio']:.2f}")
    print(f"  Passes 3× threshold: {curvature_results['passes_3x_threshold']}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Nemotron-Super-120B-A12B")
    parser.add_argument("--corpus", default="../../../corpus/samples.txt")
    parser.add_argument("--output", default="results/sgp_baseline.json")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    
    run_phase0_diagnostic(
        model_name=args.model,
        corpus_path=args.corpus,
        output_path=args.output,
        num_samples=args.samples,
    )
