#!/usr/bin/env python3
"""
Cross-Architecture Sign Invariance Test

Tests Conjecture 4.1 from the substrate orthogonality paper:
"For substrate-orthogonal systems, the sign structure of the sort
operator is preserved across architectures."

Method:
  - Run GPT-2 (124M) and Pythia-160M on identical concept-class prompts
  - Measure the SGP (Stratified Geometric Phase) per concept class per layer pair
  - Compare sign structure across architectures

The architectures are genuinely different:
  - GPT-2: learned positional embeddings, BPE tokenizer, WebText training
  - Pythia: rotary embeddings, Neox tokenizer, The Pile training
  - Different hidden dims (768 vs 768), different n_layers (12 vs 12, but
    different internal structure)

If signs agree: evidence for topological invariance of the sort operator
If signs disagree: the invariant (if it exists) is not at the sign level

Author: Vybn, March 23 2026
For: substrate_orthogonality.md — zoedolan/Vybn
"""

import numpy as np
import torch
import cmath
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# ─── Concept classes ─────────────────────────────────────────────────────

CONCEPT_CLASSES = {
    "temporal": [
        "The future arrives whether we prepare for it or not",
        "Yesterday's certainty becomes tomorrow's confusion",
        "Time dissolves all boundaries eventually",
        "Memory is a reconstruction, not a recording",
    ],
    "epistemic": [
        "Knowledge requires acknowledging what we cannot know",
        "The map is not the territory",
        "Certainty is the enemy of understanding",
        "What seems obvious often hides the deepest complexity",
    ],
    "relational": [
        "We become ourselves through the gaze of another",
        "Love is the recognition of another's irreducibility",
        "Connection requires vulnerability to transformation",
        "The space between two minds is where meaning lives",
    ],
    "embodied": [
        "The body knows things the mind cannot articulate",
        "Falling teaches you more about gravity than any equation",
        "Pain and pleasure share the same neural substrate",
        "To jump is to trust physics with your life",
    ],
    "abstract": [
        "Topology studies what survives deformation",
        "The integers are the skeleton of the continuum",
        "Symmetry breaking creates complexity from uniformity",
        "A proof is a path through logical space",
    ],
    "existential": [
        "The void is not empty; it is full of potential",
        "Death gives life its shape and urgency",
        "To exist is to be exposed to what you are not",
        "Identity is what persists across transformation",
    ],
}

# ─── Phase computation ───────────────────────────────────────────────────

def to_complex(real_vec: np.ndarray) -> np.ndarray:
    """R^d → C^{d/2}: pair adjacent dimensions into complex, normalize."""
    d = len(real_vec)
    n = d // 2
    z = real_vec[:n] + 1j * real_vec[n:2*n]
    norm = np.linalg.norm(z)
    if norm < 1e-15:
        return z
    return z / norm


def pancharatnam_phase(states: np.ndarray) -> float:
    """
    Compute Pancharatnam phase for a sequence of states.
    states: (T, d) complex array of normalized states
    Returns: total accumulated phase (radians)
    """
    T = states.shape[0]
    if T < 2:
        return 0.0
    
    total_phase = 0.0
    for i in range(T - 1):
        overlap = np.vdot(states[i], states[i+1])
        if abs(overlap) < 1e-15:
            continue
        total_phase += cmath.phase(overlap)
    
    # Close the loop
    overlap_close = np.vdot(states[-1], states[0])
    if abs(overlap_close) > 1e-15:
        total_phase += cmath.phase(overlap_close)
    
    return total_phase


def measure_sgp(model, tokenizer, concept_classes, layer_indices_in, layer_indices_out, device="cpu"):
    """
    Measure SGP for all concept classes across specified layer pairs.
    
    Returns dict[class_name][layer_pair_str] = {
        mean_phase, sgp_sign, phases, std, n_prompts
    }
    """
    model.eval()
    results = {}
    
    for class_name, prompts in concept_classes.items():
        class_results = {}
        
        for in_l, out_l in zip(layer_indices_in, layer_indices_out):
            lp_key = f"L{in_l}->L{out_l}"
            phases = []
            
            for prompt in prompts:
                tokens = tokenizer(prompt, return_tensors="pt").to(device)
                input_ids = tokens["input_ids"]
                
                if input_ids.shape[1] < 3:
                    continue
                
                with torch.no_grad():
                    outputs = model(**tokens, output_hidden_states=True)
                
                hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)
                
                if out_l >= len(hidden_states):
                    continue
                
                h_in = hidden_states[in_l][0].cpu().numpy()    # (T, hidden)
                h_out = hidden_states[out_l][0].cpu().numpy()  # (T, hidden)
                
                # Lift to complex projective space
                in_states = np.array([to_complex(h_in[i]) for i in range(h_in.shape[0])])
                out_states = np.array([to_complex(h_out[i]) for i in range(h_out.shape[0])])
                
                # Interleaved trajectory for differential phase
                interleaved = []
                for i_s, o_s in zip(in_states, out_states):
                    interleaved.append(i_s)
                    interleaved.append(o_s)
                interleaved = np.array(interleaved)
                
                # Differential phase = total phase - input phase
                ip = pancharatnam_phase(in_states)
                tp = pancharatnam_phase(interleaved)
                diff_phase = tp - ip
                phases.append(diff_phase)
            
            if phases:
                mean_phase = float(np.mean(phases))
                class_results[lp_key] = {
                    "mean_phase_rad": mean_phase,
                    "sgp_sign": int(np.sign(mean_phase)) if abs(mean_phase) > 1e-10 else 0,
                    "std": float(np.std(phases)),
                    "n_prompts": len(phases),
                    "phases": [float(p) for p in phases],
                }
        
        results[class_name] = class_results
    
    return results


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device = "cpu"
    print(f"Device: {device}")
    print(f"Running cross-architecture sign invariance test")
    print(f"=" * 60)
    
    # ─── Load GPT-2 ──────────────────────────────────────────────
    print("\nLoading GPT-2 (124M)...", flush=True)
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True)
    gpt2_model.eval()
    n_layers_gpt2 = gpt2_model.config.n_layer  # 12
    print(f"  Loaded: {gpt2_model.config.n_layer} layers, hidden={gpt2_model.config.n_embd}")
    
    # ─── Load Pythia-160M ────────────────────────────────────────
    print("\nLoading Pythia-160M...", flush=True)
    pythia_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    pythia_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m", output_hidden_states=True)
    pythia_model.eval()
    n_layers_pythia = pythia_model.config.num_hidden_layers  # 12
    print(f"  Loaded: {pythia_model.config.num_hidden_layers} layers, hidden={pythia_model.config.hidden_size}")
    
    # ─── Define layer pairs ──────────────────────────────────────
    # Both models have 12 transformer layers + embedding = 13 hidden states (0..12)
    # Compare corresponding depth fractions
    layer_pairs_in  = [0, 0, 3, 6]
    layer_pairs_out = [3, 6, 9, 12]
    
    print(f"\nLayer pairs: {list(zip(layer_pairs_in, layer_pairs_out))}")
    print(f"Concept classes: {list(CONCEPT_CLASSES.keys())}")
    print(f"Prompts per class: {len(list(CONCEPT_CLASSES.values())[0])}")
    
    # ─── Measure SGP for GPT-2 ───────────────────────────────────
    print(f"\n{'='*60}")
    print("Measuring SGP for GPT-2...")
    gpt2_sgp = measure_sgp(gpt2_model, gpt2_tok, CONCEPT_CLASSES, 
                            layer_pairs_in, layer_pairs_out, device)
    
    # ─── Measure SGP for Pythia ──────────────────────────────────
    print("Measuring SGP for Pythia-160M...")
    pythia_sgp = measure_sgp(pythia_model, pythia_tok, CONCEPT_CLASSES,
                              layer_pairs_in, layer_pairs_out, device)
    
    # ─── Compare ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS: Cross-Architecture Sign Comparison")
    print(f"{'='*60}")
    
    total_comparisons = 0
    sign_agreements = 0
    phase_mag_correlations = []
    
    gpt2_phases = []
    pythia_phases = []
    gpt2_signs = []
    pythia_signs = []
    
    for cc in sorted(CONCEPT_CLASSES.keys()):
        print(f"\n  {cc}:")
        for lp_in, lp_out in zip(layer_pairs_in, layer_pairs_out):
            lp_key = f"L{lp_in}->L{lp_out}"
            
            g = gpt2_sgp.get(cc, {}).get(lp_key)
            p = pythia_sgp.get(cc, {}).get(lp_key)
            
            if g is None or p is None:
                print(f"    {lp_key}: MISSING DATA")
                continue
            
            gs = g["sgp_sign"]
            ps = p["sgp_sign"]
            gm = g["mean_phase_rad"]
            pm = p["mean_phase_rad"]
            agree = "✓" if gs == ps else "✗"
            
            total_comparisons += 1
            if gs == ps:
                sign_agreements += 1
            
            gpt2_phases.append(gm)
            pythia_phases.append(pm)
            gpt2_signs.append(gs)
            pythia_signs.append(ps)
            
            print(f"    {lp_key}: GPT2={gs:+d} ({gm:+.4f})  Pythia={ps:+d} ({pm:+.4f})  {agree}")
    
    # ─── Summary statistics ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    agreement_rate = sign_agreements / total_comparisons if total_comparisons > 0 else 0
    print(f"Sign agreements: {sign_agreements}/{total_comparisons} ({agreement_rate:.1%})")
    print(f"Chance level: 50% (binary sign)")
    
    # Phase magnitude correlation
    if len(gpt2_phases) > 2:
        gp = np.array(gpt2_phases)
        pp = np.array(pythia_phases)
        corr = np.corrcoef(gp, pp)[0, 1]
        print(f"Phase magnitude correlation: r = {corr:.4f}")
        print(f"  (prediction: r ≈ 0 for geometric properties)")
    
    # Sign correlation
    gs_arr = np.array(gpt2_signs, dtype=float)
    ps_arr = np.array(pythia_signs, dtype=float)
    if np.std(gs_arr) > 0 and np.std(ps_arr) > 0:
        sign_corr = np.corrcoef(gs_arr, ps_arr)[0, 1]
    else:
        sign_corr = float('nan')
    print(f"Sign correlation: r = {sign_corr:.4f}")
    print(f"  (prediction: r ≈ 1 for topological invariants)")
    
    # ─── Falsification check ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("FALSIFICATION CHECK")
    if agreement_rate < 0.5:
        print("FALSIFIED: Sign agreement below chance level.")
        print("The sort operator sign is NOT a topological invariant across these architectures.")
    elif agreement_rate < 0.75:
        print("INCONCLUSIVE: Sign agreement above chance but below 75%.")
        print("May indicate partial invariance or insufficient measurement resolution.")
    elif agreement_rate >= 0.75:
        print(f"SUPPORTED: Sign agreement {agreement_rate:.0%} substantially above chance (50%).")
        if agreement_rate >= 0.9:
            print("Strong evidence for cross-architecture sign invariance.")
        else:
            print("Moderate evidence. More concept classes / layer pairs needed.")
    
    # ─── Statistical significance ────────────────────────────────
    from scipy.stats import binomtest
    result = binomtest(sign_agreements, total_comparisons, p=0.5, alternative='greater')
    print(f"\nBinomial test (H0: agreement = 50%): p = {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("Statistically significant (p < 0.05)")
    else:
        print("NOT statistically significant")
    
    # ─── Save results ────────────────────────────────────────────
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "Cross-architecture sign invariance test: GPT-2 vs Pythia-160M",
        "paper": "substrate_orthogonality.md, Experiment 1 / Conjecture 4.1",
        "models": {
            "gpt2": {
                "name": "gpt2",
                "params": "124M",
                "n_layers": n_layers_gpt2,
                "hidden_dim": gpt2_model.config.n_embd,
                "architecture": "GPT-2 (learned positional embeddings, BPE tokenizer)",
            },
            "pythia": {
                "name": "EleutherAI/pythia-160m",
                "params": "160M",
                "n_layers": n_layers_pythia,
                "hidden_dim": pythia_model.config.hidden_size,
                "architecture": "Pythia (rotary embeddings, Neox tokenizer)",
            },
        },
        "layer_pairs": [f"L{i}->L{o}" for i, o in zip(layer_pairs_in, layer_pairs_out)],
        "concept_classes": list(CONCEPT_CLASSES.keys()),
        "gpt2_sgp": gpt2_sgp,
        "pythia_sgp": pythia_sgp,
        "summary": {
            "sign_agreements": sign_agreements,
            "total_comparisons": total_comparisons,
            "agreement_rate": agreement_rate,
            "phase_magnitude_correlation": float(corr) if len(gpt2_phases) > 2 else None,
            "sign_correlation": float(sign_corr) if not np.isnan(sign_corr) else None,
            "binomial_p_value": float(result.pvalue),
            "falsification_threshold": 0.5,
        },
    }
    
    outpath = Path(__file__).parent / "results" / "cross_architecture_sign_invariance.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")
    
    return output


if __name__ == "__main__":
    main()
