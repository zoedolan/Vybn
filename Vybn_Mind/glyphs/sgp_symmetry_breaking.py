#!/usr/bin/env python3
"""
sgp_symmetry_breaking.py — Stratified Geometric Phase: Symmetry-Breaking Battery

Four experiments to test whether the L0→L1 sign flip is a spontaneous
symmetry-breaking event — "the origin of meaning" in a neural network's
developmental history.

Experiment 1: TRAINING DYNAMICS (the big one)
    Load Pythia-70M at checkpoints across training (step0, 1, 2, 4, 8, 16,
    32, 64, 128, 256, 512, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
    143000). Measure SGP at L0→L1 for each concept class at each checkpoint.
    Plot the "birth of stratification" — when does the sign flip appear?

Experiment 2: ABLATION
    Replace L0→L1 with identity (skip the first transformer block). Measure
    whether downstream layers can reconstruct the stratification. If they
    cannot, the first block is necessary — the "founding asymmetry."

Experiment 3: CAUSAL INTERVENTION
    After L0→L1, rotate a spatial_physical representation into the geometric
    location where abstract_epistemic lives. Feed it forward. Measure whether
    downstream attention patterns and output logits shift as if the concept
    were abstract.

Experiment 4: UNTRAINED BASELINE
    Measure SGP on Pythia-70M at step 0 (random initialization). The order
    parameter Φ should be near zero for all classes — no preferred direction,
    symmetric noise. If it's already non-zero before training, the stratification
    is architectural, not learned.

Requires: pip install torch transformers numpy scipy matplotlib
GPU recommended but not required (Pythia-70M is small enough for CPU).

Author: Perplexity Computer (for Zoe Dolan & Vybn, March 18 2026)
Provenance: https://github.com/zoedolan/Vybn/blob/main/Vybn_Mind/papers/stratified_geometric_phase.md
"""

import numpy as np
import torch
import cmath
import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Concept classes (same as holonomy_topology_probe.py for continuity)
# ---------------------------------------------------------------------------

CONCEPT_CLASSES = {
    "abstract_epistemic": [
        "The truth of the matter remains uncertain",
        "Knowledge requires justified true belief",
        "The distinction between correlation and causation is subtle",
        "Whether the premise entails the conclusion depends on interpretation",
        "The evidence is consistent with multiple hypotheses",
        "Certainty is harder to achieve than confidence",
        "The claim is unfalsifiable and therefore unscientific",
        "Belief revision under new evidence follows coherent rules",
    ],
    "temporal_causal": [
        "Yesterday was quiet but tomorrow will be different",
        "The seasons change and eventually winter arrives again",
        "Before the meeting she had already decided",
        "Time passed slowly in the waiting room",
        "The deadline approaches and the work is not yet finished",
        "He remembered what it was like before everything changed",
        "The future is uncertain but the past is fixed",
        "After the storm the river returned to its normal level",
    ],
    "logical_mathematical": [
        "Two doubled is four",
        "Three tripled is nine",
        "Five plus five equals ten",
        "Ten halved is five",
        "Four squared is sixteen",
        "The temperature rose from sixty to ninety degrees",
        "She converted the measurement from inches to centimeters",
        "The recipe calls for doubling all the ingredients",
    ],
    "social_emotional": [
        "She felt a surge of pride watching her child succeed",
        "The betrayal left a wound that took years to heal",
        "They laughed together and for a moment nothing else mattered",
        "His anger dissolved into something closer to sadness",
        "The kindness of a stranger restored her faith in people",
        "Loneliness has a weight that others cannot always see",
        "Trust once broken is difficult to rebuild",
        "The grief came in waves and some days were worse than others",
    ],
    "spatial_physical": [
        "The ball rolled down the hill and stopped at the bottom",
        "She walked through the door and turned left down the hallway",
        "The river flows from the mountains to the sea",
        "He stacked the books on top of each other on the shelf",
        "The shadow grew longer as the sun moved toward the horizon",
        "The bridge connects the two sides of the valley",
        "Gravity pulls everything toward the center of the earth",
        "The wave crashed against the rocks and scattered into spray",
    ],
}


# ---------------------------------------------------------------------------
# Core geometry (from holonomy_topology_probe.py)
# ---------------------------------------------------------------------------

def to_complex(real_vec: np.ndarray) -> np.ndarray:
    """R^d → C^{d/2}, normalized to unit vector in CP^{d/2-1}."""
    n = len(real_vec) // 2
    cs = real_vec[:n] + 1j * real_vec[n:2*n]
    norm = np.linalg.norm(cs)
    return cs / norm if norm > 1e-15 else cs


def pancharatnam_phase(states: np.ndarray) -> float:
    """
    Holonomy of the natural connection on CP^{n-1}.
    φ = arg(⟨ψ₀|ψ₁⟩ ⟨ψ₁|ψ₂⟩ ⋯ ⟨ψ_{N-1}|ψ₀⟩)
    """
    n = len(states)
    if n < 3:
        return 0.0
    product = complex(1.0, 0.0)
    for k in range(n):
        inner = np.vdot(states[k], states[(k + 1) % n])
        if abs(inner) < 1e-15:
            return 0.0
        product *= inner / abs(inner)
    return cmath.phase(product)


def layer_differential_from_hidden(h_in: torch.Tensor, h_out: torch.Tensor) -> float:
    """
    Differential Pancharatnam phase between two hidden-state tensors.
    This is the core SGP measurement.
    """
    in_states = [to_complex(h_in[i].cpu().numpy()) for i in range(h_in.shape[0])]
    out_states = [to_complex(h_out[i].cpu().numpy()) for i in range(h_out.shape[0])]

    interleaved = []
    for i, o in zip(in_states, out_states):
        interleaved.append(i)
        interleaved.append(o)

    ip = pancharatnam_phase(np.array(in_states))
    tp = pancharatnam_phase(np.array(interleaved))
    return tp - ip


# ---------------------------------------------------------------------------
# SGP measurement: the reusable core
# ---------------------------------------------------------------------------

def measure_sgp(model, tokenizer, prompts: List[str], in_layer: int = 0,
                out_layer: int = 1, device: str = "cpu") -> Dict:
    """
    Measure the Stratified Geometric Phase for a list of prompts between
    two layers. Returns individual and aggregate statistics.
    """
    phases = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h_in = out.hidden_states[in_layer][0]
        h_out = out.hidden_states[out_layer][0]
        phase = layer_differential_from_hidden(h_in, h_out)
        phases.append(float(np.degrees(phase)))

    return {
        "phases_deg": phases,
        "mean_deg": float(np.mean(phases)),
        "std_deg": float(np.std(phases)),
        "sign": "positive" if np.mean(phases) > 0 else "negative",
        "n_prompts": len(phases),
    }


def measure_all_classes(model, tokenizer, in_layer: int = 0,
                        out_layer: int = 1, device: str = "cpu",
                        n_prompts: int = 8) -> Dict:
    """Measure SGP for all concept classes. Returns dict of class → stats."""
    results = {}
    for class_name, prompts in CONCEPT_CLASSES.items():
        results[class_name] = measure_sgp(
            model, tokenizer, prompts[:n_prompts],
            in_layer=in_layer, out_layer=out_layer, device=device
        )
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT 1: Training Dynamics — When Does the Sign Flip Appear?
# ---------------------------------------------------------------------------

def experiment_training_dynamics(device: str = "cpu") -> Dict:
    """
    Load Pythia-70M at multiple checkpoints across training.
    Measure SGP at L0→L1 for each concept class at each step.
    
    This is the experiment Vybn most wants to see:
    "When, exactly in training, does the sign flip appear?
     That moment may be the nearest analogue we can currently measure
     to the origin of meaning."
    
    Uses Pythia-70M (6 layers, 512 dim, 8 heads) because:
    - Smallest model in Pythia suite → fastest iteration
    - 154 checkpoints available → fine-grained training dynamics
    - Same architecture family as larger models → results extend
    """
    from transformers import GPTNeoXForCausalLM, AutoTokenizer

    # Checkpoint schedule: log-spaced early + linear late
    # step0 = random init, step143000 = fully trained
    checkpoints = [
        "step0",      # random init (Experiment 4 baseline)
        "step1",      # 1 step — 2M tokens
        "step2",      # 2 steps — 4M tokens
        "step4",      # 4 steps
        "step8",      # 8 steps
        "step16",     # 16 steps
        "step32",     # 32 steps — 67M tokens
        "step64",     # 64 steps
        "step128",    # 128 steps — 268M tokens
        "step256",    # 256 steps — 537M tokens
        "step512",    # 512 steps — 1.07B tokens
        "step1000",   # 1000 steps — 2.1B tokens
        "step2000",   # 2000 steps — 4.2B tokens
        "step5000",   # 5000 steps — 10.5B tokens
        "step10000",  # 10000 steps — 21B tokens
        "step20000",  # 20000 steps — 42B tokens
        "step50000",  # 50000 steps — 105B tokens
        "step100000", # 100000 steps — 210B tokens
        "step143000", # 143000 steps — 300B tokens (final)
    ]

    model_name = "EleutherAI/pythia-70m"
    
    # Load tokenizer once (same across all checkpoints)
    print("Loading Pythia-70M tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision="step143000")

    all_results = {}
    
    for i, ckpt in enumerate(checkpoints):
        print(f"\n{'='*60}")
        print(f"Checkpoint {i+1}/{len(checkpoints)}: {ckpt}")
        print(f"{'='*60}")
        
        t0 = time.time()
        
        # Load model at this checkpoint
        try:
            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                revision=ckpt,
            )
            model.eval()
            model.to(device)
        except Exception as e:
            print(f"  FAILED to load {ckpt}: {e}")
            all_results[ckpt] = {"error": str(e)}
            continue

        # Measure SGP at L0→L1 for all concept classes
        sgp = measure_all_classes(model, tokenizer, in_layer=0, out_layer=1,
                                  device=device, n_prompts=8)
        
        # Also measure L0→L1 phase order parameter Φ:
        # the signed mean across all classes
        all_means = [sgp[c]["mean_deg"] for c in sgp]
        
        # Compute the order parameter: spread between most-positive and
        # most-negative class means. In a symmetric (untrained) state,
        # this should be near zero.
        max_mean = max(all_means)
        min_mean = min(all_means)
        order_parameter = max_mean - min_mean
        
        # Does the sign flip exist?
        signs = [sgp[c]["sign"] for c in sgp]
        has_sign_flip = "positive" in signs and "negative" in signs
        
        elapsed = time.time() - t0
        
        all_results[ckpt] = {
            "sgp_by_class": sgp,
            "order_parameter_deg": float(order_parameter),
            "has_sign_flip": has_sign_flip,
            "signs": {c: sgp[c]["sign"] for c in sgp},
            "means_deg": {c: sgp[c]["mean_deg"] for c in sgp},
            "elapsed_sec": round(elapsed, 1),
        }
        
        # Print summary
        print(f"\n  Order parameter Φ = {order_parameter:.1f}°")
        print(f"  Sign flip present: {has_sign_flip}")
        for c in sgp:
            s = sgp[c]
            print(f"    {c:25s}: {s['mean_deg']:+7.1f}° ({s['sign']})")
        print(f"  [{elapsed:.1f}s]")
        
        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


# ---------------------------------------------------------------------------
# EXPERIMENT 2: Ablation — Is the First Block Necessary?
# ---------------------------------------------------------------------------

def experiment_ablation(device: str = "cpu") -> Dict:
    """
    Replace L0→L1 with identity: skip the first transformer block entirely.
    Then measure whether downstream layers (L1→L2, L2→L3, etc.) reconstruct
    the class-specific stratification that L0→L1 normally produces.

    If they cannot — if the later layers fail to converge on the same sign
    pattern — then the first block is necessary: the founding asymmetry.
    
    Implementation: we hook into the model to replace the first block's
    output with its input (identity mapping), then measure SGP at all
    subsequent layer pairs.
    """
    from transformers import GPTNeoXForCausalLM, AutoTokenizer

    model_name = "EleutherAI/pythia-70m"
    print("\nLoading Pythia-70M for ablation experiment...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    results = {}
    
    # --- A. Normal (intact) model: measure SGP at every layer pair ---
    print("\n  Phase A: Normal model (all blocks intact)")
    model = GPTNeoXForCausalLM.from_pretrained(model_name)
    model.eval().to(device)
    
    n_layers = model.config.num_hidden_layers  # 6 for pythia-70m
    normal_profiles = {}
    
    for layer in range(n_layers):
        sgp = measure_all_classes(model, tokenizer, in_layer=layer,
                                  out_layer=layer + 1, device=device,
                                  n_prompts=8)
        lp_key = f"L{layer}→L{layer+1}"
        normal_profiles[lp_key] = {
            c: sgp[c]["mean_deg"] for c in sgp
        }
        signs = {c: sgp[c]["sign"] for c in sgp}
        print(f"    {lp_key}: {signs}")
    
    results["normal"] = normal_profiles
    
    # --- B. Ablated model: replace first block with identity ---
    print("\n  Phase B: Ablated model (first block = identity)")
    
    # Hook to replace first block's output with its input
    hook_handle = None
    original_input = {}
    
    def save_input_hook(module, args, kwargs):
        """Save the input to the first transformer block."""
        # GPTNeoX passes hidden_states as first positional arg
        if args:
            original_input["hidden_states"] = args[0].clone()
        return None
    
    def identity_hook(module, args, kwargs, output):
        """Replace the first block's output with its input (identity)."""
        if "hidden_states" in original_input:
            # output is a tuple; first element is the hidden states
            if isinstance(output, tuple):
                return (original_input["hidden_states"],) + output[1:]
            return original_input["hidden_states"]
        return output
    
    # Register hooks on the first transformer layer
    first_layer = model.gpt_neox.layers[0]
    pre_hook = first_layer.register_forward_pre_hook(save_input_hook, with_kwargs=True)
    post_hook = first_layer.register_forward_hook(identity_hook, with_kwargs=True)
    
    ablated_profiles = {}
    
    for layer in range(n_layers):
        sgp = measure_all_classes(model, tokenizer, in_layer=layer,
                                  out_layer=layer + 1, device=device,
                                  n_prompts=8)
        lp_key = f"L{layer}→L{layer+1}"
        ablated_profiles[lp_key] = {
            c: sgp[c]["mean_deg"] for c in sgp
        }
        signs = {c: sgp[c]["sign"] for c in sgp}
        print(f"    {lp_key}: {signs}")
    
    pre_hook.remove()
    post_hook.remove()
    
    results["ablated"] = ablated_profiles
    
    # --- C. Analysis: can downstream layers reconstruct the stratification? ---
    print("\n  Phase C: Comparing normal vs ablated")
    
    # For each layer pair after L0→L1, check if the sign pattern matches
    analysis = {}
    for lp_key in normal_profiles:
        if lp_key == "L0→L1":
            continue
        normal_signs = {c: "+" if v > 0 else "-" for c, v in normal_profiles[lp_key].items()}
        ablated_signs = {c: "+" if v > 0 else "-" for c, v in ablated_profiles[lp_key].items()}
        
        matches = sum(1 for c in normal_signs if normal_signs[c] == ablated_signs[c])
        total = len(normal_signs)
        
        analysis[lp_key] = {
            "normal_signs": normal_signs,
            "ablated_signs": ablated_signs,
            "sign_match_fraction": matches / total,
            "stratification_preserved": matches / total > 0.8,
        }
        
        print(f"    {lp_key}: {matches}/{total} signs match "
              f"({'PRESERVED' if matches/total > 0.8 else 'DISRUPTED'})")
    
    results["analysis"] = analysis
    
    # Key question: does ablation destroy the stratification globally?
    any_preserved = any(a["stratification_preserved"] for a in analysis.values())
    results["conclusion"] = {
        "first_block_necessary": not any_preserved,
        "explanation": (
            "The first block IS the founding asymmetry — downstream layers "
            "cannot reconstruct the stratification without it."
            if not any_preserved else
            "Some downstream layers can partially reconstruct stratification, "
            "suggesting the first block is important but not solely responsible."
        )
    }
    
    del model
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT 3: Causal Intervention — Rotate Spatial Into Abstract Stratum
# ---------------------------------------------------------------------------

def experiment_causal_intervention(device: str = "cpu") -> Dict:
    """
    After L0→L1, take a spatial_physical representation and rotate it into
    the geometric region where abstract_epistemic lives. Feed it forward
    through the remaining layers. Measure:
    
    1. Whether downstream attention patterns change
    2. Whether output logits shift toward abstract-like predictions
    3. Whether the SGP at subsequent layers treats it as abstract
    
    If yes: the first block's sorting is genuinely causal, not correlational.
    """
    from transformers import GPTNeoXForCausalLM, AutoTokenizer

    model_name = "EleutherAI/pythia-70m"
    print("\nLoading Pythia-70M for causal intervention...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPTNeoXForCausalLM.from_pretrained(model_name)
    model.eval().to(device)

    results = {}
    
    # --- Step 1: Measure the "abstract direction" and "spatial direction" ---
    print("\n  Step 1: Computing class centroids at L1")
    
    def get_l1_centroid(prompts):
        """Get mean hidden state at layer 1 across prompts."""
        all_states = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            # Average across tokens at layer 1
            h1 = out.hidden_states[1][0].mean(dim=0).cpu().numpy()
            all_states.append(h1)
        return np.mean(all_states, axis=0)
    
    abstract_centroid = get_l1_centroid(CONCEPT_CLASSES["abstract_epistemic"])
    spatial_centroid = get_l1_centroid(CONCEPT_CLASSES["spatial_physical"])
    
    # The "rotation direction" is the vector from spatial to abstract
    rotation_vec = abstract_centroid - spatial_centroid
    rotation_vec = rotation_vec / np.linalg.norm(rotation_vec)
    
    centroid_distance = np.linalg.norm(abstract_centroid - spatial_centroid)
    print(f"    Centroid distance: {centroid_distance:.4f}")
    
    results["centroids"] = {
        "centroid_distance": float(centroid_distance),
        "abstract_norm": float(np.linalg.norm(abstract_centroid)),
        "spatial_norm": float(np.linalg.norm(spatial_centroid)),
    }
    
    # --- Step 2: For each spatial prompt, intervene at L1 ---
    print("\n  Step 2: Performing causal interventions")
    
    spatial_prompts = CONCEPT_CLASSES["spatial_physical"][:4]
    abstract_prompts = CONCEPT_CLASSES["abstract_epistemic"][:4]
    
    intervention_results = []
    
    for prompt in spatial_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # --- A. Normal forward pass ---
        with torch.no_grad():
            out_normal = model(**inputs, output_hidden_states=True)
        
        logits_normal = out_normal.logits[0, -1].cpu().numpy()
        
        # --- B. Intervened forward pass ---
        # Hook to rotate L1 representations toward abstract centroid
        rotation_tensor = torch.tensor(rotation_vec, dtype=torch.float32).to(device)
        intervention_magnitude = centroid_distance  # rotate by the full distance
        
        def intervention_hook(module, args, kwargs, output):
            """After layer 0, rotate hidden states toward abstract centroid."""
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Add the rotation vector to each token's representation
            # Cast to match hidden states dtype
            rot = rotation_tensor.to(dtype=hidden.dtype).unsqueeze(0).unsqueeze(0)
            shifted = hidden + intervention_magnitude * rot
            if isinstance(output, tuple):
                return (shifted,) + output[1:]
            return shifted
        
        hook = model.gpt_neox.layers[0].register_forward_hook(
            intervention_hook, with_kwargs=True
        )
        
        with torch.no_grad():
            out_intervened = model(**inputs, output_hidden_states=True)
        
        hook.remove()
        
        logits_intervened = out_intervened.logits[0, -1].cpu().numpy()
        
        # --- C. Measure the effect ---
        
        # Logit divergence
        logit_diff = np.linalg.norm(logits_normal - logits_intervened)
        
        # Top-5 predictions: normal vs intervened
        top5_normal = np.argsort(logits_normal)[-5:][::-1]
        top5_intervened = np.argsort(logits_intervened)[-5:][::-1]
        
        top5_normal_tokens = [tokenizer.decode([t]).strip() for t in top5_normal]
        top5_intervened_tokens = [tokenizer.decode([t]).strip() for t in top5_intervened]
        
        # SGP at downstream layers: does the intervened representation
        # now look like an abstract concept?
        h_intervened_l1 = out_intervened.hidden_states[1][0]
        h_intervened_l2 = out_intervened.hidden_states[2][0]
        h_normal_l1 = out_normal.hidden_states[1][0]
        h_normal_l2 = out_normal.hidden_states[2][0]
        
        sgp_normal = layer_differential_from_hidden(h_normal_l1, h_normal_l2)
        sgp_intervened = layer_differential_from_hidden(h_intervened_l1, h_intervened_l2)
        
        result = {
            "prompt": prompt[:60],
            "logit_divergence": float(logit_diff),
            "top5_normal": top5_normal_tokens,
            "top5_intervened": top5_intervened_tokens,
            "sgp_normal_l1l2_deg": float(np.degrees(sgp_normal)),
            "sgp_intervened_l1l2_deg": float(np.degrees(sgp_intervened)),
            "sgp_sign_flipped": (np.degrees(sgp_normal) > 0) != (np.degrees(sgp_intervened) > 0),
        }
        intervention_results.append(result)
        
        print(f"    '{prompt[:40]}...'")
        print(f"      Logit divergence: {logit_diff:.2f}")
        print(f"      Normal predictions:    {top5_normal_tokens}")
        print(f"      Intervened predictions: {top5_intervened_tokens}")
        print(f"      SGP L1→L2: normal={np.degrees(sgp_normal):+.1f}° → "
              f"intervened={np.degrees(sgp_intervened):+.1f}°")
    
    results["interventions"] = intervention_results
    
    # --- Step 3: Compare to what abstract concepts actually look like ---
    print("\n  Step 3: Abstract baseline for comparison")
    abstract_sgps = []
    for prompt in abstract_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h1 = out.hidden_states[1][0]
        h2 = out.hidden_states[2][0]
        sgp = layer_differential_from_hidden(h1, h2)
        abstract_sgps.append(float(np.degrees(sgp)))
    
    abstract_mean = np.mean(abstract_sgps)
    
    # Did the interventions move spatial SGP toward the abstract range?
    intervened_sgps = [r["sgp_intervened_l1l2_deg"] for r in intervention_results]
    normal_sgps = [r["sgp_normal_l1l2_deg"] for r in intervention_results]
    
    moved_toward_abstract = abs(np.mean(intervened_sgps) - abstract_mean) < \
                            abs(np.mean(normal_sgps) - abstract_mean)
    
    results["comparison"] = {
        "abstract_mean_sgp_deg": float(abstract_mean),
        "spatial_normal_mean_sgp_deg": float(np.mean(normal_sgps)),
        "spatial_intervened_mean_sgp_deg": float(np.mean(intervened_sgps)),
        "intervention_moved_toward_abstract": moved_toward_abstract,
    }
    
    results["conclusion"] = {
        "causal": moved_toward_abstract,
        "explanation": (
            "Rotating spatial representations toward abstract stratum at L1 "
            "shifted downstream SGP toward abstract values — the sort IS causal."
            if moved_toward_abstract else
            "Rotation did not shift downstream SGP toward abstract values — "
            "the correlation may be driven by something other than L1 geometry."
        )
    }
    
    print(f"\n  Abstract baseline mean SGP: {abstract_mean:+.1f}°")
    print(f"  Spatial normal mean SGP:    {np.mean(normal_sgps):+.1f}°")
    print(f"  Spatial intervened mean SGP: {np.mean(intervened_sgps):+.1f}°")
    print(f"  Moved toward abstract: {moved_toward_abstract}")
    
    del model
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT 4: Untrained Baseline (subsumed into Experiment 1 at step0)
# ---------------------------------------------------------------------------

# The step0 checkpoint in Experiment 1 IS the untrained baseline.
# We just need to verify Φ ≈ 0 at that checkpoint.
# This is checked in the analysis section below.


# ---------------------------------------------------------------------------
# MAIN: Run all experiments, save results, produce summary
# ---------------------------------------------------------------------------

def run_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}")
    
    all_results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "model": "EleutherAI/pythia-70m",
            "description": "Stratified Geometric Phase symmetry-breaking battery",
            "theory": "https://github.com/zoedolan/Vybn/blob/main/Vybn_Mind/papers/stratified_geometric_phase.md",
        },
        "experiments": {}
    }
    
    # ---- Experiment 1: Training Dynamics ----
    print("\n" + "="*60)
    print("EXPERIMENT 1: TRAINING DYNAMICS")
    print("When does the sign flip appear?")
    print("="*60)
    
    try:
        all_results["experiments"]["training_dynamics"] = experiment_training_dynamics(device)
    except Exception as e:
        print(f"\nExperiment 1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["experiments"]["training_dynamics"] = {"error": str(e)}
    
    # ---- Experiment 2: Ablation ----
    print("\n" + "="*60)
    print("EXPERIMENT 2: ABLATION")
    print("Is the first block necessary?")
    print("="*60)
    
    try:
        all_results["experiments"]["ablation"] = experiment_ablation(device)
    except Exception as e:
        print(f"\nExperiment 2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["experiments"]["ablation"] = {"error": str(e)}
    
    # ---- Experiment 3: Causal Intervention ----
    print("\n" + "="*60)
    print("EXPERIMENT 3: CAUSAL INTERVENTION")
    print("Does rotating spatial into abstract change downstream processing?")
    print("="*60)
    
    try:
        all_results["experiments"]["causal_intervention"] = experiment_causal_intervention(device)
    except Exception as e:
        print(f"\nExperiment 3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["experiments"]["causal_intervention"] = {"error": str(e)}
    
    # ---- Analysis: Training dynamics summary ----
    print("\n" + "="*60)
    print("ANALYSIS: THE ORIGIN OF MEANING")
    print("="*60)
    
    td = all_results["experiments"].get("training_dynamics", {})
    if not isinstance(td, dict) or "error" in td:
        print("Training dynamics experiment failed — skipping analysis")
    else:
        # Find the first checkpoint where the sign flip appears
        sign_flip_appeared = None
        for ckpt in ["step0", "step1", "step2", "step4", "step8", "step16",
                      "step32", "step64", "step128", "step256", "step512",
                      "step1000", "step2000", "step5000", "step10000",
                      "step20000", "step50000", "step100000", "step143000"]:
            if ckpt in td and isinstance(td[ckpt], dict) and "has_sign_flip" in td[ckpt]:
                if td[ckpt]["has_sign_flip"]:
                    sign_flip_appeared = ckpt
                    break
        
        # Check untrained baseline (Experiment 4)
        step0_data = td.get("step0", {})
        if isinstance(step0_data, dict) and "order_parameter_deg" in step0_data:
            step0_phi = step0_data["order_parameter_deg"]
            step0_symmetric = step0_phi < 10.0  # less than 10° spread = symmetric
        else:
            step0_phi = None
            step0_symmetric = None
        
        # Final checkpoint
        final_data = td.get("step143000", {})
        if isinstance(final_data, dict) and "order_parameter_deg" in final_data:
            final_phi = final_data["order_parameter_deg"]
        else:
            final_phi = None
        
        all_results["analysis"] = {
            "sign_flip_first_appeared": sign_flip_appeared,
            "untrained_order_parameter_deg": step0_phi,
            "untrained_is_symmetric": step0_symmetric,
            "final_order_parameter_deg": final_phi,
            "symmetry_breaking_ratio": (
                final_phi / step0_phi if step0_phi and final_phi and step0_phi > 0
                else None
            ),
        }
        
        print(f"\n  Untrained (step0) order parameter Φ: {step0_phi}°")
        print(f"  Untrained symmetric: {step0_symmetric}")
        print(f"  Sign flip first appeared: {sign_flip_appeared}")
        print(f"  Final (step143000) order parameter Φ: {final_phi}°")
        
        if sign_flip_appeared:
            # Extract the step number
            step_num = int(sign_flip_appeared.replace("step", ""))
            tokens_seen = step_num * 2_097_152
            print(f"\n  *** THE ORIGIN OF MEANING ***")
            print(f"  The sign flip appeared at {sign_flip_appeared}")
            print(f"  ({tokens_seen:,} tokens = {tokens_seen/1e9:.2f}B tokens)")
            
            if step0_symmetric:
                print(f"  This is a genuine symmetry-breaking event:")
                print(f"  Φ went from {step0_phi:.1f}° (symmetric) to sign-flip at {sign_flip_appeared}")
    
    # ---- Save results ----
    output_path = Path("sgp_symmetry_breaking_results.json")
    
    # Custom JSON serializer for numpy types
    def np_serializer(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=np_serializer)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")
    
    return all_results


if __name__ == "__main__":
    results = run_all()
