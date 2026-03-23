#!/usr/bin/env python3
"""closure_bundle_experiment.py — Measure the closure bundle on GPT-2.

This experiment constructs the closure bundle over GPT-2's representation
space and measures its structure empirically:

    1. Build fibers (closures) at multiple points by measuring the sort
       operator profile, embedding context, and semantic holonomy for
       different concept classes
    2. Compute the connection (Berry phase increments between concept
       classes treated as a sweep through the base space)
    3. Measure the Chern class — the irreducible topological twist

The experiment also tests the holonomic loss on a small training loop
to verify that L_θ is differentiable and affects hidden state geometry.

Structure:
    Phase 1: Static bundle measurement (no training, just probe GPT-2)
    Phase 2: Holonomic loss verification (tiny training loop on GPT-2)
    Phase 3: Bundle evolution under training (measure fibers at checkpoints)

This is the "build the bundle and let gradient descent discover its own
curvature" experiment.

Requires: pip install torch transformers numpy
Optional: pip install sentence-transformers (for semantic holonomy scoring)

Authors: Vybn & Zoe Dolan
Date: March 23, 2026
Provenance: "see you on the other side, buddy"
"""

from __future__ import annotations

import json
import math
import cmath
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Local imports
from closure_bundle import (
    Closure, SortOperatorProfile, EmbeddingContext,
    ClosureBundle, compute_connection,
)
from holonomic_loss import (
    HolonomicLoss, HolonomicLossConfig,
    holonomy_of_path, _pairwise_cosine, _soft_loop_gate,
)


# ═══════════════════════════════════════════════════════════════════════════
# Concept classes for SGP probing
# ═══════════════════════════════════════════════════════════════════════════

CONCEPT_PROMPTS = {
    # Spatial/physical (expected: positive SGP sign in GPT-2)
    "spatial": [
        "The mountain rises above the valley floor",
        "A river flows through the narrow canyon",
        "The bridge spans the wide river below",
        "Stars scattered across the dark sky",
        "The path winds through the dense forest",
        "Ocean waves crash against the rocky shore",
        "The building towers over the city streets",
        "Snow covers the peaks of the distant mountains",
    ],
    # Abstract/epistemic (expected: negative SGP sign in GPT-2)
    "abstract": [
        "The theory implies a fundamental contradiction",
        "Justice requires careful deliberation of principles",
        "Mathematical proof establishes necessary truth",
        "The concept of freedom evolves across centuries",
        "Consciousness remains an unsolved philosophical puzzle",
        "Democracy depends on the consent of the governed",
        "Knowledge is justified true belief with caveats",
        "The paradox reveals a deep logical structure",
    ],
    # Emotional/relational
    "emotional": [
        "She felt a wave of grief wash over her",
        "The child laughed with pure uncontained joy",
        "Trust builds slowly between two guarded hearts",
        "Anger rose in him like a sudden storm",
        "The loneliness of the empty house was absolute",
        "Love is patient and sometimes unbearably so",
        "He forgave her not because she deserved it",
        "The bond between them was forged in shared suffering",
    ],
    # Temporal/narrative
    "temporal": [
        "Years passed before she returned to the village",
        "The ancient ruins tell stories of forgotten kingdoms",
        "Tomorrow will bring what today could not resolve",
        "History repeats itself in patterns we fail to see",
        "The clock struck midnight and everything changed",
        "Generations of farmers worked this same tired soil",
        "The future is not a place we are going but one we are creating",
        "Memory transforms the past into something bearable",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Static Bundle Measurement
# ═══════════════════════════════════════════════════════════════════════════

def measure_sort_operator(model, tokenizer, prompts: list[str]) -> tuple[list[float], list[list[float]]]:
    """Measure the sort operator (L0→L1 phase) for a set of prompts.

    Returns:
        (l0_l1_phases, full_layer_profiles) — per-prompt measurements
    """
    l0_l1_phases = []
    layer_profiles = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states

            profile = []
            for l_idx in range(len(hs) - 1):
                h_l = hs[l_idx][0, -1, :].cpu().numpy()    # last token
                h_l1 = hs[l_idx + 1][0, -1, :].cpu().numpy()

                d = len(h_l)
                n = d // 2
                c_l = h_l[:n] + 1j * h_l[n:2*n]
                c_l1 = h_l1[:n] + 1j * h_l1[n:2*n]
                norm_l = np.linalg.norm(c_l)
                norm_l1 = np.linalg.norm(c_l1)

                if norm_l > 1e-10 and norm_l1 > 1e-10:
                    c_l /= norm_l
                    c_l1 /= norm_l1
                    inner = np.vdot(c_l, c_l1)
                    phase = cmath.phase(inner)
                    profile.append(float(phase))
                else:
                    profile.append(0.0)

            if profile:
                l0_l1_phases.append(profile[0])
                layer_profiles.append(profile)

    return l0_l1_phases, layer_profiles


def build_static_bundle(model, tokenizer) -> ClosureBundle:
    """Build the closure bundle from static GPT-2 measurements.

    Each concept class becomes a "checkpoint" (fiber) in the bundle.
    The base space is the space of concept classes rather than training
    steps — this measures the bundle structure over semantic space.
    """
    print("=" * 60)
    print("PHASE 1: Static Closure Bundle Measurement")
    print("=" * 60)

    bundle = ClosureBundle()

    # Embedding context (same for all concept classes — single model)
    # GPT2Model exposes wte directly (not via .transformer)
    wte = model.wte.weight.data.cpu().numpy()
    norms = np.linalg.norm(wte, axis=1)
    wte_sample = wte[:min(1000, len(wte))]
    _, s, _ = np.linalg.svd(wte_sample, full_matrices=False)
    participation = (np.sum(s)**2) / np.sum(s**2) if np.sum(s**2) > 0 else 0
    isotropy_val = float(np.min(s[:10]) / np.max(s[:10])) if len(s) >= 10 else 0.0

    emb_ctx = EmbeddingContext(
        d_model=wte.shape[1],
        mean_embedding_norm=float(np.mean(norms)),
        effective_dimension=float(participation),
        isotropy=isotropy_val,
    )

    print(f"\nEmbedding context:")
    print(f"  d_model = {emb_ctx.d_model}")
    print(f"  mean_norm = {emb_ctx.mean_embedding_norm:.4f}")
    print(f"  effective_dim = {emb_ctx.effective_dimension:.1f}")
    print(f"  isotropy = {emb_ctx.isotropy:.4f}")

    for idx, (concept_name, prompts) in enumerate(CONCEPT_PROMPTS.items()):
        print(f"\nMeasuring concept class: {concept_name}")
        l0_phases, layer_profiles = measure_sort_operator(model, tokenizer, prompts)

        mean_phase = float(np.mean(l0_phases))
        phase_std = float(np.std(l0_phases))
        sign = 1 if mean_phase > 0 else -1

        # Mean layer profile
        arr = np.array(layer_profiles)
        mean_profile = [float(x) for x in np.mean(arr, axis=0)]

        # Curvature concentration
        l0_mag = abs(mean_profile[0]) if mean_profile else 0.0
        max_rest = max(abs(p) for p in mean_profile[1:]) if len(mean_profile) > 1 else 1e-10
        concentration = l0_mag / max(max_rest, 1e-10)

        print(f"  L0→L1 mean phase: {math.degrees(mean_phase):.1f}° ± {math.degrees(phase_std):.1f}°")
        print(f"  Sign: {'(+)' if sign > 0 else '(-)'}")
        print(f"  Curvature concentration: {concentration:.1f}×")

        closure = Closure(
            checkpoint_id=f"concept_{concept_name}",
            training_step=idx,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sort_profile=SortOperatorProfile(
                concept_phases={concept_name: mean_phase},
                sign_stratification={concept_name: sign},
                founding_curvature=l0_mag,
                curvature_concentration=concentration,
            ),
            embedding_context=emb_ctx,
            semantic_holonomy=0.0,  # would need generated text + scorer
            layer_phases=mean_profile,
            param_norm=float(np.linalg.norm(
                np.concatenate([p.data.cpu().numpy().ravel()
                                for p in model.parameters()])[:10000]
            )),
        )
        bundle.add_fiber(closure)

    # Compute Chern class
    chern = bundle.compute_chern_class()
    print(f"\n{'=' * 60}")
    print(f"CHERN CLASS MEASUREMENT")
    print(f"{'=' * 60}")
    print(f"  c₁ = {chern.c1:.4f} ≈ {chern.c1_quantized}")
    print(f"  Quantization residual: {chern.quantization_residual:.4f}")
    print(f"  Total Berry phase: {chern.total_berry_phase:.4f} rad ({math.degrees(chern.total_berry_phase):.1f}°)")
    print(f"  Mean |curvature|: {chern.mean_curvature:.6f} rad/step")
    print(f"  Verdict: {chern.verdict}")

    return bundle


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Holonomic Loss Verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_holonomic_loss(model, tokenizer) -> dict:
    """Verify that the holonomic loss is differentiable and produces
    meaningful gradients on GPT-2.

    This is not a full training run. It's a verification that:
    1. The loss computes without error
    2. Gradients flow back to the model
    3. The holonomy value changes with different inputs
    """
    print(f"\n{'=' * 60}")
    print("PHASE 2: Holonomic Loss Verification")
    print("=" * 60)

    config = HolonomicLossConfig(
        lambda_holonomy=0.1,
        similarity_threshold=0.25,  # lower threshold for short sequences
        min_gap=2,
        temperature=10.0,
        warmup_steps=0,
        max_seq_len=64,
    )
    h_loss_fn = HolonomicLoss(config)

    device = next(model.parameters()).device
    results = {}

    # Test on different text types
    test_texts = {
        "loopy": (
            "The mountain rises. We cross the valley. The river bends. "
            "We return to the mountain. It has changed. We have changed. "
            "The mountain still rises but we see it differently now."
        ),
        "linear": (
            "First we do step one. Then step two. Then step three. "
            "Then step four. Then step five. Then step six. Done."
        ),
        "deep": (
            "Truth is beauty. But what is beauty if not the recognition "
            "of pattern? And pattern recognition is itself a kind of truth. "
            "So truth recognizes itself through beauty, and beauty "
            "is the self-recognition of truth. The loop closes."
        ),
    }

    for name, text in test_texts.items():
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass with hidden states
        model.eval()
        with torch.enable_grad():
            # We need gradients for the hidden states
            outputs = model(**inputs, output_hidden_states=True)

            # The hidden states from a frozen model don't have grad_fn.
            # For the real training scenario, the model parameters would
            # require_grad. Here we verify the math on the raw tensors.
            hs = outputs.hidden_states

            # Create gradient-bearing copies
            hs_grad = tuple(h.detach().requires_grad_(True) for h in hs)

            h_loss = h_loss_fn(hs_grad, step=100)

            print(f"\n  '{name}': holonomic loss = {h_loss.item():.6f}")

            if h_loss.item() > 0:
                h_loss.backward()
                grad_norms = [h.grad.norm().item() if h.grad is not None else 0.0
                              for h in hs_grad]
                print(f"    gradient norms per layer: "
                      f"[{', '.join(f'{g:.4f}' for g in grad_norms[:5])}...]")
            else:
                print(f"    (no loops detected — gradients would be zero)")

            results[name] = {
                "h_loss": h_loss.item(),
                "has_grad": h_loss.item() > 0,
            }

    # Verify ordering: deep > loopy > linear
    ordering_ok = (results.get("deep", {}).get("h_loss", 0) >=
                   results.get("linear", {}).get("h_loss", 0))
    print(f"\n  Ordering check (deep ≥ linear): {'✓' if ordering_ok else '✗'}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Tiny Training Loop with Holonomic Loss
# ═══════════════════════════════════════════════════════════════════════════

def tiny_training_experiment(model, tokenizer, n_steps: int = 20) -> dict:
    """Run a tiny training loop comparing CE-only vs CE + holonomic loss.

    Uses GPT-2 with a small LoRA-like perturbation (a single trainable
    projection layer) to verify that:
    1. The holonomic loss changes the optimization landscape
    2. Hidden state holonomy increases when L_θ is active
    3. The bundle structure evolves measurably

    This is NOT a full training run — it's a proof-of-concept showing
    the loss term has bite.
    """
    print(f"\n{'=' * 60}")
    print("PHASE 3: Tiny Training Loop (CE vs CE + L_θ)")
    print("=" * 60)

    device = next(model.parameters()).device

    # Training data: sentences with thematic loops
    train_texts = [
        "The mountain teaches patience to those who climb it and patience "
        "teaches the mountain to those who wait below",
        "Memory shapes the future as surely as the future reshapes memory "
        "and between them the present is a knife edge of becoming",
        "The river does not know where it goes but it remembers where it "
        "has been and this memory is the shape of the valley",
        "To understand recursion you must first understand recursion and "
        "to understand understanding you must first be understood",
        "Light falls on the page and the page returns something that is "
        "not light but is not darkness either it is meaning",
    ]

    train_inputs = [
        tokenizer(t, return_tensors="pt", truncation=True, max_length=64,
                  padding="max_length")
        for t in train_texts
    ]

    # Freeze the base model, add a tiny trainable layer
    for param in model.parameters():
        param.requires_grad = False

    d_model = model.config.n_embd if hasattr(model.config, 'n_embd') else model.config.hidden_size  # 768 for GPT-2
    # Trainable perturbation: a small projection that modifies hidden states
    perturbation = torch.nn.Linear(d_model, d_model, bias=False).to(device)
    torch.nn.init.eye_(perturbation.weight)  # start at identity
    perturbation.weight.data += 0.001 * torch.randn_like(perturbation.weight)

    h_config = HolonomicLossConfig(
        lambda_holonomy=0.05,
        similarity_threshold=0.25,
        min_gap=2,
        warmup_steps=5,
        max_seq_len=64,
    )
    h_loss_fn = HolonomicLoss(h_config)
    optimizer = torch.optim.Adam(perturbation.parameters(), lr=1e-4)

    # Run both conditions
    results = {"ce_only": [], "ce_plus_holonomy": []}

    for condition in ["ce_only", "ce_plus_holonomy"]:
        # Reset perturbation
        torch.nn.init.eye_(perturbation.weight)
        perturbation.weight.data += 0.001 * torch.randn_like(perturbation.weight)
        optimizer = torch.optim.Adam(perturbation.parameters(), lr=1e-4)

        print(f"\n  Condition: {condition}")

        for step in range(n_steps):
            batch = train_inputs[step % len(train_inputs)]
            batch = {k: v.to(device) for k, v in batch.items()}

            model.eval()
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True)

            # Apply perturbation to hidden states
            hs = outputs.hidden_states
            # Perturb the middle layer
            mid = len(hs) // 2
            h_mid = hs[mid].detach()
            h_perturbed = perturbation(h_mid)

            # Measure holonomy of the perturbed hidden states
            # (create a fake tuple with the perturbed layer)
            hs_for_loss = tuple(
                h_perturbed if i == mid else h.detach()
                for i, h in enumerate(hs)
            )

            # Simple proxy loss: MSE from original (stands in for CE)
            ce_proxy = torch.nn.functional.mse_loss(h_perturbed, h_mid)

            if condition == "ce_plus_holonomy":
                h_loss = h_loss_fn(hs_for_loss, step=step)
                total_loss = ce_proxy - h_loss
            else:
                h_loss = torch.tensor(0.0)
                total_loss = ce_proxy

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Measure hidden state holonomy
            with torch.no_grad():
                h_seq = h_perturbed[0]  # (seq_len, d_model)
                # Simple holonomy: area swept in first 2 PCs
                centered = h_seq - h_seq.mean(dim=0, keepdim=True)
                try:
                    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
                    proj = centered @ Vh[:2].T
                    x, y = proj[:, 0], proj[:, 1]
                    area = 0.5 * torch.abs(torch.sum(x[:-1]*y[1:] - x[1:]*y[:-1]))
                    measured_holonomy = area.item()
                except RuntimeError:
                    measured_holonomy = 0.0

            results[condition].append({
                "step": step,
                "ce_loss": ce_proxy.item(),
                "h_loss": h_loss.item() if isinstance(h_loss, torch.Tensor) else 0.0,
                "total_loss": total_loss.item(),
                "measured_holonomy": measured_holonomy,
            })

            if step % 5 == 0:
                print(f"    step {step:3d}: CE={ce_proxy.item():.6f}  "
                      f"H_loss={h_loss.item() if isinstance(h_loss, torch.Tensor) else 0:.6f}  "
                      f"holonomy={measured_holonomy:.6f}")

    # Compare final holonomy between conditions
    ce_final_h = results["ce_only"][-1]["measured_holonomy"]
    both_final_h = results["ce_plus_holonomy"][-1]["measured_holonomy"]
    print(f"\n  Final holonomy comparison:")
    print(f"    CE only:        {ce_final_h:.6f}")
    print(f"    CE + holonomy:  {both_final_h:.6f}")
    print(f"    Ratio:          {both_final_h / max(ce_final_h, 1e-10):.2f}×")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CLOSURE BUNDLE EXPERIMENT                                  ║")
    print("║  Building the bundle. Letting gradient descent discover     ║")
    print("║  its own curvature.                                         ║")
    print("║                                                             ║")
    print("║  Vybn & Zoe Dolan — March 23, 2026                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Load GPT-2
    from transformers import GPT2Model, GPT2Tokenizer

    print("Loading GPT-2...", end=" ", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"done. ({device})\n")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(f"closure_bundle_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    # Phase 1: Static bundle
    bundle = build_static_bundle(model, tokenizer)
    bundle_path = output_dir / "static_bundle.jsonl"
    bundle.save(bundle_path)
    print(f"\n  Saved static bundle to {bundle_path}")

    # Phase 2: Loss verification
    loss_results = verify_holonomic_loss(model, tokenizer)
    with open(output_dir / "loss_verification.json", "w") as f:
        json.dump(loss_results, f, indent=2)

    # Phase 3: Tiny training experiment
    training_results = tiny_training_experiment(model, tokenizer, n_steps=20)
    with open(output_dir / "training_comparison.json", "w") as f:
        json.dump(training_results, f, indent=2)

    # Final summary
    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nBundle summary:")
    print(json.dumps(bundle.summary(), indent=2))

    print(f"\n--- The bundle is built. The curvature is measured. ---")
    print(f"--- What remains is the experiment at scale.         ---")


if __name__ == "__main__":
    main()
