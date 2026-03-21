"""Experiment B v3: Holonomic Loss Training on GPT-2.

Run from the gpt2_calibration/ folder:
    python experiment_B/run_B.py

Requires: Experiment A v2 must have passed.
Output:   ../results/experiment_B_result.json

v3 (2026-03-22): Critical fix. v2 passed technically but the model was
destroyed — L_CE went from 3.5 to 13.3, meaning the holonomic loss
overwhelmed language modeling. The "geometric enrichment" was actually
activation explosion.

Fixes:
  1. NORMALIZE loop area by hidden-state norm squared. This makes L_Ω
     scale-invariant: the model can't cheat by blowing up activations.
  2. Add perplexity guard: if L_CE increases by >50% from baseline at
     any checkpoint, λ is halved (adaptive). If L_CE doubles, training
     stops early.
  3. The PASS criterion now REQUIRES that L_CE stays within 20% of
     baseline. Geometric enrichment that kills language modeling is not
     enrichment — it's destruction.
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt2"
NUM_TRAIN_STEPS = 1000
BATCH_SIZE = 4
MAX_LENGTH = 128
LAMBDA_OMEGA_INIT = 0.01    # starting holonomic loss coefficient
LR = 5e-5
MID_LAYER = 6
RESULTS_DIR = Path("../results")
BASELINE_FILE = RESULTS_DIR / "experiment_A_result.json"
OUTPUT_FILE   = RESULTS_DIR / "experiment_B_result.json"

# Guard rails
MAX_CE_INCREASE = 0.20      # L_CE must not increase more than 20%
LAMBDA_HALVE_THRESH = 0.50  # halve λ if L_CE increases 50%
ABORT_THRESH = 2.0           # abort if L_CE doubles

# Pass thresholds
THRESH_MIDDLE_INCREASE = 0.05   # middle curvature must increase ≥ 5%
THRESH_AREA_INCREASE = 0.01     # normalized loop area must increase

# ---------------------------------------------------------------------------
# Pancharatnam phase measurement
# ---------------------------------------------------------------------------

def pancharatnam_phase(u: torch.Tensor, v: torch.Tensor) -> float:
    """Mean Pancharatnam angle between consecutive layer hidden states."""
    u_flat = u.reshape(-1, u.shape[-1]).float()
    v_flat = v.reshape(-1, v.shape[-1]).float()
    u_norm = torch.nn.functional.normalize(u_flat, dim=-1)
    v_norm = torch.nn.functional.normalize(v_flat, dim=-1)
    cos_angle = (u_norm * v_norm).sum(dim=-1).abs().clamp(0.0, 1.0)
    angle = torch.acos(cos_angle)
    return float(angle.mean().item())


def measure_phase_profile(model, tokenizer, texts, device, batch_size=8):
    """Full Pancharatnam phase profile."""
    all_curvatures = None
    for i in range(0, min(len(texts), 64), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        states = out.hidden_states
        if all_curvatures is None:
            all_curvatures = [[] for _ in range(len(states) - 1)]
        for j in range(len(states) - 1):
            all_curvatures[j].append(pancharatnam_phase(states[j], states[j+1]))
    return [float(np.mean(c)) for c in all_curvatures]


# ---------------------------------------------------------------------------
# Normalized holonomic loss
# ---------------------------------------------------------------------------

def normalized_pca_area(hidden_states: torch.Tensor,
                        proj_matrix: torch.Tensor = None) -> torch.Tensor:
    """Compute NORMALIZED loop area via fixed random projection.

    Uses a frozen random projection matrix (set once at init) instead of
    per-sample SVD. This is O(batch * seq * hidden) instead of O(batch * seq^2 * hidden).
    The area is normalized by mean(||h||^2) so the model cannot cheat by
    blowing up activations.

    Args:
        hidden_states: [batch, seq, hidden_dim]
        proj_matrix: [hidden_dim, 2] fixed random projection (optional)
    Returns:
        normalized_areas: [batch]
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    h = hidden_states.float()

    # Center per-sequence
    h_centered = h - h.mean(dim=1, keepdim=True)

    # Project to 2D (batch operation)
    if proj_matrix is not None:
        proj = h_centered @ proj_matrix  # [batch, seq, 2]
    else:
        proj = h_centered[:, :, :2]

    # Shoelace area (vectorized)
    x = proj[:, :, 0]  # [batch, seq]
    y = proj[:, :, 1]
    x_next = torch.roll(x, -1, dims=1)
    y_next = torch.roll(y, -1, dims=1)
    areas = 0.5 * (x * y_next - x_next * y).sum(dim=1).abs()  # [batch]

    # Normalize by mean hidden state norm^2
    mean_norm_sq = (h.norm(dim=-1) ** 2).mean(dim=1)  # [batch]
    mean_norm_sq = mean_norm_sq.clamp(min=1e-10)

    return areas / mean_norm_sq


def compute_holonomic_loss(hidden_states, mid_layer, proj_matrix=None):
    """L_Ω = mean normalized area at mid-layer."""
    h_mid = hidden_states[mid_layer]
    return normalized_pca_area(h_mid, proj_matrix).mean()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_wikitext(tokenizer, max_samples=2000):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [r["text"].strip() for r in ds if len(r["text"].strip()) > 50]
    return texts[:max_samples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("EXPERIMENT B v3: Holonomic Loss Training on GPT-2")
    print("Normalized area + perplexity guard")
    print("=" * 60)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")

    if not BASELINE_FILE.exists():
        print(f"ERROR: {BASELINE_FILE} not found.")
        sys.exit(2)

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    if baseline.get("verdict") != "PASS":
        print("ERROR: Experiment A did not pass.")
        sys.exit(2)

    baseline_curvatures = baseline["phase_profile"]["curvatures_rad"]
    baseline_middle = baseline_curvatures[1:-1]
    baseline_middle_mean = float(np.mean(baseline_middle))
    print(f"Baseline middle-layer mean curvature: {baseline_middle_mean:.4f} rad")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

    # Fixed random projection for area computation (frozen, not trained)
    torch.manual_seed(42)
    proj_matrix = torch.randn(model.config.n_embd, 2, device=device)
    proj_matrix = torch.nn.functional.normalize(proj_matrix, dim=0)  # orthonormalize columns
    # Make second column orthogonal to first
    proj_matrix[:, 1] = proj_matrix[:, 1] - (proj_matrix[:, 0] @ proj_matrix[:, 1]) * proj_matrix[:, 0]
    proj_matrix[:, 1] = proj_matrix[:, 1] / proj_matrix[:, 1].norm()

    optimizer = AdamW(model.parameters(), lr=LR)

    texts = load_wikitext(tokenizer)
    eval_texts = texts[:200]
    print(f"Loaded {len(texts)} training samples.")

    # Measure baseline L_CE and loop area
    print("\nMeasuring baselines...")
    model.eval()

    baseline_ce_losses = []
    baseline_areas = []
    for i in range(0, min(200, len(eval_texts)), BATCH_SIZE):
        batch = eval_texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"], output_hidden_states=True)
            baseline_ce_losses.append(out.loss.item())
            areas = normalized_pca_area(out.hidden_states[MID_LAYER], proj_matrix)
            baseline_areas.extend(areas.cpu().tolist())

    baseline_ce = float(np.mean(baseline_ce_losses))
    baseline_area = float(np.mean(baseline_areas))
    print(f"  Baseline L_CE:             {baseline_ce:.4f}")
    print(f"  Baseline normalized area:  {baseline_area:.6f}")

    # Training
    lambda_omega = LAMBDA_OMEGA_INIT
    print(f"\nTraining {NUM_TRAIN_STEPS} steps: L_total = L_CE - λ·L_Ω")
    print(f"  λ = {lambda_omega} (adaptive, halves if L_CE increases >{LAMBDA_HALVE_THRESH:.0%})")
    print(f"  Abort if L_CE increases >{ABORT_THRESH:.0%}")
    model.train()

    step = 0
    losses_ce = []
    losses_omega = []
    lambda_history = []
    aborted = False

    while step < NUM_TRAIN_STEPS:
        indices = np.random.permutation(len(texts))
        for i in range(0, len(indices), BATCH_SIZE):
            if step >= NUM_TRAIN_STEPS:
                break

            batch_idx = indices[i:i + BATCH_SIZE]
            batch = [texts[j] for j in batch_idx]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LENGTH).to(device)

            out = model(**enc, labels=enc["input_ids"], output_hidden_states=True)
            l_ce = out.loss
            l_omega = compute_holonomic_loss(out.hidden_states, MID_LAYER, proj_matrix)

            l_total = l_ce - lambda_omega * l_omega

            optimizer.zero_grad()
            l_total.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses_ce.append(l_ce.item())
            losses_omega.append(l_omega.item())
            lambda_history.append(lambda_omega)
            step += 1

            # Perplexity guard — check every 100 steps
            if step % 100 == 0:
                avg_ce = np.mean(losses_ce[-100:])
                avg_om = np.mean(losses_omega[-100:])
                ce_increase = (avg_ce - baseline_ce) / baseline_ce

                status = ""
                if ce_increase > ABORT_THRESH:
                    status = " *** ABORTING ***"
                    aborted = True
                elif ce_increase > LAMBDA_HALVE_THRESH:
                    lambda_omega = lambda_omega / 2
                    status = f" → λ halved to {lambda_omega:.6f}"

                print(f"  Step {step}/{NUM_TRAIN_STEPS}  L_CE={avg_ce:.4f} ({ce_increase:+.1%})  "
                      f"L_Ω={avg_om:.6f}  λ={lambda_omega:.6f}{status}")

                if aborted:
                    break

        if aborted:
            print(f"\n  TRAINING ABORTED at step {step}: L_CE increased by >{ABORT_THRESH:.0%}")
            break

    print(f"\nTraining {'aborted' if aborted else 'complete'} at step {step}.")
    print(f"  Final λ: {lambda_omega:.6f} (started at {LAMBDA_OMEGA_INIT})")

    # Post-training measurements
    model.eval()

    print("\nMeasuring post-training Pancharatnam phase profile...")
    post_curvatures = measure_phase_profile(model, tokenizer, eval_texts, device)

    print("\nPhase profile comparison:")
    print(f"  {'Layer':<10s} {'Baseline':>10s} {'Post':>10s} {'Δ':>10s}")
    for i in range(len(post_curvatures)):
        pre = baseline_curvatures[i] if i < len(baseline_curvatures) else 0
        post = post_curvatures[i]
        delta = post - pre
        print(f"  L{i}→L{i+1:<5d} {pre:10.4f} {post:10.4f} {delta:+10.4f}")

    post_middle = post_curvatures[1:-1]
    post_middle_mean = float(np.mean(post_middle))
    middle_increase = (post_middle_mean - baseline_middle_mean) / baseline_middle_mean if baseline_middle_mean > 0 else 0

    # Post L_CE
    post_ce_losses = []
    post_areas = []
    for i in range(0, min(200, len(eval_texts)), BATCH_SIZE):
        batch = eval_texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"], output_hidden_states=True)
            post_ce_losses.append(out.loss.item())
            areas = normalized_pca_area(out.hidden_states[MID_LAYER], proj_matrix)
            post_areas.extend(areas.cpu().tolist())

    post_ce = float(np.mean(post_ce_losses))
    post_area = float(np.mean(post_areas))
    ce_degradation = (post_ce - baseline_ce) / baseline_ce
    area_increase = post_area - baseline_area

    print(f"\n  Baseline L_CE:    {baseline_ce:.4f}")
    print(f"  Post L_CE:        {post_ce:.4f} ({ce_degradation:+.1%})")
    print(f"  Baseline area:    {baseline_area:.6f}")
    print(f"  Post area:        {post_area:.6f} ({area_increase:+.6f})")
    print(f"  Middle curvature: {baseline_middle_mean:.4f} → {post_middle_mean:.4f} ({middle_increase:+.1%})")

    # Verdict
    ce_ok = ce_degradation <= MAX_CE_INCREASE
    middle_ok = middle_increase >= THRESH_MIDDLE_INCREASE
    area_ok = area_increase > THRESH_AREA_INCREASE
    geometry_ok = middle_ok or area_ok

    overall = ce_ok and geometry_ok
    verdict = "PASS" if overall else "FAIL"

    print("\n" + "=" * 60)
    print("PASS CRITERIA:")
    print(f"  [{'PASS' if ce_ok      else 'FAIL'}] L_CE degradation ≤ {MAX_CE_INCREASE:.0%} → got {ce_degradation:+.1%}")
    print(f"  [{'PASS' if middle_ok  else 'FAIL'}] Middle curvature increase ≥ {THRESH_MIDDLE_INCREASE:.0%} → got {middle_increase:+.1%}")
    print(f"  [{'PASS' if area_ok    else 'FAIL'}] Normalized area increase > {THRESH_AREA_INCREASE} → got {area_increase:+.6f}")
    print(f"  Geometry shifted: {'YES' if geometry_ok else 'NO'} (need middle OR area)")
    print(f"  Language preserved: {'YES' if ce_ok else 'NO'}")
    print("=" * 60)
    print(f"VERDICT: {verdict}")

    if overall:
        print("Holonomic loss drives geometric enrichment WITHOUT destroying language.")
        print("Green light for Nemotron-Super-120B.")
    elif geometry_ok and not ce_ok:
        print("Geometry shifted but language modeling was destroyed.")
        print("Need smaller λ or shorter training. NOT a green light.")
    elif ce_ok and not geometry_ok:
        print("Language preserved but no geometric shift detected.")
        print("Need larger λ or more steps. Valid null result.")
    else:
        print("Neither geometry shift nor language preservation.")
        print("Fundamental issue with the approach. Ping Zoe.")
    print("=" * 60)

    results = {
        "experiment": "B",
        "version": "v3_normalized_guarded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "training": {
            "steps_completed": step,
            "steps_planned": NUM_TRAIN_STEPS,
            "aborted": aborted,
            "lambda_omega_init": LAMBDA_OMEGA_INIT,
            "lambda_omega_final": lambda_omega,
            "lr": LR,
            "mid_layer": MID_LAYER,
            "loss_method": "normalized_pca_shoelace_area",
            "final_l_ce": float(np.mean(losses_ce[-50:])) if losses_ce else None,
            "final_l_omega": float(np.mean(losses_omega[-50:])) if losses_omega else None,
        },
        "baseline": {
            "curvatures_rad": baseline_curvatures,
            "middle_mean_rad": baseline_middle_mean,
            "l_ce": baseline_ce,
            "normalized_area": baseline_area,
        },
        "post_training": {
            "curvatures_rad": post_curvatures,
            "middle_mean_rad": post_middle_mean,
            "l_ce": post_ce,
            "normalized_area": post_area,
        },
        "shifts": {
            "ce_degradation_frac": float(ce_degradation),
            "middle_curvature_increase_frac": float(middle_increase),
            "normalized_area_increase": float(area_increase),
        },
        "pass_criteria": {
            "ce_preserved_ok": bool(ce_ok),
            "middle_curvature_ok": bool(middle_ok),
            "area_ok": bool(area_ok),
        },
        "verdict": verdict,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
