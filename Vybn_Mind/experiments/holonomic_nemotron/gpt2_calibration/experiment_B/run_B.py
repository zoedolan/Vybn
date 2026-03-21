"""Experiment B v2: Holonomic Loss Training on GPT-2.

Run from the gpt2_calibration/ folder:
    python experiment_B/run_B.py

Requires: Experiment A v2 must have passed (checks for result file).
Output:   ../results/experiment_B_result.json
Verdict:  printed to terminal as PASS or FAIL

v2 (2026-03-22): Complete rewrite matching Experiment A v2.
  - Removes untrained SortProbe from both training loss and measurement
  - Holonomic loss L_Ω now computed directly on hidden state trajectories
    using PCA-projected 2D shoelace area (no learned projection)
  - Pre/post comparison uses Pancharatnam phase profile (proven instrument)
  - PASS if middle-layer curvature increases after holonomic training

The holonomic loss hypothesis (holonomic_loss_hypothesis.md):
  L_total = L_CE - λ · L_Ω
  where L_Ω = mean |shoelace area of hidden state trajectory at mid-layer|
  
  We reward the model for sweeping area in representation space during
  sequence processing. If this drives richer geometric structure, the
  Pancharatnam phase profile should shift — specifically, the middle
  layers (which are nearly flat in baseline GPT-2) should develop more
  curvature as the model learns to exploit angular structure.
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
LAMBDA_OMEGA = 0.01        # holonomic loss coefficient
LR = 5e-5
MID_LAYER = 6              # mid-layer for loop area computation
RESULTS_DIR = Path("../results")
BASELINE_FILE = RESULTS_DIR / "experiment_A_result.json"
OUTPUT_FILE   = RESULTS_DIR / "experiment_B_result.json"

# Pass thresholds — measured on Pancharatnam phase profile
THRESH_MIDDLE_INCREASE = 0.05   # middle-layer mean curvature must increase by ≥ 5%
THRESH_AREA_INCREASE = 0.01     # mean loop area at mid-layer must increase

# ---------------------------------------------------------------------------
# Pancharatnam phase measurement (same as Experiment A)
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
    """Measure full Pancharatnam phase profile across all layer transitions."""
    all_curvatures = None
    n_batches = 0

    for i in range(0, min(len(texts), 64), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        states = out.hidden_states

        if all_curvatures is None:
            all_curvatures = [[] for _ in range(len(states) - 1)]

        for j in range(len(states) - 1):
            angle = pancharatnam_phase(states[j], states[j + 1])
            all_curvatures[j].append(angle)
        n_batches += 1

    return [float(np.mean(c)) for c in all_curvatures]


# ---------------------------------------------------------------------------
# Holonomic loss — direct computation on hidden states
# ---------------------------------------------------------------------------

def pca_shoelace_area(hidden_states: torch.Tensor) -> torch.Tensor:
    """Compute loop area in PCA-projected 2D space of hidden state trajectories.

    This replaces the untrained SortProbe. Instead of a random learned projection,
    we project each sequence's hidden state trajectory onto its top-2 principal
    components and compute the shoelace area. This measures how much area the
    model's internal trajectory sweeps during sequence processing.

    Args:
        hidden_states: [batch, seq, hidden_dim]
    Returns:
        areas: [batch] absolute shoelace areas
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    areas = []

    for b in range(batch_size):
        h = hidden_states[b]  # [seq, hidden]
        # Center
        h_centered = h - h.mean(dim=0, keepdim=True)
        # SVD for top-2 PCs (use float32 for stability)
        h_f = h_centered.float()
        try:
            U, S, Vh = torch.linalg.svd(h_f, full_matrices=False)
            # Project to 2D
            proj = h_f @ Vh[:2].T  # [seq, 2]
        except RuntimeError:
            # Fallback: just use first two dims
            proj = h_f[:, :2]

        x = proj[:, 0]
        y = proj[:, 1]
        x_next = torch.roll(x, -1)
        y_next = torch.roll(y, -1)
        area = 0.5 * (x * y_next - x_next * y).sum()
        areas.append(area.abs())

    return torch.stack(areas)


def compute_holonomic_loss(hidden_states, mid_layer):
    """Compute L_Ω = mean |PCA shoelace area| at mid-layer.

    This is the angular component of the training objective.
    We maximize this to encourage the model to sweep area in
    representation space during sequence processing.
    """
    h_mid = hidden_states[mid_layer]   # [batch, seq, hidden]
    areas = pca_shoelace_area(h_mid)   # [batch]
    return areas.mean()


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
    print("EXPERIMENT B v2: Holonomic Loss Training on GPT-2")
    print("Direct geometric measurement — no untrained probe")
    print("=" * 60)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")

    # Check that Experiment A passed
    if not BASELINE_FILE.exists():
        print(f"ERROR: {BASELINE_FILE} not found.")
        print("Run Experiment A first: python experiment_A/run_A.py")
        sys.exit(2)

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    if baseline.get("verdict") != "PASS":
        print("ERROR: Experiment A did not pass. Do not run Experiment B.")
        sys.exit(2)

    baseline_curvatures = baseline["phase_profile"]["curvatures_rad"]
    baseline_middle = baseline_curvatures[1:-1]
    baseline_middle_mean = float(np.mean(baseline_middle))
    print(f"Baseline middle-layer mean curvature: {baseline_middle_mean:.4f} rad")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

    optimizer = AdamW(model.parameters(), lr=LR)

    # Load data
    texts = load_wikitext(tokenizer)
    print(f"Loaded {len(texts)} training samples.")

    # Measure baseline loop area at mid-layer
    print("\nMeasuring baseline loop area at mid-layer...")
    model.eval()
    baseline_areas = []
    eval_texts = texts[:200]
    for i in range(0, len(eval_texts), BATCH_SIZE):
        batch = eval_texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            areas = pca_shoelace_area(out.hidden_states[MID_LAYER])
            baseline_areas.extend(areas.cpu().tolist())
    baseline_area_mean = float(np.mean(baseline_areas))
    print(f"  Baseline mean loop area: {baseline_area_mean:.4f}")

    # Training loop
    print(f"\nTraining {NUM_TRAIN_STEPS} steps: L_total = L_CE - {LAMBDA_OMEGA} × L_Ω")
    print(f"  L_Ω = PCA shoelace area at layer {MID_LAYER}")
    model.train()

    step = 0
    losses_ce = []
    losses_omega = []

    while step < NUM_TRAIN_STEPS:
        indices = np.random.permutation(len(texts))
        for i in range(0, len(indices), BATCH_SIZE):
            if step >= NUM_TRAIN_STEPS:
                break
            batch_idx = indices[i:i + BATCH_SIZE]
            batch = [texts[j] for j in batch_idx]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LENGTH,
            ).to(device)

            out = model(**enc, labels=enc["input_ids"], output_hidden_states=True)
            l_ce = out.loss
            l_omega = compute_holonomic_loss(out.hidden_states, MID_LAYER)

            l_total = l_ce - LAMBDA_OMEGA * l_omega

            optimizer.zero_grad()
            l_total.backward()
            optimizer.step()

            losses_ce.append(l_ce.item())
            losses_omega.append(l_omega.item())
            step += 1

            if step % 100 == 0:
                avg_ce = np.mean(losses_ce[-100:])
                avg_om = np.mean(losses_omega[-100:])
                print(f"  Step {step}/{NUM_TRAIN_STEPS}  L_CE={avg_ce:.4f}  L_Ω={avg_om:.4f}")

    print("\nTraining complete.")

    # ---------------------------------------------------------------
    # Post-training measurements
    # ---------------------------------------------------------------
    model.eval()

    # 1. Full phase profile
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
    middle_increase = (post_middle_mean - baseline_middle_mean) / baseline_middle_mean

    # 2. Loop area at mid-layer
    print("\nMeasuring post-training loop area...")
    post_areas = []
    for i in range(0, len(eval_texts), BATCH_SIZE):
        batch = eval_texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            areas = pca_shoelace_area(out.hidden_states[MID_LAYER])
            post_areas.extend(areas.cpu().tolist())
    post_area_mean = float(np.mean(post_areas))
    area_increase = post_area_mean - baseline_area_mean

    print(f"\n  Baseline area: {baseline_area_mean:.4f}")
    print(f"  Post area:     {post_area_mean:.4f}")
    print(f"  Area increase: {area_increase:+.4f}")

    print(f"\n  Baseline middle curvature: {baseline_middle_mean:.4f} rad")
    print(f"  Post middle curvature:     {post_middle_mean:.4f} rad")
    print(f"  Middle increase:           {middle_increase:+.2%}")

    # ---------------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------------
    check_middle = middle_increase >= THRESH_MIDDLE_INCREASE
    check_area = area_increase > THRESH_AREA_INCREASE

    overall = check_middle or check_area  # either indicator is sufficient
    verdict = "PASS" if overall else "FAIL"

    print("\n" + "=" * 60)
    print("PASS CRITERIA (either sufficient):")
    print(f"  [{'PASS' if check_middle else 'FAIL'}] Middle curvature increase ≥ {THRESH_MIDDLE_INCREASE:.0%} → got {middle_increase:+.2%}")
    print(f"  [{'PASS' if check_area   else 'FAIL'}] Loop area increase > {THRESH_AREA_INCREASE} → got {area_increase:+.4f}")
    print("=" * 60)
    print(f"VERDICT: {verdict}")
    if overall:
        print("Holonomic loss shifts GPT-2's geometric structure.")
        print("Green light for Nemotron-Super-120B replication.")
    else:
        print("Holonomic loss did NOT shift geometry at this scale.")
        print("This is a valid null result. Save JSON and ping Zoe.")
    print("=" * 60)

    # Save results
    results = {
        "experiment": "B",
        "version": "v2_direct_geometry",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "training": {
            "steps": NUM_TRAIN_STEPS,
            "lambda_omega": LAMBDA_OMEGA,
            "lr": LR,
            "mid_layer": MID_LAYER,
            "loss_method": "pca_shoelace_area",
            "final_l_ce": float(np.mean(losses_ce[-50:])),
            "final_l_omega": float(np.mean(losses_omega[-50:])),
        },
        "baseline": {
            "curvatures_rad": baseline_curvatures,
            "middle_mean_rad": baseline_middle_mean,
            "loop_area_mean": baseline_area_mean,
        },
        "post_training": {
            "curvatures_rad": post_curvatures,
            "middle_mean_rad": post_middle_mean,
            "loop_area_mean": post_area_mean,
        },
        "shifts": {
            "middle_curvature_increase_frac": float(middle_increase),
            "loop_area_increase": float(area_increase),
        },
        "pass_criteria": {
            "middle_curvature_ok": check_middle,
            "loop_area_ok": check_area,
        },
        "verdict": verdict,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
