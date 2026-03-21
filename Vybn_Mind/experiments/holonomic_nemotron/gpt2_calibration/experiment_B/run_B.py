"""Experiment B: Holonomic Loss Training on GPT-2.

Run from the gpt2_calibration/ folder:
    python experiment_B/run_B.py

Requires: Experiment A must have passed (checks for result file).
Output:   ../results/experiment_B_result.json
Verdict:  printed to terminal as PASS or FAIL

Fix (2026-03-21): Same two measurement bugs fixed as in run_A.py:
  1. shoelace_area returns SIGNED area (no .abs()).
  2. SGP post-training measurement uses signed mean phase shift, not
     absolute phase shift, for the phase-shift criterion.
  The holonomic loss itself (L_omega = mean loop area) intentionally
  uses the raw (unsigned) mean to maximize area magnitude — that is
  correct and unchanged.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt2"
NUM_TRAIN_STEPS = 1000
BATCH_SIZE = 4
MAX_LENGTH = 128
LAMBDA_OMEGA = 0.01       # holonomic loss coefficient
LR = 5e-5
MID_LAYER = 6             # mid-layer checkpoint for loop area (GPT-2 has 12)
RESULTS_DIR = Path("../results")
BASELINE_FILE = RESULTS_DIR / "experiment_A_result.json"
OUTPUT_FILE   = RESULTS_DIR / "experiment_B_result.json"

# Pass thresholds
THRESH_ENTROPY_SHIFT = 0.2   # sign-class entropy must increase by > 0.2 bits
THRESH_PHASE_SHIFT   = 0.01  # mean signed phase must increase (become less negative / more positive)

# ---------------------------------------------------------------------------
# Sort Probe (same as Experiment A, must match)
# ---------------------------------------------------------------------------
class SortProbe(nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )

    def forward(self, h):
        return self.proj(h)


def shoelace_area(traj):
    """SIGNED shoelace area — FIX: no .abs()."""
    x, y = traj[:, :, 0], traj[:, :, 1]
    xn, yn = torch.roll(x, -1, 1), torch.roll(y, -1, 1)
    return 0.5 * (x * yn - xn * y).sum(dim=1)   # signed, no .abs()


def sign_class_entropy(phases):
    pos = (phases > 0.01).mean()
    neg = (phases < -0.01).mean()
    neu = 1.0 - pos - neg
    p = np.array([pos, neg, neu])
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_wikitext(tokenizer, max_samples=2000):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [r["text"].strip() for r in ds if len(r["text"].strip()) > 50]
    return texts[:max_samples]

# ---------------------------------------------------------------------------
# Holonomic loss computation
# ---------------------------------------------------------------------------
def compute_holonomic_loss(hidden_states, mid_layer, probe, device):
    """Compute L_omega = mean |loop area| at mid-layer via sort probe.

    We take the mean of the *absolute* areas here because we want to
    MAXIMIZE loop area magnitude regardless of sign. This is intentional
    and correct — it is distinct from the calibration measurement in
    measure_sgp_post which uses signed area to detect topological shift.
    """
    h_mid = hidden_states[mid_layer]       # [batch, seq, hidden]
    traj  = probe(h_mid)                   # [batch, seq, 2]
    areas = shoelace_area(traj)            # [batch] signed
    return areas.abs().mean()              # maximize magnitude

# ---------------------------------------------------------------------------
# SGP measurement (post-training) — uses SIGNED area
# ---------------------------------------------------------------------------
def measure_sgp_post(model, tokenizer, texts, probe, device):
    model.eval()
    probe.eval()
    all_phases = []
    for i in range(0, min(200, len(texts)), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[1]       # block-0
            traj = probe(h)
            phases = shoelace_area(traj).cpu().numpy()   # signed
            all_phases.extend(phases.tolist())
    return np.array(all_phases)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("EXPERIMENT B: Holonomic Loss Training on GPT-2")
    print("(fixed: signed shoelace area for SGP measurement)")
    print("=" * 60)

    # Check that Experiment A passed
    if not BASELINE_FILE.exists():
        print(f"ERROR: {BASELINE_FILE} not found.")
        print("You must run Experiment A first: python experiment_A/run_A.py")
        sys.exit(2)

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    if baseline.get("verdict") != "PASS":
        print("ERROR: Experiment A did not pass. Do not run Experiment B.")
        sys.exit(2)

    baseline_entropy = baseline["sgp"]["sign_class_entropy_bits"]
    baseline_phase   = baseline["sgp"]["mean_phase"]
    print(f"Baseline entropy  : {baseline_entropy:.4f} bits")
    print(f"Baseline mean phase: {baseline_phase:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model and probe
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model     = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    hidden_dim = model.config.n_embd
    probe = SortProbe(hidden_dim=hidden_dim).to(device)

    # Both model and probe are trainable
    optimizer = AdamW(
        list(model.parameters()) + list(probe.parameters()),
        lr=LR,
    )

    # Load data
    texts = load_wikitext(tokenizer)
    print(f"Loaded {len(texts)} training samples.")

    # Training loop
    print(f"\nTraining for {NUM_TRAIN_STEPS} steps with L_total = L_CE - {LAMBDA_OMEGA} * L_omega...")
    model.train()
    probe.train()
    step      = 0
    epoch     = 0
    losses_ce    = []
    losses_omega = []

    while step < NUM_TRAIN_STEPS:
        epoch += 1
        indices = np.random.permutation(len(texts))
        for i in range(0, len(indices), BATCH_SIZE):
            if step >= NUM_TRAIN_STEPS:
                break
            batch_idx = indices[i : i + BATCH_SIZE]
            batch     = [texts[j] for j in batch_idx]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            ).to(device)

            # Forward pass
            out = model(
                **enc,
                labels=enc["input_ids"],
                output_hidden_states=True,
            )
            l_ce    = out.loss
            l_omega = compute_holonomic_loss(out.hidden_states, MID_LAYER, probe, device)

            # L_total = L_CE - lambda * L_omega  (subtract to MAXIMIZE loop area)
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
                print(f"  Step {step}/{NUM_TRAIN_STEPS}  L_CE={avg_ce:.4f}  L_omega={avg_om:.4f}")

    print("\nTraining complete.")

    # Re-measure SGP (signed area)
    print("\nMeasuring post-training SGP (signed area)...")
    eval_texts      = load_wikitext(tokenizer, max_samples=200)
    post_phases     = measure_sgp_post(model, tokenizer, eval_texts, probe, device)
    post_entropy    = sign_class_entropy(post_phases)
    post_mean_phase = float(np.mean(post_phases))

    print(f"  Post-training entropy  : {post_entropy:.4f} bits")
    print(f"  Post-training mean phase: {post_mean_phase:.4f}")

    # Compute shifts
    entropy_shift = post_entropy    - baseline_entropy
    phase_shift   = post_mean_phase - baseline_phase

    print(f"\n  Entropy shift   : {entropy_shift:+.4f} bits")
    print(f"  Mean phase shift: {phase_shift:+.4f}")

    # Evaluate
    check_entropy = entropy_shift > THRESH_ENTROPY_SHIFT
    check_phase   = phase_shift   > THRESH_PHASE_SHIFT
    overall       = check_entropy and check_phase
    verdict       = "PASS" if overall else "FAIL"

    print("\n" + "=" * 60)
    print("PASS CRITERIA:")
    print(f"  [{'PASS' if check_entropy else 'FAIL'}] Entropy shift > +{THRESH_ENTROPY_SHIFT} -> got {entropy_shift:+.4f}")
    print(f"  [{'PASS' if check_phase   else 'FAIL'}] Phase shift > +{THRESH_PHASE_SHIFT} -> got {phase_shift:+.4f}")
    print("=" * 60)
    print(f"VERDICT: {verdict}")
    if overall:
        print("Holonomic loss drives SGP enrichment on GPT-2.")
        print("Green light to replicate on Nemotron-Super-120B.")
    else:
        print("Holonomic loss did NOT shift SGP distribution.")
        print("This is a valid null result. Save the JSON and ping Zoe.")
    print("=" * 60)

    # Save
    results = {
        "experiment": "B",
        "model": MODEL_NAME,
        "fix_version": "signed-shoelace",
        "training": {
            "steps":        NUM_TRAIN_STEPS,
            "lambda_omega": LAMBDA_OMEGA,
            "lr":           LR,
            "mid_layer":    MID_LAYER,
            "final_l_ce":   float(np.mean(losses_ce[-50:])),
            "final_l_omega": float(np.mean(losses_omega[-50:])),
        },
        "baseline": {
            "entropy_bits": baseline_entropy,
            "mean_phase":   baseline_phase,
        },
        "post_training": {
            "entropy_bits": post_entropy,
            "mean_phase":   post_mean_phase,
            "phases_summary": {
                "positive_frac": float((post_phases > 0.01).mean()),
                "negative_frac": float((post_phases < -0.01).mean()),
                "neutral_frac":  float(((post_phases >= -0.01) & (post_phases <= 0.01)).mean()),
            },
        },
        "shifts": {
            "entropy_shift_bits": entropy_shift,
            "mean_phase_shift":   phase_shift,
        },
        "pass_criteria": {
            "entropy_shift_ok": check_entropy,
            "phase_shift_ok":   check_phase,
        },
        "verdict": verdict,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
