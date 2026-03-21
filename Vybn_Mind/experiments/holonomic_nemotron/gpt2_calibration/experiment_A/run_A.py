"""Experiment A: Probe Calibration on GPT-2.

Run from the gpt2_calibration/ folder:
    python experiment_A/run_A.py

This script validates the sort probe instrument against known GPT-2 geometry.
It MUST pass before running Experiment B.

Output: ../results/experiment_A_result.json
Verdict: printed to terminal as PASS or FAIL
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt2"          # 117M, downloads automatically
NUM_SAMPLES = 200             # wikitext-2 test samples
BATCH_SIZE = 8
MAX_LENGTH = 128
RESULTS_DIR = Path("../results")
OUTPUT_FILE = RESULTS_DIR / "experiment_A_result.json"

# Pass thresholds (from known compute_sort_degree.py result: deg(S) = 0)
THRESH_MEAN_PHASE = 0.05      # mean loop area should be near zero
THRESH_ENTROPY = 0.5          # sign class entropy should be low (bits)
THRESH_CURVATURE_RATIO = 3.0  # block-0 curvature >= 3x any later block

# ---------------------------------------------------------------------------
# Sort Probe module
# ---------------------------------------------------------------------------
class SortProbe(nn.Module):
    """Lightweight MLP: hidden_dim → 512 → 2D phase space."""
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[batch, seq, hidden] → [batch, seq, 2]"""
        return self.proj(h)


def shoelace_area(trajectory: torch.Tensor) -> torch.Tensor:
    """Signed loop area per sequence via shoelace formula.

    Args:
        trajectory: [batch, seq, 2]
    Returns:
        area: [batch]  (absolute value of signed area)
    """
    x = trajectory[:, :, 0]
    y = trajectory[:, :, 1]
    x_next = torch.roll(x, -1, dims=1)
    y_next = torch.roll(y, -1, dims=1)
    area = 0.5 * (x * y_next - x_next * y).sum(dim=1).abs()
    return area


def sign_class_entropy(phases: np.ndarray) -> float:
    """Shannon entropy of (positive / neutral / negative) sign classes."""
    pos = (phases > 0.01).mean()
    neg = (phases < -0.01).mean()
    neu = 1.0 - pos - neg
    probs = np.array([pos, neg, neu])
    probs = probs[probs > 0]  # avoid log(0)
    return float(-np.sum(probs * np.log2(probs)))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_wikitext_samples(tokenizer, num_samples: int, max_length: int):
    """Load wikitext-2 test split, return list of text strings."""
    print("Loading wikitext-2-raw-v1 test split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [row["text"].strip() for row in dataset if len(row["text"].strip()) > 50]
    return texts[:num_samples]


# ---------------------------------------------------------------------------
# Core measurements
# ---------------------------------------------------------------------------
def measure_sgp(model, tokenizer, texts, device):
    """Run sort probe on GPT-2 block-0 output. Return phase array."""
    hidden_dim = model.config.n_embd
    probe = SortProbe(hidden_dim=hidden_dim).to(device)
    probe.eval()

    all_phases = []
    for i in range(0, len(texts), BATCH_SIZE):
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
            # hidden_states[0] = embeddings, [1] = after block 0
            h_block0 = out.hidden_states[1]  # [batch, seq, hidden]
            phase_traj = probe(h_block0)      # [batch, seq, 2]
            phases = shoelace_area(phase_traj).cpu().numpy()
            all_phases.extend(phases.tolist())

    return np.array(all_phases)


def measure_curvature_ratio(model, tokenizer, texts, device):
    """Compute norm-change between consecutive layers; return ratio L0/max(L1+)."""
    enc = tokenizer(
        texts[:BATCH_SIZE],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        states = out.hidden_states  # tuple: (embed, after_blk0, after_blk1, ...)

    curvatures = []
    for i in range(len(states) - 1):
        diff = (states[i + 1] - states[i]).norm(p=2, dim=-1).mean().item()
        curvatures.append(diff)

    curvatures = np.array(curvatures)
    ratio = curvatures[0] / curvatures[1:].max() if len(curvatures) > 1 else 0.0
    return float(ratio), curvatures.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("EXPERIMENT A: Sort Probe Calibration on GPT-2")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected. Results will be slow but valid.")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"Model loaded. Hidden dim: {model.config.n_embd}, Layers: {model.config.n_layer}")

    # Load data
    texts = load_wikitext_samples(tokenizer, NUM_SAMPLES, MAX_LENGTH)
    print(f"Loaded {len(texts)} samples.")

    # Measure SGP
    print("\nMeasuring SGP (sort probe on block-0)...")
    phases = measure_sgp(model, tokenizer, texts, device)
    mean_phase = float(np.mean(phases))
    std_phase = float(np.std(phases))
    entropy = sign_class_entropy(phases)

    print(f"  Mean phase magnitude : {mean_phase:.4f}")
    print(f"  Std phase            : {std_phase:.4f}")
    print(f"  Sign class entropy   : {entropy:.4f} bits")

    # Measure curvature ratio
    print("\nMeasuring curvature ratio...")
    ratio, all_curvatures = measure_curvature_ratio(model, tokenizer, texts, device)
    print(f"  L0→L1 curvature ratio vs max(later): {ratio:.2f}")

    # Evaluate pass criteria
    check_phase = mean_phase < THRESH_MEAN_PHASE
    check_entropy = entropy < THRESH_ENTROPY
    check_ratio = ratio >= THRESH_CURVATURE_RATIO

    print("\n" + "=" * 60)
    print("PASS CRITERIA:")
    print(f"  [{'PASS' if check_phase else 'FAIL'}] Mean phase < {THRESH_MEAN_PHASE}  →  got {mean_phase:.4f}")
    print(f"  [{'PASS' if check_entropy else 'FAIL'}] Entropy < {THRESH_ENTROPY} bits  →  got {entropy:.4f}")
    print(f"  [{'PASS' if check_ratio else 'FAIL'}] Curvature ratio >= {THRESH_CURVATURE_RATIO}  →  got {ratio:.2f}")

    overall = check_phase and check_entropy and check_ratio
    verdict = "PASS" if overall else "FAIL"

    print("=" * 60)
    print(f"VERDICT: {verdict}")
    if overall:
        print("The probe reproduces known GPT-2 geometry. Proceed to Experiment B.")
    else:
        print("The probe does NOT reproduce known geometry. Do NOT run Experiment B.")
        print("Save this output and experiment_A_result.json and ping Zoe.")
    print("=" * 60)

    # Save results
    results = {
        "experiment": "A",
        "model": MODEL_NAME,
        "num_samples": len(texts),
        "sgp": {
            "mean_phase": mean_phase,
            "std_phase": std_phase,
            "sign_class_entropy_bits": entropy,
            "phases_summary": {
                "positive_frac": float((phases > 0.01).mean()),
                "negative_frac": float((phases < -0.01).mean()),
                "neutral_frac": float(((phases >= -0.01) & (phases <= 0.01)).mean()),
            },
        },
        "curvature": {
            "l0_l1_ratio": ratio,
            "all_layer_curvatures": all_curvatures,
        },
        "pass_criteria": {
            "mean_phase_ok": check_phase,
            "entropy_ok": check_entropy,
            "curvature_ratio_ok": check_ratio,
        },
        "verdict": verdict,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
