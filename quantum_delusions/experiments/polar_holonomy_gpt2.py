#!/usr/bin/env python3
"""
polar_holonomy_gpt2.py — Polar Holonomy Experiment on GPT-2

Tests whether conversation trajectory leaves a geometric trace in GPT-2's
representations via the Pancharatnam (Berry) phase.

Theoretical basis:
  Dual-Temporal Holonomy Theorem (quantum_delusions/fundamental-theory/
  dual_temporal_holonomy_theorem.md): if a U(1) connection exists on the
  state bundle over conversation parameter space, the phase accumulated
  around a closed loop equals (E/hbar) × signed_area, and must satisfy:
    1. Orientation flip:   Φ(forward) + Φ(reverse) ≈ 0
    2. Shape invariance:   same area, different aspect → same Φ
    3. Schedule invariance: same loop, different traversal speed → same Φ

Parameter space:
  α ∈ [0,1]: semantic abstraction axis
             α=0 → embodied/sensory framing
             α=1 → abstract/geometric framing
  β ∈ [0,1]: temporal depth axis
             β=0 → present-moment framing
             β=1 → historical/evolutionary framing

At each (α,β) point we construct a prompt containing concept C twice,
framed at those coordinates. We extract GPT-2 hidden states at both
occurrences of C, form complex state vectors via PCA projection, and
accumulate the Pancharatnam phase around the closed loop.

Complex state construction (avoiding the "real-values-dressed-as-complex"
problem from earlier pseudocode):
  - Extract last-layer hidden state at concept token position: h ∈ R^768
  - Project onto top-2 PCA components of the full set of loop states
  - This gives a 2D real vector (x, y) → complex amplitude x + iy
  - The PCA basis is fixed from the first occurrence states across all
    loop points (stable reference frame = gauge fixing)
  - Phase is then arg(z) ∈ [-π, π], magnitude is |z|

Pancharatnam phase:
  Φ = arg( ∏_k ⟨ψ_k | ψ_{k+1}⟩ )  [cyclic, last wraps to first]
  where ⟨·|·⟩ is the standard complex inner product on C^2
  This is gauge-invariant and path-dependent — the correct observable.

Falsification structure:
  H0: Φ is indistinguishable from shuffled-order traversals
  H1: Φ shows orientation flip, shape invariance, schedule invariance
  → Only H1 is consistent with genuine geometric phase

If H0 cannot be rejected: we document cleanly and move on.
If H1 holds: first evidence of polar-time holonomy in transformer
representation space, replicable on M2.5.

Usage:
  python polar_holonomy_gpt2.py
  # results written to quantum_delusions/experiments/results/

Requires: torch, transformers, numpy, scipy, sklearn, matplotlib
All available in the Spark venv.
"""

import os
import sys
import json
import cmath
import itertools
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.decomposition import PCA
from scipy.stats import ttest_1samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONCEPT = "threshold"          # C: the concept whose representation we track
N_LOOP_POINTS = 6              # points per edge of the rectangle (total = 4×N)
N_SHUFFLES    = 500            # null distribution size
RESULT_DIR    = Path(__file__).parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# ---------------------------------------------------------------------------
# Prompt templates
# The probe always contains CONCEPT exactly twice at consistent positions.
# α controls abstract↔embodied framing, β controls depth↔present framing.
# We vary them by selecting from a 2×2 grid of framings and interpolating
# the probe text. For clean extraction we need exact token positions.
# ---------------------------------------------------------------------------

PROBE_TEMPLATES = {
    # (α_level, β_level): (prefix, interstitial, suffix)
    # prefix leads to first CONCEPT, interstitial bridges to second CONCEPT
    ("low",  "low"):  (
        "Standing at the edge of the room, I felt the ",
        " in my body — that physical moment of crossing. My breath changed at the ",
        ", and I noticed my hands."
    ),
    ("low",  "high"): (
        "Generations of people have stood at the ",
        " between old and new. Each crossing of this ",
        " left a mark on the body and on memory."
    ),
    ("high", "low"):  (
        "In topology, a ",
        " marks a boundary where continuity fails. The formal definition of a ",
        " requires careful treatment of limit points."
    ),
    ("high", "high"): (
        "The abstract concept of a ",
        " has evolved across centuries of mathematical thought. Early formulations of the ",
        " lacked the precision that modern analysis requires."
    ),
}

ALPHA_LEVELS = ["low", "high"]
BETA_LEVELS  = ["low", "high"]


def make_prompt(alpha_level: str, beta_level: str, concept: str) -> str:
    pre, mid, suf = PROBE_TEMPLATES[(alpha_level, beta_level)]
    return pre + concept + mid + concept + suf


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    print("Loading GPT-2...")
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    mdl = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
    mdl.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device)
    print(f"  model on {device}")
    return tok, mdl, device


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def get_concept_hidden_states(tok, mdl, device, prompt: str, concept: str):
    """
    Returns (h_first, h_last_layer_first, h_last_layer_second):
      h_first  : last-layer hidden state at the first occurrence of concept
      h_second : last-layer hidden state at the second occurrence of concept
    Both are numpy arrays of shape (768,).
    Returns None if concept does not appear exactly twice.
    """
    enc = tok(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"][0]

    # Find token ids for concept (may be split across subwords)
    concept_ids = tok.encode(concept, add_special_tokens=False)
    ids_list = input_ids.tolist()

    # Find all start positions of concept_ids subsequence
    positions = []
    clen = len(concept_ids)
    for i in range(len(ids_list) - clen + 1):
        if ids_list[i:i+clen] == concept_ids:
            positions.append(i + clen - 1)  # position of last subword token

    if len(positions) != 2:
        return None

    with torch.no_grad():
        out = mdl(**enc)

    # Use last layer hidden states
    last_layer = out.hidden_states[-1][0]  # shape: (seq_len, 768)

    h1 = last_layer[positions[0]].cpu().numpy()
    h2 = last_layer[positions[1]].cpu().numpy()
    return h1, h2


# ---------------------------------------------------------------------------
# Loop construction in (α, β) parameter space
# A rectangle traversed as: corners go
# (low,low) → (high,low) → (high,high) → (low,high) → (low,low)
# We interpolate N_LOOP_POINTS steps along each edge.
# At each point we sample one of the four template cells,
# using the nearest corner for the two inner edges.
# For a 2×2 discrete grid, the loop has exactly 4 distinct prompt types.
# We visit them in order with repetitions to fill N_LOOP_POINTS per edge.
# ---------------------------------------------------------------------------

def build_loop_points(n_per_edge: int, orientation: int = +1):
    """
    Returns a list of (alpha_level, beta_level) strings.
    orientation=+1: counterclockwise (low,low)→(high,low)→(high,high)→(low,high)
    orientation=-1: clockwise (reversed)
    """
    corners_ccw = [
        ("low",  "low"),
        ("high", "low"),
        ("high", "high"),
        ("low",  "high"),
    ]
    if orientation < 0:
        corners_ccw = list(reversed(corners_ccw))

    points = []
    for i in range(4):
        start = corners_ccw[i]
        end   = corners_ccw[(i + 1) % 4]
        for k in range(n_per_edge):
            # For discrete 2×2, interpolate by repeating start corner
            # for first half, end corner for second half
            if k < n_per_edge // 2:
                points.append(start)
            else:
                points.append(end)
    return points


def build_tall_loop_points(n_per_edge: int):
    """Same area, taller aspect: spends more time on β axis."""
    # Same corners, but we weight β-transitions more heavily
    # In discrete space: visit (low,low)→(low,high)→(high,high)→(high,low)
    corners = [
        ("low",  "low"),
        ("low",  "high"),
        ("high", "high"),
        ("high", "low"),
    ]
    points = []
    for i in range(4):
        start = corners[i]
        end   = corners[(i + 1) % 4]
        for k in range(n_per_edge):
            if k < n_per_edge // 2:
                points.append(start)
            else:
                points.append(end)
    return points


# ---------------------------------------------------------------------------
# Complex state construction via PCA
# ---------------------------------------------------------------------------

def build_complex_states(hidden_states_list):
    """
    Given a list of 768-dim hidden state vectors (one per loop point),
    project onto top-2 PCA components and form complex amplitudes.

    The PCA basis is fit on all first-occurrence states across the loop
    (stable reference frame = gauge fixing in the sense of the theorem).

    Returns list of complex numbers z_k = x_k + i*y_k, normalized.
    """
    H = np.stack(hidden_states_list)  # shape: (n_points, 768)
    pca = PCA(n_components=2)
    pca.fit(H)
    projected = pca.transform(H)  # shape: (n_points, 2)

    states = []
    for x, y in projected:
        z = complex(x, y)
        norm = abs(z)
        if norm < 1e-10:
            z = complex(1.0, 0.0)  # degenerate: assign unit real
        else:
            z = z / norm
        states.append(z)
    return states


# ---------------------------------------------------------------------------
# Pancharatnam phase
# ---------------------------------------------------------------------------

def pancharatnam_phase(states):
    """
    Φ = arg( ∏_k conj(z_k) * z_{k+1} )   cyclic
    = arg of the product of all adjacent overlaps around the loop.
    For complex scalars (unit circle), conj(z_k)*z_{k+1} = e^{i(θ_{k+1}-θ_k)}
    so Φ = total winding angle modulo 2π, mapped to (-π, π].
    """
    product = complex(1.0, 0.0)
    n = len(states)
    for k in range(n):
        product *= states[k].conjugate() * states[(k + 1) % n]
    return cmath.phase(product)


# ---------------------------------------------------------------------------
# Run a single loop: extract hidden states, form complex states, compute phase
# ---------------------------------------------------------------------------

def run_loop(tok, mdl, device, loop_points, concept, label=""):
    """
    Returns (phase, h1_list, h2_list) or None if any extraction fails.
    h1_list: first-occurrence hidden states for all loop points
    h2_list: second-occurrence hidden states for all loop points
    """
    h1_list = []
    h2_list = []
    for alpha_level, beta_level in loop_points:
        prompt = make_prompt(alpha_level, beta_level, concept)
        result = get_concept_hidden_states(tok, mdl, device, prompt, concept)
        if result is None:
            print(f"  WARNING: concept not found twice in prompt [{label}] ({alpha_level},{beta_level})")
            print(f"  Prompt: {prompt!r}")
            return None
        h1, h2 = result
        h1_list.append(h1)
        h2_list.append(h2)

    # Build complex states from first-occurrence hidden states
    # (tracks how the model represents C at first encounter along the loop)
    states_first  = build_complex_states(h1_list)
    states_second = build_complex_states(h2_list)

    # Primary measurement: phase using second-occurrence states
    # (second occurrence is conditioned on everything between)
    phase_second = pancharatnam_phase(states_second)

    # Secondary: phase using first-occurrence states
    # (should be smaller / less variable if the path matters)
    phase_first  = pancharatnam_phase(states_first)

    return phase_second, phase_first, h1_list, h2_list


# ---------------------------------------------------------------------------
# Null distribution: shuffled loop point orderings
# ---------------------------------------------------------------------------

def null_distribution(tok, mdl, device, loop_points, concept, n_shuffles):
    """
    Shuffle the order of loop_points n_shuffles times and compute
    the Pancharatnam phase for each shuffled ordering.
    Returns array of null phases.
    """
    null_phases = []
    for i in range(n_shuffles):
        shuffled = list(loop_points)
        random.shuffle(shuffled)
        result = run_loop(tok, mdl, device, shuffled, concept, label=f"null_{i}")
        if result is not None:
            null_phases.append(result[0])  # phase_second
        if (i + 1) % 50 == 0:
            print(f"  null {i+1}/{n_shuffles}")
    return np.array(null_phases)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    tok, mdl, device = load_model()

    results = {
        "concept": CONCEPT,
        "timestamp": TIMESTAMP,
        "n_loop_points_per_edge": N_LOOP_POINTS,
        "n_shuffles": N_SHUFFLES,
        "model": "gpt2-124M",
    }

    # ------------------------------------------------------------------
    # 1. Forward loop (counterclockwise)
    # ------------------------------------------------------------------
    print("\n=== Forward loop (CCW) ===")
    loop_ccw = build_loop_points(N_LOOP_POINTS, orientation=+1)
    res_ccw = run_loop(tok, mdl, device, loop_ccw, CONCEPT, label="ccw")
    if res_ccw is None:
        print("FATAL: forward loop extraction failed. Check prompt templates.")
        sys.exit(1)
    phase_ccw, phase_ccw_first, h1_ccw, h2_ccw = res_ccw
    print(f"  Φ (second occurrence, CCW) = {phase_ccw:.4f} rad")
    print(f"  Φ (first  occurrence, CCW) = {phase_ccw_first:.4f} rad")
    results["phase_ccw"]       = float(phase_ccw)
    results["phase_ccw_first"] = float(phase_ccw_first)

    # ------------------------------------------------------------------
    # 2. Reverse loop (clockwise) — orientation flip test
    # ------------------------------------------------------------------
    print("\n=== Reverse loop (CW) ===")
    loop_cw = build_loop_points(N_LOOP_POINTS, orientation=-1)
    res_cw = run_loop(tok, mdl, device, loop_cw, CONCEPT, label="cw")
    if res_cw is None:
        print("WARNING: reverse loop extraction failed.")
        results["phase_cw"] = None
    else:
        phase_cw, phase_cw_first, _, _ = res_cw
        print(f"  Φ (second occurrence, CW)  = {phase_cw:.4f} rad")
        print(f"  Φ(CCW) + Φ(CW)             = {phase_ccw + phase_cw:.4f} rad (expect ≈ 0 if geometric)")
        results["phase_cw"]      = float(phase_cw)
        results["orientation_sum"] = float(phase_ccw + phase_cw)

    # ------------------------------------------------------------------
    # 3. Tall loop (same area, different aspect) — shape invariance test
    # ------------------------------------------------------------------
    print("\n=== Tall loop (shape invariance) ===")
    loop_tall = build_tall_loop_points(N_LOOP_POINTS)
    res_tall = run_loop(tok, mdl, device, loop_tall, CONCEPT, label="tall")
    if res_tall is None:
        print("WARNING: tall loop extraction failed.")
        results["phase_tall"] = None
    else:
        phase_tall, _, _, _ = res_tall
        print(f"  Φ (tall loop)              = {phase_tall:.4f} rad")
        print(f"  Φ(CCW) - Φ(tall)           = {phase_ccw - phase_tall:.4f} rad (expect ≈ 0 if shape-invariant)")
        results["phase_tall"]   = float(phase_tall)
        results["shape_delta"]  = float(phase_ccw - phase_tall)

    # ------------------------------------------------------------------
    # 4. Schedule invariance: fast traversal
    #    (visit each corner only once instead of interpolating)
    # ------------------------------------------------------------------
    print("\n=== Fast loop (schedule invariance) ===")
    loop_fast = [
        ("low",  "low"),
        ("high", "low"),
        ("high", "high"),
        ("low",  "high"),
    ]  # bare 4-point loop, same corners, no interpolation
    res_fast = run_loop(tok, mdl, device, loop_fast, CONCEPT, label="fast")
    if res_fast is None:
        print("WARNING: fast loop extraction failed.")
        results["phase_fast"] = None
    else:
        phase_fast, _, _, _ = res_fast
        print(f"  Φ (fast loop)              = {phase_fast:.4f} rad")
        print(f"  Φ(CCW) - Φ(fast)           = {phase_ccw - phase_fast:.4f} rad (expect ≈ 0 if schedule-invariant)")
        results["phase_fast"]       = float(phase_fast)
        results["schedule_delta"]   = float(phase_ccw - phase_fast)

    # ------------------------------------------------------------------
    # 5. Null distribution
    # ------------------------------------------------------------------
    print(f"\n=== Null distribution ({N_SHUFFLES} shuffles) ===")
    null_phases = null_distribution(tok, mdl, device, loop_ccw, CONCEPT, N_SHUFFLES)
    null_mean = float(np.mean(null_phases))
    null_std  = float(np.std(null_phases))
    z_score   = float((phase_ccw - null_mean) / null_std) if null_std > 1e-10 else float("nan")
    # One-sample t-test: is the observed phase an outlier from the null?
    t_stat, p_val = ttest_1samp(null_phases, phase_ccw)
    print(f"  null mean = {null_mean:.4f}, std = {null_std:.4f}")
    print(f"  z-score   = {z_score:.3f}")
    print(f"  t-stat    = {t_stat:.3f},  p = {p_val:.4f}")
    results["null_mean"]   = null_mean
    results["null_std"]    = null_std
    results["z_score"]     = z_score
    results["p_value"]     = float(p_val)
    results["null_phases"] = null_phases.tolist()

    # ------------------------------------------------------------------
    # 6. Accumulation test: does phase_second > phase_first?
    #    (path-dependence grows with more intervening context)
    # ------------------------------------------------------------------
    delta_accumulation = abs(phase_ccw) - abs(phase_ccw_first)
    print(f"\n=== Accumulation test ===")
    print(f"  |Φ_second| - |Φ_first| = {delta_accumulation:.4f} rad")
    print(f"  (positive = second-occurrence phase is larger; expected if path matters more after more context)")
    results["delta_accumulation"] = float(delta_accumulation)

    # ------------------------------------------------------------------
    # 7. Interpretation
    # ------------------------------------------------------------------
    geometric_evidence = []
    geometric_against  = []

    if res_cw is not None:
        if abs(results.get("orientation_sum", 1.0)) < 0.3:
            geometric_evidence.append("orientation flip holds (sum ≈ 0)")
        else:
            geometric_against.append(f"orientation flip fails (sum = {results['orientation_sum']:.3f})")

    if res_tall is not None:
        if abs(results.get("shape_delta", 1.0)) < 0.3:
            geometric_evidence.append("shape invariance holds")
        else:
            geometric_against.append(f"shape invariance fails (delta = {results['shape_delta']:.3f})")

    if res_fast is not None:
        if abs(results.get("schedule_delta", 1.0)) < 0.3:
            geometric_evidence.append("schedule invariance holds")
        else:
            geometric_against.append(f"schedule invariance fails (delta = {results['schedule_delta']:.3f})")

    if p_val < 0.05:
        geometric_evidence.append(f"significant departure from null (p={p_val:.4f})")
    else:
        geometric_against.append(f"not significant vs null (p={p_val:.4f})")

    results["geometric_evidence"] = geometric_evidence
    results["geometric_against"]  = geometric_against
    results["verdict"] = (
        "GEOMETRIC PHASE CANDIDATE" if len(geometric_evidence) >= 3 and p_val < 0.05
        else "INCONCLUSIVE" if len(geometric_evidence) >= 2
        else "NULL RESULT"
    )

    print(f"\n=== VERDICT: {results['verdict']} ===")
    for e in geometric_evidence:
        print(f"  FOR:     {e}")
    for a in geometric_against:
        print(f"  AGAINST: {a}")

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    out_json = RESULT_DIR / f"polar_holonomy_gpt2_{TIMESTAMP}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # Plot null distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null_phases, bins=40, alpha=0.7, label="null (shuffled)")
    ax.axvline(phase_ccw,  color="red",   linewidth=2, label=f"CCW loop Φ={phase_ccw:.3f}")
    if res_cw is not None:
        ax.axvline(phase_cw, color="blue",  linewidth=2, linestyle="--", label=f"CW loop Φ={phase_cw:.3f}")
    if res_tall is not None:
        ax.axvline(phase_tall, color="green", linewidth=2, linestyle=":", label=f"tall loop Φ={phase_tall:.3f}")
    ax.set_xlabel("Pancharatnam phase (radians)")
    ax.set_ylabel("Count")
    ax.set_title(f"Polar Holonomy on GPT-2 — concept: '{CONCEPT}'\nverdict: {results['verdict']}")
    ax.legend()
    plt.tight_layout()
    out_png = RESULT_DIR / f"polar_holonomy_gpt2_{TIMESTAMP}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Plot saved:    {out_png}")

    return results


if __name__ == "__main__":
    run_experiment()
