#!/usr/bin/env python3
"""
polar_holonomy_gpt2_v3.py — Polar Holonomy Experiment (Corrected)

Fixes the fatal flaw in v1/v2: projecting to C^1 (= 2 real PCA components)
kills all geometric phase. Pancharatnam phase requires states in C^n with
n >= 2 to have non-trivial curvature in projective space CP^{n-1}.

Key changes from v2:
  1. Project to C^n (n configurable, default n=8 → 16 real PCA components)
     giving states in C^8 living on CP^7
  2. Pancharatnam phase via proper complex inner product: ⟨ψ_k|ψ_{k+1}⟩ = Σ_i conj(z_k^i) z_{k+1}^i
  3. States normalized to unit vectors in C^n: |ψ| = 1
  4. External PCA gauge (held-out calibration set, same as v2)
  5. K independent loop traversals with diverse prompts (same as v2)
  6. Three falsification tests preserved

Mathematical note:
  In C^1, the state space is CP^0 = a point. No curvature, no phase. Ever.
  In C^2, the state space is CP^1 = S^2 (Bloch sphere). Berry phase = half solid angle.
  In C^n, the state space is CP^{n-1}. Phase depends on the loop's relationship to
  the Fubini-Study metric. This is the correct setting.

We also test multiple values of n to check convergence.
"""

import sys
import json
import cmath
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu, ttest_ind, ttest_1samp
import warnings
from prompt_banks import BANK_FEAR, BANK_TABLE
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONCEPT = "fear"
COMPLEX_DIMS = [2, 4, 8, 16]  # n values to test (real PCA dims = 2n)
K_LOOPS = 200
N_LOOP_POINTS = 8
N_GAUGE_SAMPLES = 40
N_SHUFFLES = 200
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# ---------------------------------------------------------------------------
# Prompt bank (same as v2)
# ---------------------------------------------------------------------------
PROMPT_BANK = BANK_FEAR

for key, prompts in PROMPT_BANK.items():
    for i, p in enumerate(prompts):
        assert p.lower().count(CONCEPT.lower()) == 2, f"({key})[{i}] has wrong count"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model():
    print("Loading GPT-2...", flush=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    mdl = GPT2Model.from_pretrained("gpt2", output_hidden_states=True); mdl.eval()
    print("  loaded on cpu", flush=True)
    return tok, mdl

def find_concept_positions(tok, prompt, concept):
    input_ids = tok.encode(prompt)
    positions = []
    for i, tid in enumerate(input_ids):
        decoded = tok.decode([tid]).lower()
        if concept.lower() in decoded and len(decoded.strip()) <= len(concept) + 2:
            positions.append(i)
    return positions

def extract_both(tok, mdl, prompt, concept):
    positions = find_concept_positions(tok, prompt, concept)
    if len(positions) != 2:
        return None
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc)
    L = out.hidden_states[-1][0]
    return L[positions[0]].numpy(), L[positions[1]].numpy()

def precompute_all_states(tok, mdl):
    print("Pre-computing hidden states...", flush=True)
    states = {}
    for (al, bl), prompts in PROMPT_BANK.items():
        cell = []
        for prompt in prompts:
            r = extract_both(tok, mdl, prompt, CONCEPT)
            if r is not None:
                cell.append(r)
        states[(al, bl)] = cell
        print(f"  ({al},{bl}): {len(cell)}/{len(prompts)} OK", flush=True)
    return states

# ---------------------------------------------------------------------------
# Complex state construction — THE FIX
# ---------------------------------------------------------------------------
def to_complex_vector(h, pca, n_complex):
    """
    Project hidden state to C^n via PCA.
    h: (768,) real vector
    pca: fitted PCA with >= 2*n_complex components
    n_complex: dimension of complex vector space
    
    Returns: numpy array of shape (n_complex,) dtype=complex128, unit normalized
    """
    proj = pca.transform(h.reshape(1, -1))[0]  # (2*n_complex,) real
    # Pair up: (x0, x1) -> x0 + i*x1, (x2, x3) -> x2 + i*x3, ...
    z = np.array([complex(proj[2*i], proj[2*i+1]) for i in range(n_complex)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    if norm < 1e-10:
        z = np.zeros(n_complex, dtype=complex)
        z[0] = 1.0
    else:
        z = z / norm
    return z

def pancharatnam_phase(states_list):
    """
    states_list: list of unit vectors in C^n (numpy arrays)
    Φ = arg(∏_k ⟨ψ_k|ψ_{k+1}⟩) [cyclic]
    where ⟨ψ_k|ψ_{k+1}⟩ = Σ_i conj(ψ_k^i) * ψ_{k+1}^i (complex inner product)
    """
    prod = complex(1.0, 0.0)
    n = len(states_list)
    for k in range(n):
        overlap = np.vdot(states_list[k], states_list[(k + 1) % n])  # conjugate-linear in first arg
        prod *= overlap
    return cmath.phase(prod)

# ---------------------------------------------------------------------------
# PCA gauge calibration
# ---------------------------------------------------------------------------
def fit_gauge(all_states, n_gauge, n_real_components, rng):
    gauge_vectors = []
    used_indices = {}
    for (al, bl), cell in all_states.items():
        n_cell = min(n_gauge // 4, len(cell))
        indices = rng.choice(len(cell), size=n_cell, replace=False)
        used_indices[(al, bl)] = set(indices.tolist())
        for idx in indices:
            gauge_vectors.append(cell[idx][1])
    H = np.stack(gauge_vectors)
    pca = PCA(n_components=min(n_real_components, H.shape[0], H.shape[1]))
    pca.fit(H)
    return pca, used_indices

# ---------------------------------------------------------------------------
# Loop sampling and phase computation
# ---------------------------------------------------------------------------
CORNERS_CCW = [("low","low"), ("high","low"), ("high","high"), ("low","high")]
CORNERS_TALL = [("low","low"), ("low","high"), ("high","high"), ("high","low")]

def sample_loop_states(all_states, gauge_used, corners, n_points, rng, occurrence=1):
    n_corners = len(corners)
    per_corner = [n_points // n_corners] * n_corners
    for i in range(n_points % n_corners):
        per_corner[i] += 1
    hs = []
    for ci, (al, bl) in enumerate(corners):
        cell = all_states[(al, bl)]
        avail = [i for i in range(len(cell)) if i not in gauge_used.get((al, bl), set())]
        if len(avail) < per_corner[ci]:
            return None
        chosen = rng.choice(avail, size=per_corner[ci], replace=False)
        for idx in chosen:
            hs.append(cell[idx][occurrence])
    return hs

def compute_phase(hs, pca, n_complex):
    states = [to_complex_vector(h, pca, n_complex) for h in hs]
    return pancharatnam_phase(states)

def run_trial(all_states, gauge_used, pca, n_complex, corners, n_points, rng, occurrence=1, shuffle=False):
    hs = sample_loop_states(all_states, gauge_used, corners, n_points, rng, occurrence)
    if hs is None:
        return None
    if shuffle:
        rng.shuffle(hs)
    return compute_phase(hs, pca, n_complex)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_experiment():
    rng = np.random.default_rng(42)
    random.seed(42)
    torch.manual_seed(42)
    
    tok, mdl = load_model()
    
    # Verify
    print("\nVerifying tokenization...", flush=True)
    for (al, bl), prompts in PROMPT_BANK.items():
        for p in prompts:
            pos = find_concept_positions(tok, p, CONCEPT)
            assert len(pos) == 2, f"({al},{bl}): {len(pos)} occurrences"
    print("  All OK.", flush=True)
    
    all_states = precompute_all_states(tok, mdl)
    
    all_results = {
        "concept": CONCEPT, "timestamp": TIMESTAMP, "model": "gpt2-124M",
        "k_loops": K_LOOPS, "n_loop_points": N_LOOP_POINTS,
        "complex_dims_tested": COMPLEX_DIMS,
        "dimensions": {},
    }
    
    for n_complex in COMPLEX_DIMS:
        n_real = 2 * n_complex
        print(f"\n{'='*60}")
        print(f"  C^{n_complex} (= {n_real} real PCA components, state space CP^{n_complex-1})")
        print(f"{'='*60}", flush=True)
        
        # Fresh gauge for each dimension
        pca, gauge_used = fit_gauge(all_states, N_GAUGE_SAMPLES, n_real, rng)
        var_exp = pca.explained_variance_ratio_
        print(f"  PCA var explained: {sum(var_exp):.3f} (top: {var_exp[0]:.3f}, {var_exp[1]:.3f})")
        
        # CCW loops
        phases_ccw = [run_trial(all_states, gauge_used, pca, n_complex, CORNERS_CCW, N_LOOP_POINTS, rng) for _ in range(K_LOOPS)]
        phases_ccw = np.array([p for p in phases_ccw if p is not None])
        
        # CW loops
        corners_cw = list(reversed(CORNERS_CCW))
        phases_cw = [run_trial(all_states, gauge_used, pca, n_complex, corners_cw, N_LOOP_POINTS, rng) for _ in range(K_LOOPS)]
        phases_cw = np.array([p for p in phases_cw if p is not None])
        
        # Tall loops
        phases_tall = [run_trial(all_states, gauge_used, pca, n_complex, CORNERS_TALL, N_LOOP_POINTS, rng) for _ in range(K_LOOPS)]
        phases_tall = np.array([p for p in phases_tall if p is not None])
        
        # Fast loops (4 points)
        phases_fast = [run_trial(all_states, gauge_used, pca, n_complex, CORNERS_CCW, 4, rng) for _ in range(K_LOOPS)]
        phases_fast = np.array([p for p in phases_fast if p is not None])
        
        # Null (shuffled)
        phases_null = [run_trial(all_states, gauge_used, pca, n_complex, CORNERS_CCW, N_LOOP_POINTS, rng, shuffle=True) for _ in range(N_SHUFFLES)]
        phases_null = np.array([p for p in phases_null if p is not None])
        
        # First-occurrence for accumulation test
        phases_first = [run_trial(all_states, gauge_used, pca, n_complex, CORNERS_CCW, N_LOOP_POINTS, rng, occurrence=0) for _ in range(K_LOOPS)]
        phases_first = np.array([p for p in phases_first if p is not None])
        
        # Stats
        orient_sum = np.mean(phases_ccw) + np.mean(phases_cw)
        shape_d = abs(np.mean(phases_ccw)) - abs(np.mean(phases_tall))
        sched_d = np.mean(phases_ccw) - np.mean(phases_fast)
        
        U, p_mw = mannwhitneyu(phases_ccw, phases_null, alternative='two-sided')
        t_wt, p_wt = ttest_ind(phases_ccw, phases_null, equal_var=False)
        t_z, p_z = ttest_1samp(phases_ccw, 0.0)
        
        # Accumulation
        t_acc, p_acc = ttest_ind(np.abs(phases_ccw), np.abs(phases_first), equal_var=False)
        
        print(f"  CCW:  mean={np.mean(phases_ccw):+.4f}  std={np.std(phases_ccw):.4f}  range=[{phases_ccw.min():.3f}, {phases_ccw.max():.3f}]")
        print(f"  CW:   mean={np.mean(phases_cw):+.4f}  std={np.std(phases_cw):.4f}")
        print(f"  tall: mean={np.mean(phases_tall):+.4f}  std={np.std(phases_tall):.4f}")
        print(f"  fast: mean={np.mean(phases_fast):+.4f}  std={np.std(phases_fast):.4f}")
        print(f"  null: mean={np.mean(phases_null):+.4f}  std={np.std(phases_null):.4f}")
        print(f"  1st:  mean={np.mean(phases_first):+.4f}  std={np.std(phases_first):.4f}")
        print(f"  ---")
        print(f"  Orientation sum:    {orient_sum:+.4f}")
        print(f"  Shape delta:        {shape_d:+.4f}")
        print(f"  Schedule delta:     {sched_d:+.4f}")
        print(f"  Mann-Whitney p:     {p_mw:.4f}")
        print(f"  Phase vs zero p:    {p_z:.4f}")
        print(f"  Accumulation p:     {p_acc:.4f}")
        print(f"  mean|Φ_2nd|={np.mean(np.abs(phases_ccw)):.4f}  mean|Φ_1st|={np.mean(np.abs(phases_first)):.4f}")
        
        # Verdict for this dimension
        ev_for, ev_against = [], []
        tol = max(2 * max(np.std(phases_ccw), np.std(phases_cw)) / np.sqrt(K_LOOPS), 0.05)
        if abs(orient_sum) < tol:
            ev_for.append(f"orientation flip (sum={orient_sum:.4f})")
        else:
            ev_against.append(f"orientation flip FAILS (sum={orient_sum:.4f})")
        if abs(shape_d) < 0.15:
            ev_for.append(f"shape invariance (Δ={shape_d:.4f})")
        else:
            ev_against.append(f"shape invariance FAILS (Δ={shape_d:.4f})")
        if abs(sched_d) < 0.15:
            ev_for.append(f"schedule invariance (Δ={sched_d:.4f})")
        else:
            ev_against.append(f"schedule invariance FAILS (Δ={sched_d:.4f})")
        if p_mw < 0.05:
            ev_for.append(f"significant vs null (p={p_mw:.4f})")
        else:
            ev_against.append(f"NOT significant vs null (p={p_mw:.4f})")
        if p_z < 0.05 and np.std(phases_ccw) > 0.01:  # require non-trivial spread
            ev_for.append(f"phase ≠ 0 (mean={np.mean(phases_ccw):.4f}, p={p_z:.4f})")
        else:
            ev_against.append(f"phase ≈ 0 (mean={np.mean(phases_ccw):.4f}, p={p_z:.4f})")
        
        v = ("GEOMETRIC PHASE DETECTED" if len(ev_for) >= 4 and p_mw < 0.05
             else "GEOMETRIC PHASE CANDIDATE" if len(ev_for) >= 3
             else "INCONCLUSIVE" if len(ev_for) >= 2
             else "NULL RESULT")
        
        print(f"  VERDICT (C^{n_complex}): {v}")
        for e in ev_for: print(f"    ✓ {e}")
        for a in ev_against: print(f"    ✗ {a}")
        
        all_results["dimensions"][str(n_complex)] = {
            "n_complex": n_complex,
            "n_real_pca": n_real,
            "pca_total_variance": float(sum(var_exp)),
            "ccw_mean": float(np.mean(phases_ccw)),
            "ccw_std": float(np.std(phases_ccw)),
            "cw_mean": float(np.mean(phases_cw)),
            "cw_std": float(np.std(phases_cw)),
            "tall_mean": float(np.mean(phases_tall)),
            "null_mean": float(np.mean(phases_null)),
            "null_std": float(np.std(phases_null)),
            "orientation_sum": float(orient_sum),
            "shape_delta": float(shape_d),
            "schedule_delta": float(sched_d),
            "mann_whitney_p": float(p_mw),
            "phase_vs_zero_p": float(p_z),
            "accumulation_p": float(p_acc),
            "mean_abs_2nd": float(np.mean(np.abs(phases_ccw))),
            "mean_abs_1st": float(np.mean(np.abs(phases_first))),
            "verdict": v,
            "evidence_for": ev_for,
            "evidence_against": ev_against,
            "phases_ccw": phases_ccw.tolist(),
            "phases_cw": phases_cw.tolist(),
            "phases_null": phases_null.tolist(),
        }
    
    # Save
    out_json = RESULT_DIR / f"polar_holonomy_v3_{TIMESTAMP}.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {out_json}", flush=True)
    
    # Summary plot
    if HAS_MPL:
        fig, axes = plt.subplots(1, len(COMPLEX_DIMS), figsize=(4*len(COMPLEX_DIMS), 4), sharey=True)
        if len(COMPLEX_DIMS) == 1:
            axes = [axes]
        for ax, n_c in zip(axes, COMPLEX_DIMS):
            d = all_results["dimensions"][str(n_c)]
            ccw = np.array(d["phases_ccw"])
            null = np.array(d["phases_null"])
            lo = min(ccw.min(), null.min()) - 0.1
            hi = max(ccw.max(), null.max()) + 0.1
            bins = np.linspace(lo, hi, 35)
            ax.hist(null, bins=bins, alpha=0.5, color="gray", label="null")
            ax.hist(ccw, bins=bins, alpha=0.6, color="red", label="CCW")
            ax.axvline(0, color="k", ls=":", alpha=0.3)
            ax.set_xlabel("Φ (rad)")
            ax.set_title(f"C^{n_c} (CP^{n_c-1})\np_MW={d['mann_whitney_p']:.3f}\n{d['verdict']}", fontsize=9)
            ax.legend(fontsize=7)
        axes[0].set_ylabel("Count")
        plt.suptitle(f"Polar Holonomy v3 — GPT-2 — '{CONCEPT}'", fontweight="bold")
        plt.tight_layout()
        out_png = RESULT_DIR / f"polar_holonomy_v3_{TIMESTAMP}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Plot: {out_png}", flush=True)
    
    return all_results

if __name__ == "__main__":
    run_experiment()
