#!/usr/bin/env python3
"""
flatness_test_v2.py — Frame-transition commutativity with subsampling.

WHAT WAS WRONG WITH v1:
  2300 triangles x full frame-transition computation on CPU = timeout.
  Also: no semantic null model. Was measuring commutativity of the ambient
  connection on CP^15, not whether concept-conditioned states have flat or
  curved parallel transport relative to a randomized baseline.

THIS VERSION:
  1. Subsamples 200 triangles from the 25-cell grid.
  2. For each triangle (A, B, C), computes the frame-transition residual:
       R(A,B,C) = ||T_BC ∘ T_AB - T_AC||_F / ||T_AC||_F
     where T_XY = closest-orthogonal map from cell X to cell Y
     (Procrustes alignment of their state matrices).
  3. Compares semantic triangles (corners drawn from semantically adjacent
     cells — manhattan distance <= 1 in the 5x5 grid) vs. random triangles.
  4. Reports mean residual +/- bootstrap CI and Mann-Whitney p.

Interpretation:
  Flat connection -> residuals near 0 (T_BC ∘ T_AB ≈ T_AC).
  Curved connection -> residuals > 0, and semantic adjacency should predict
  SMALLER residuals (smoother transitions between neighboring cells).
  If semantic adjacency has no effect: the connection structure is random
  with respect to concept topology — another null.

Estimated runtime: ~10 minutes on Spark.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
from scipy.linalg import orthogonal_procrustes
from scipy.stats import mannwhitneyu, bootstrap

sys.path.insert(0, str(Path(__file__).parent))
import polar_holonomy_gpt2_v3 as v3
from area_law_test import make_5x5_prompt_bank, precompute_5x5_states

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

N_TRIANGLES = 200
SEED = 42
MIN_STATES_PER_CELL = 3


def cell_matrix(states_list, max_states=6):
    vecs = [h2 for h1, h2 in states_list[:max_states]]
    if len(vecs) < MIN_STATES_PER_CELL:
        return None
    return np.stack(vecs)


def frame_transition(mat_a, mat_b):
    """T = argmin ||T @ mat_a - mat_b||_F via Procrustes."""
    R, _ = orthogonal_procrustes(mat_a.T, mat_b.T)
    return R.T


def transition_residual(mat_a, mat_b, mat_c):
    """||T_BC ∘ T_AB - T_AC||_F / ||T_AC||_F"""
    T_AB = frame_transition(mat_a, mat_b)
    T_BC = frame_transition(mat_b, mat_c)
    T_AC = frame_transition(mat_a, mat_c)
    composed = T_BC @ T_AB
    diff = composed - T_AC
    norm_ac = np.linalg.norm(T_AC, 'fro')
    return np.linalg.norm(diff, 'fro') / (norm_ac + 1e-10)


def manhattan_dist(k1, k2):
    return abs(k1[0] - k2[0]) + abs(k1[1] - k2[1])


def run():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("  FLATNESS TEST v2: FRAME-TRANSITION COMMUTATIVITY (SUBSAMPLED)")
    print("=" * 70)
    print()
    print("Loading GPT-2 and 5x5 states...")
    tok, mdl = v3.load_model()
    prompt_bank = make_5x5_prompt_bank()
    all_states = precompute_5x5_states(tok, mdl, "threshold", prompt_bank)

    cell_keys = sorted(all_states.keys())
    cell_mats = {}
    for key in cell_keys:
        m = cell_matrix(all_states[key])
        if m is not None:
            cell_mats[key] = m
    valid_keys = sorted(cell_mats.keys())
    print(f"Valid cells: {len(valid_keys)} / 25")

    all_triples = list(combinations(valid_keys, 3))
    print(f"Total possible triangles: {len(all_triples)}, subsampling {N_TRIANGLES}")

    chosen_idxs = rng.choice(len(all_triples), size=min(N_TRIANGLES, len(all_triples)), replace=False)
    chosen = [all_triples[i] for i in chosen_idxs]

    adjacent_residuals = []
    distant_residuals = []
    all_residuals = []

    print("Computing transition residuals...")
    for idx, (ka, kb, kc) in enumerate(chosen):
        if idx % 50 == 0:
            print(f"  {idx}/{len(chosen)}...", flush=True)
        r = transition_residual(cell_mats[ka], cell_mats[kb], cell_mats[kc])
        all_residuals.append(r)

        dists = [manhattan_dist(ka, kb), manhattan_dist(kb, kc), manhattan_dist(ka, kc)]
        if max(dists) <= 1:
            adjacent_residuals.append(r)
        elif min(dists) >= 2:
            distant_residuals.append(r)

    all_r = np.array(all_residuals)
    print(f"\nAll triangles: n={len(all_r)}, residual = {all_r.mean():.4f} +/- {all_r.std():.4f}")

    if adjacent_residuals and distant_residuals:
        adj = np.array(adjacent_residuals)
        dist_arr = np.array(distant_residuals)
        print(f"Adjacent (dist<=1): n={len(adj)}, residual = {adj.mean():.4f} +/- {adj.std():.4f}")
        print(f"Distant (dist>=2):  n={len(dist_arr)}, residual = {dist_arr.mean():.4f} +/- {dist_arr.std():.4f}")
        U, p_mw = mannwhitneyu(adj, dist_arr, alternative='less')
        print(f"Mann-Whitney (adjacent < distant): p = {p_mw:.4f}")
    else:
        p_mw = 1.0
        print("Not enough adjacent/distant triangles for comparison.")
        adj = np.array(all_residuals)
        dist_arr = np.array(all_residuals)

    boot = bootstrap((all_r,), np.mean, n_resamples=1000, random_state=SEED)
    ci_lo, ci_hi = boot.confidence_interval
    print(f"Bootstrap 95% CI for mean residual: [{ci_lo:.4f}, {ci_hi:.4f}]")

    if p_mw < 0.05:
        verdict = ("TOPOLOGY-CONSISTENT TRANSPORT — adjacent cells have smoother "
                   "frame transitions, curvature is real and concept-local")
    elif all_r.mean() < 0.05:
        verdict = ("FLAT — connection is approximately flat (residuals near 0), "
                   "consistent with trivial holonomy group")
    else:
        verdict = ("NULL — residuals are large but not patterned by semantic distance; "
                   "frame transitions are noisy / connection is not concept-structured")

    print(f"\nVERDICT: {verdict}")
    print()
    print("NOTE: Flat connection is COMPATIBLE with area law experiments.")
    print("This tests whether CONCEPT TOPOLOGY predicts TRANSPORT SMOOTHNESS.")

    results = {
        "timestamp": TIMESTAMP,
        "n_triangles_sampled": N_TRIANGLES,
        "n_triangles_computed": len(all_residuals),
        "mean_residual": float(all_r.mean()),
        "std_residual": float(all_r.std()),
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "n_adjacent": len(adjacent_residuals),
        "n_distant": len(distant_residuals),
        "adjacent_mean": float(np.mean(adjacent_residuals)) if adjacent_residuals else None,
        "distant_mean": float(np.mean(distant_residuals)) if distant_residuals else None,
        "mann_whitney_p": float(p_mw),
        "verdict": verdict,
        "what_v1_problem": "2300 triangles -> timeout; no semantic null model",
        "what_v2_fixes": "subsampled to 200; compares adjacent vs distant cell triangles",
    }
    out = RESULT_DIR / f"flatness_test_v2_{TIMESTAMP}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out}")
    return results


if __name__ == "__main__":
    run()
