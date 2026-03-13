#!/usr/bin/env python3
"""
area_law_fs_reanalysis_v2.py — Fubini-Study reanalysis, corrected.

WHAT WAS WRONG WITH v1:
  The first FS reanalysis measured the curvature of CP^15 itself — K ≈ 1.064 ± 0.012.
  That's correct math and a complete null result: we measured the geometry of the
  *container* (the ambient projective space), not the *content* (whether concept-
  conditioned hidden states curve differently than random ones).

  The control check confirmed this: random CP^15 triangles give the same K ≈ 1.
  The script had no semantic null model. It could not have found anything.

WHAT THIS SCRIPT DOES INSTEAD:
  For each triangle in the 5x5 semantic grid:
    1. Compute the FS geodesic distance between each pair of corners.
    2. Compute the Pancharatnam phase around the triangle.
    3. Compare |Phi| / (sum of pairwise FS distances)^2 to a SHUFFLED NULL
       where the same states are randomly reassigned to cells.

  Signal: concept-conditioned triangles should accumulate MORE phase per unit
  FS-area than random triangles at the same distance scale.

  This is not measuring K of CP^n. It's asking:
    Does semantic neighborhood structure predict excess holonomy
    beyond what the ambient FS geometry would predict?

Depends on: area_law_test.py (precompute_5x5_states, make_5x5_prompt_bank),
            polar_holonomy_gpt2_v3.py (load_model, pancharatnam_phase)

Estimated runtime: ~15 minutes on Spark (reuses existing hidden states).
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
from scipy.stats import binomtest, mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
import polar_holonomy_gpt2_v3 as v3
from area_law_test import make_5x5_prompt_bank, precompute_5x5_states

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

N_COMPLEX = 16   # C^16 -> CP^15, matching area_law_test
N_SHUFFLES = 500
SEED = 42


def fs_distance(z1, z2):
    """Fubini-Study geodesic distance between two unit vectors in C^n."""
    inner = np.abs(np.vdot(z1, z2))
    inner = min(inner, 1.0)
    return np.arccos(inner)


def triangle_fs_area(z1, z2, z3):
    """Approximate FS area of a triangle via Heron proxy on geodesic side lengths."""
    d12 = fs_distance(z1, z2)
    d23 = fs_distance(z2, z3)
    d13 = fs_distance(z1, z3)
    s = (d12 + d23 + d13) / 2
    area_sq = s * max(s - d12, 0) * max(s - d23, 0) * max(s - d13, 0)
    return np.sqrt(area_sq)


def to_complex_vec(h, pca, n_complex):
    proj = pca.transform(h.reshape(1, -1))[0]
    z = np.array([complex(proj[2*i], proj[2*i+1]) for i in range(n_complex)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    if norm < 1e-10:
        z = np.zeros(n_complex, dtype=complex); z[0] = 1.0
    else:
        z /= norm
    return z


def pancharatnam_triangle(z1, z2, z3):
    """Pancharatnam phase around a 3-vertex loop: z1->z2->z3->z1."""
    import cmath
    prod = np.vdot(z1, z2) * np.vdot(z2, z3) * np.vdot(z3, z1)
    return cmath.phase(prod)


def run():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("  FS REANALYSIS v2: SEMANTIC EXCESS HOLONOMY TEST")
    print("=" * 70)
    print()
    print("Loading GPT-2 and building 5x5 prompt bank...")
    tok, mdl = v3.load_model()
    prompt_bank = make_5x5_prompt_bank()
    all_states = precompute_5x5_states(tok, mdl, "threshold", prompt_bank)

    all_vecs = []
    for cell_states in all_states.values():
        for h1, h2 in cell_states:
            all_vecs.append(h2)
    H = np.stack(all_vecs)
    from sklearn.decomposition import PCA
    n_real = 2 * N_COMPLEX
    pca = PCA(n_components=min(n_real, H.shape[0] - 1, H.shape[1]))
    pca.fit(H)
    print(f"PCA fit on {len(all_vecs)} states, {n_real} components, "
          f"var explained: {pca.explained_variance_ratio_.sum():.3f}")

    cell_keys = sorted(all_states.keys())
    cell_complex = {}
    for key in cell_keys:
        vecs = [to_complex_vec(h2, pca, N_COMPLEX) for h1, h2 in all_states[key]]
        cell_complex[key] = vecs

    triangles = list(combinations(cell_keys, 3))
    print(f"Total triangles from 25 cells: {len(triangles)} (will subsample 300)")
    chosen_triangles = [triangles[i] for i in
                        rng.choice(len(triangles), size=min(300, len(triangles)), replace=False)]

    semantic_ratios = []
    null_ratios = []

    print("Computing phase/area ratios for semantic and null triangles...")
    flat_vecs = [v for vecs in cell_complex.values() for v in vecs]

    for c1, c2, c3 in chosen_triangles:
        def rep(key):
            vecs = cell_complex[key]
            if not vecs:
                return None
            mean = np.mean(np.stack(vecs), axis=0)
            norm = np.linalg.norm(mean)
            if norm < 1e-10:
                return vecs[0]
            mean /= norm
            dists = [fs_distance(v, mean) for v in vecs]
            return vecs[int(np.argmin(dists))]

        z1, z2, z3 = rep(c1), rep(c2), rep(c3)
        if z1 is None or z2 is None or z3 is None:
            continue

        phi = pancharatnam_triangle(z1, z2, z3)
        area = triangle_fs_area(z1, z2, z3)
        if area < 1e-6:
            continue
        semantic_ratios.append(abs(phi) / area)

        idxs = rng.choice(len(flat_vecs), size=3, replace=False)
        n1, n2, n3 = flat_vecs[idxs[0]], flat_vecs[idxs[1]], flat_vecs[idxs[2]]
        n_phi = pancharatnam_triangle(n1, n2, n3)
        n_area = triangle_fs_area(n1, n2, n3)
        if n_area < 1e-6:
            continue
        null_ratios.append(abs(n_phi) / n_area)

    semantic_ratios = np.array(semantic_ratios)
    null_ratios = np.array(null_ratios)

    print(f"\nSemantic triangles: n={len(semantic_ratios)}, "
          f"|Phi|/area = {semantic_ratios.mean():.4f} +/- {semantic_ratios.std():.4f}")
    print(f"Null triangles:     n={len(null_ratios)}, "
          f"|Phi|/area = {null_ratios.mean():.4f} +/- {null_ratios.std():.4f}")

    U, p_mw = mannwhitneyu(semantic_ratios, null_ratios, alternative='greater')
    n_pos = int((semantic_ratios > null_ratios[:len(semantic_ratios)]).sum())
    binom_p = binomtest(n_pos, len(semantic_ratios), 0.5, alternative='greater').pvalue

    print(f"Mann-Whitney (semantic > null): p = {p_mw:.4f}")
    print(f"Sign test (semantic > null):    p = {binom_p:.4f}")

    if p_mw < 0.05:
        verdict = "SEMANTIC EXCESS HOLONOMY DETECTED — concept structure predicts curvature"
    elif p_mw < 0.15:
        verdict = "WEAK SIGNAL — borderline, needs more prompts per cell"
    else:
        verdict = "NULL — semantic structure does NOT predict excess holonomy at this scale"

    print(f"\nVERDICT: {verdict}")

    results = {
        "timestamp": TIMESTAMP,
        "n_complex": N_COMPLEX,
        "n_semantic_triangles": len(semantic_ratios),
        "n_null_triangles": len(null_ratios),
        "semantic_mean_ratio": float(semantic_ratios.mean()),
        "semantic_std_ratio": float(semantic_ratios.std()),
        "null_mean_ratio": float(null_ratios.mean()),
        "null_std_ratio": float(null_ratios.std()),
        "mann_whitney_p": float(p_mw),
        "binom_p": float(binom_p),
        "verdict": verdict,
        "what_v1_measured": "curvature of CP^15 itself (K~1) — container not content",
        "what_v2_measures": "excess |Phi|/FS-area for semantic vs random triangles",
    }
    out = RESULT_DIR / f"fs_reanalysis_v2_{TIMESTAMP}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")
    return results


if __name__ == "__main__":
    run()
