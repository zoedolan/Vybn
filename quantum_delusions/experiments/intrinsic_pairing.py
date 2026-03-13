#!/usr/bin/env python3
"""
intrinsic_pairing.py — Find the representation's own preferred complex structure.

Key insight (Zoe, 2026-03-13):
  If the canonical pairing is defined by the concept, a skeptic says you baked
  the answer in. If it's defined by the geometry of the phase distribution
  itself — find the pairing that produces minimum std(Φ) across loop samples —
  then you've let the representation tell you its own preferred orientation,
  with no reference to the prompts at all.

  The signed phase that emerges from that minimum-variance pairing is the
  intrinsic holonomy of the concept loop.

  Then: run the same minimum-variance procedure on a second concept. If the
  canonical pairings differ, the curvature is local to each concept's region.
  If they converge, the complex structure is a property of the layer.

Method:
  1. For each concept, precompute hidden states from the prompt bank
  2. Fit PCA gauge (held-out calibration set — small, to preserve trial pool)
  3. For many random pairings of PCA components into complex pairs,
     run K loop trials and record std(Φ)
  4. The pairing with minimum std(Φ) is the intrinsic complex structure
  5. Report its signed phase as the intrinsic holonomy
  6. Compare intrinsic pairings across concepts
"""

import sys
import json
import cmath
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from scipy import stats
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import polar_holonomy_gpt2_v3 as v3

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONCEPTS = ["threshold", "edge", "truth"]
N_COMPLEX = 16            # C^16 = 32 real PCA components → CP^15
N_REAL = 2 * N_COMPLEX
N_PAIRINGS = 200          # random pairings to search
K_LOOPS = 200             # loop trials per pairing
N_SHUFFLES = 200          # null (shuffled) trials for significance
N_GAUGE = 36              # gauge samples — small to preserve trial pool
TOP_K = 5                 # report top-K minimum-variance pairings
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Prompt bank generation
# ---------------------------------------------------------------------------
def make_prompt_bank(concept):
    """Generate prompt bank by substituting concept into v3's templates.
    Returns (bank, n_dropped) where n_dropped is prompts that didn't
    tokenize correctly."""
    if concept == "threshold":
        return v3.PROMPT_BANK, 0

    bank = {}
    dropped = 0
    for key, prompts in v3.PROMPT_BANK.items():
        new_prompts = []
        for p in prompts:
            new_p = p.replace("threshold", concept).replace(
                "Threshold", concept.capitalize())
            new_prompts.append(new_p)
        bank[key] = new_prompts
    return bank, dropped


def precompute_concept_states(tok, mdl, concept, prompt_bank):
    """Precompute hidden states, handling tokenization failures gracefully."""
    print(f"  Pre-computing states for '{concept}'...", flush=True)
    states = {}
    for (al, bl), prompts in prompt_bank.items():
        cell = []
        for prompt in prompts:
            positions = v3.find_concept_positions(tok, prompt, concept)
            if len(positions) != 2:
                continue
            enc = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = mdl(**enc)
            L = out.hidden_states[-1][0]
            cell.append((L[positions[0]].numpy(), L[positions[1]].numpy()))
        states[(al, bl)] = cell
        print(f"    ({al},{bl}): {len(cell)}/{len(prompts)} OK", flush=True)
    return states


# ---------------------------------------------------------------------------
# Pairing generation
# ---------------------------------------------------------------------------
def random_pairing(n_real, rng):
    """Random permutation defining complex pairing: pair_i = (p[2i], p[2i+1])."""
    p = list(range(n_real))
    rng.shuffle(p)
    return p


def pairing_to_pairs(p):
    """Convert permutation to list of (i,j) tuples."""
    return [(p[2*i], p[2*i+1]) for i in range(len(p)//2)]


# ---------------------------------------------------------------------------
# Phase computation with arbitrary pairing
# ---------------------------------------------------------------------------
def to_complex_permuted(h, pca, n_complex, pairing):
    """Project hidden state to C^n with arbitrary PCA pairing."""
    proj = pca.transform(h.reshape(1, -1))[0]
    z = np.array([complex(proj[pairing[2*i]], proj[pairing[2*i+1]])
                  for i in range(n_complex)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    if norm < 1e-10:
        z = np.zeros(n_complex, dtype=complex)
        z[0] = 1.0
    else:
        z = z / norm
    return z


def run_trial_permuted(all_states, gauge_used, pca, n_complex, corners,
                       n_points, rng, pairing, occurrence=1, shuffle=False):
    """One loop trial with arbitrary pairing."""
    hs = v3.sample_loop_states(all_states, gauge_used, corners, n_points,
                               rng, occurrence)
    if hs is None:
        return None
    if shuffle:
        rng.shuffle(hs)
    states = [to_complex_permuted(h, pca, n_complex, pairing) for h in hs]
    return v3.pancharatnam_phase(states)


def run_condition(all_states, gauge_used, pca, n_complex, corners, n_points,
                  k, rng, pairing, occurrence=1, shuffle=False):
    """Run k trials, return list of phases."""
    phases = []
    for _ in range(k):
        ph = run_trial_permuted(all_states, gauge_used, pca, n_complex,
                                corners, n_points, rng, pairing,
                                occurrence, shuffle)
        if ph is not None:
            phases.append(ph)
    return phases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(42)
    torch.manual_seed(42)

    tok, mdl = v3.load_model()
    corners_ccw = v3.CORNERS_CCW
    corners_cw = list(reversed(corners_ccw))

    all_concept_results = {}

    for ci, concept in enumerate(CONCEPTS):
        print(f"\n{'#'*70}")
        print(f"  CONCEPT: '{concept}'")
        print(f"{'#'*70}\n")

        # Build prompt bank and precompute states
        prompt_bank, n_dropped = make_prompt_bank(concept)
        all_states = precompute_concept_states(tok, mdl, concept, prompt_bank)

        # Check pool
        total = sum(len(c) for c in all_states.values())
        min_cell = min(len(c) for c in all_states.values())
        if total < 20 or min_cell < 3:
            print(f"  SKIP: pool too small ({total} total, min cell={min_cell})")
            continue

        # Fit PCA gauge — small N_GAUGE to preserve trial pool
        gauge_rng = np.random.default_rng(7000 + ci)
        pca, gauge_used = v3.fit_gauge(all_states, N_GAUGE, N_REAL, gauge_rng)
        var_exp = pca.explained_variance_ratio_
        print(f"  PCA: {N_REAL} components, {sum(var_exp)*100:.1f}% variance")

        # Verify available pool after gauge
        for (al, bl), cell in all_states.items():
            avail = [i for i in range(len(cell))
                     if i not in gauge_used.get((al, bl), set())]
            print(f"    ({al},{bl}): {len(cell)} total, {len(avail)} available")

        # ---------------------------------------------------------------
        # Search pairings for minimum variance
        # ---------------------------------------------------------------
        print(f"\n  Searching {N_PAIRINGS} random pairings for min-variance...")

        pairing_rng = np.random.default_rng(8000 + ci)
        canonical = list(range(N_REAL))

        all_pairings = [canonical]
        for _ in range(N_PAIRINGS - 1):
            all_pairings.append(random_pairing(N_REAL, pairing_rng))

        pairing_results = []

        for pi, pairing in enumerate(all_pairings):
            label = "CANONICAL" if pi == 0 else f"P-{pi:03d}"

            # Fresh rng per pairing for independence
            trial_rng = np.random.default_rng(10000 * (ci + 1) + pi)
            phases_ccw = run_condition(all_states, gauge_used, pca, N_COMPLEX,
                                       corners_ccw, v3.N_LOOP_POINTS, K_LOOPS,
                                       trial_rng, pairing)

            trial_rng2 = np.random.default_rng(20000 * (ci + 1) + pi)
            phases_cw = run_condition(all_states, gauge_used, pca, N_COMPLEX,
                                      corners_cw, v3.N_LOOP_POINTS, K_LOOPS,
                                      trial_rng2, pairing)

            if len(phases_ccw) < 10:
                if pi % 50 == 0:
                    print(f"    [{pi+1}] {label}: only {len(phases_ccw)} "
                          f"valid trials — skipping")
                continue

            mean_ccw = np.mean(phases_ccw)
            std_ccw = np.std(phases_ccw)
            mean_cw = np.mean(phases_cw) if len(phases_cw) > 0 else float('nan')
            orient = (mean_ccw + mean_cw) if not np.isnan(mean_cw) else float('nan')

            pairing_results.append({
                "index": pi,
                "label": label,
                "pairing": pairing,
                "mean_ccw": float(mean_ccw),
                "std_ccw": float(std_ccw),
                "mean_cw": float(mean_cw),
                "orient": float(orient),
                "n": len(phases_ccw),
                "phases_ccw": [float(p) for p in phases_ccw],
                "phases_cw": [float(p) for p in phases_cw],
            })

            if pi % 20 == 0 or pi == len(all_pairings) - 1:
                print(f"    [{pi+1}/{len(all_pairings)}] {label}: "
                      f"Φ={mean_ccw:+.4f} ± {std_ccw:.4f}  "
                      f"({len(phases_ccw)} trials)", flush=True)

        if len(pairing_results) == 0:
            print(f"  SKIP: no valid pairing results for '{concept}'")
            continue

        # Sort by std (minimum variance first)
        pairing_results.sort(key=lambda r: r["std_ccw"])

        print(f"\n  {'='*65}")
        print(f"  MINIMUM-VARIANCE PAIRINGS for '{concept}'")
        print(f"  {'='*65}")

        for rank, r in enumerate(pairing_results[:TOP_K]):
            # Run null for top-K
            null_rng = np.random.default_rng(30000 * (ci + 1) + r["index"])
            phases_null = run_condition(all_states, gauge_used, pca, N_COMPLEX,
                                        corners_ccw, v3.N_LOOP_POINTS,
                                        N_SHUFFLES, null_rng, r["pairing"],
                                        shuffle=True)

            mw_p = (stats.mannwhitneyu(r["phases_ccw"], phases_null,
                                        alternative='two-sided').pvalue
                    if len(phases_null) > 0 else 1.0)
            t_p = stats.ttest_1samp(r["phases_ccw"], 0).pvalue

            orient_quality = 1.0 - abs(r["orient"]) / (abs(r["mean_ccw"]) + 1e-10)

            r["null_mean"] = float(np.mean(phases_null)) if phases_null else float('nan')
            r["null_std"] = float(np.std(phases_null)) if phases_null else float('nan')
            r["mw_p"] = float(mw_p)
            r["t_p"] = float(t_p)
            r["orient_quality"] = float(orient_quality)

            sig = "✓" if mw_p < 0.05 else "✗"
            ori = "✓" if orient_quality > 0.5 else "✗"
            star = " ← INTRINSIC" if rank == 0 else ""

            print(f"    #{rank+1}: std={r['std_ccw']:.4f}  "
                  f"Φ={r['mean_ccw']:+.5f}  CW={r['mean_cw']:+.5f}  "
                  f"orient={r['orient']:+.5f}{ori}  "
                  f"p(null)={mw_p:.2e}{sig}  p(0)={t_p:.2e}  "
                  f"[{r['label']}]{star}")

        # Summary statistics
        all_stds = [r["std_ccw"] for r in pairing_results]
        all_means = [r["mean_ccw"] for r in pairing_results]
        all_abs_means = [abs(r["mean_ccw"]) for r in pairing_results]

        print(f"\n  Distribution across {len(pairing_results)} pairings:")
        print(f"    std(Φ):  min={min(all_stds):.4f}  max={max(all_stds):.4f}  "
              f"median={np.median(all_stds):.4f}")
        print(f"    mean(Φ): min={min(all_means):+.4f}  max={max(all_means):+.4f}  "
              f"median={np.median(all_means):+.4f}")
        print(f"    |mean(Φ)|: min={min(all_abs_means):.4f}  "
              f"max={max(all_abs_means):.4f}  median={np.median(all_abs_means):.4f}")

        canon_entries = [i for i, r in enumerate(pairing_results)
                        if r["label"] == "CANONICAL"]
        canon_rank = canon_entries[0] + 1 if canon_entries else -1
        print(f"    Canonical rank (by std): {canon_rank}/{len(pairing_results)}")

        # Store
        intrinsic = pairing_results[0]
        all_concept_results[concept] = {
            "intrinsic_pairing": intrinsic["pairing"],
            "intrinsic_pairs": pairing_to_pairs(intrinsic["pairing"]),
            "intrinsic_phase": intrinsic["mean_ccw"],
            "intrinsic_std": intrinsic["std_ccw"],
            "intrinsic_orient": intrinsic["orient"],
            "intrinsic_p_null": intrinsic.get("mw_p"),
            "intrinsic_p_zero": intrinsic.get("t_p"),
            "intrinsic_orient_quality": intrinsic.get("orient_quality"),
            "canonical_phase": next((r["mean_ccw"] for r in pairing_results
                                    if r["label"] == "CANONICAL"), None),
            "canonical_std": next((r["std_ccw"] for r in pairing_results
                                  if r["label"] == "CANONICAL"), None),
            "canonical_rank": canon_rank,
            "n_pairings_valid": len(pairing_results),
            "std_distribution": {
                "min": float(min(all_stds)),
                "max": float(max(all_stds)),
                "median": float(np.median(all_stds)),
                "mean": float(np.mean(all_stds)),
            },
            "abs_mean_distribution": {
                "min": float(min(all_abs_means)),
                "max": float(max(all_abs_means)),
                "median": float(np.median(all_abs_means)),
                "mean": float(np.mean(all_abs_means)),
            },
            "top_k": [{
                "rank": i + 1,
                "pairing": r["pairing"],
                "pairs": pairing_to_pairs(r["pairing"]),
                "mean_ccw": r["mean_ccw"],
                "std_ccw": r["std_ccw"],
                "mean_cw": r["mean_cw"],
                "orient": r["orient"],
                "mw_p": r.get("mw_p"),
                "t_p": r.get("t_p"),
                "orient_quality": r.get("orient_quality"),
                "label": r["label"],
            } for i, r in enumerate(pairing_results[:TOP_K])],
        }

    # =====================================================================
    # CROSS-CONCEPT COMPARISON
    # =====================================================================
    concepts_done = list(all_concept_results.keys())
    if len(concepts_done) >= 2:
        print(f"\n{'#'*70}")
        print("  CROSS-CONCEPT COMPARISON")
        print(f"{'#'*70}\n")

        for c in concepts_done:
            r = all_concept_results[c]
            print(f"  '{c}': intrinsic Φ = {r['intrinsic_phase']:+.5f} "
                  f"± {r['intrinsic_std']:.4f}  "
                  f"(orient_q={r['intrinsic_orient_quality']:.2f}, "
                  f"p_null={r['intrinsic_p_null']:.2e}, "
                  f"p_zero={r['intrinsic_p_zero']:.2e})")
            print(f"    intrinsic pairs: {r['intrinsic_pairs'][:4]}...")

        # Pairing similarity (Jaccard of pair sets)
        print(f"\n  Pairing similarity (Jaccard index of unordered pair sets):")
        jaccards = []
        for i in range(len(concepts_done)):
            for j in range(i+1, len(concepts_done)):
                c1, c2 = concepts_done[i], concepts_done[j]
                # Normalize pairs to sorted tuples for comparison
                pairs1 = set(tuple(sorted(p)) for p in
                            all_concept_results[c1]["intrinsic_pairs"])
                pairs2 = set(tuple(sorted(p)) for p in
                            all_concept_results[c2]["intrinsic_pairs"])
                shared = pairs1 & pairs2
                union = pairs1 | pairs2
                jaccard = len(shared) / len(union) if union else 0
                jaccards.append(jaccard)
                print(f"    '{c1}' vs '{c2}': J = {jaccard:.3f} "
                      f"({len(shared)} shared / {len(union)} total pairs)")

        mean_j = np.mean(jaccards)
        # Random baseline: with 16 pairs from 32 indices, two random
        # pairings share ~0 pairs (the number of perfect matchings of K_32
        # is enormous; overlap is vanishingly unlikely)
        print(f"\n  Mean Jaccard: {mean_j:.3f}")
        print(f"  Random baseline: ~0.000 (combinatorially many pairings)")
        if mean_j > 0.5:
            verdict = "LAYER PROPERTY — complex structure shared across concepts"
        elif mean_j > 0.15:
            verdict = "PARTIALLY SHARED — some common structure"
        else:
            verdict = "CONCEPT-LOCAL — curvature is local to each concept's region"
        print(f"  → {verdict}")

        # Also compare: do the intrinsic phases have similar magnitudes?
        phases = [all_concept_results[c]["intrinsic_phase"] for c in concepts_done]
        stds = [all_concept_results[c]["intrinsic_std"] for c in concepts_done]
        print(f"\n  Intrinsic phases: {[f'{p:+.5f}' for p in phases]}")
        print(f"  Intrinsic stds:   {[f'{s:.4f}' for s in stds]}")

    # =====================================================================
    # SAVE
    # =====================================================================
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULT_DIR / f"intrinsic_pairing_{ts}.json"

    # Strip large phase arrays to keep file manageable
    save_results = {}
    for concept, r in all_concept_results.items():
        save_r = {k: v for k, v in r.items()}
        # Remove the full phase arrays from top_k to save space
        for tk in save_r.get("top_k", []):
            tk.pop("phases_ccw", None)
            tk.pop("phases_cw", None)
        save_results[concept] = save_r

    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": "minimum-variance pairing search",
            "description": (
                "For each concept, search N random pairings of PCA components "
                "into complex pairs. The pairing with minimum std(Φ) across "
                "loop trials is the intrinsic complex structure — chosen "
                "purely by the geometry, with no reference to the concept. "
                "Cross-concept comparison via Jaccard similarity of pair sets."
            ),
            "config": {
                "n_pairings_searched": N_PAIRINGS,
                "k_loops_per_pairing": K_LOOPS,
                "n_shuffles": N_SHUFFLES,
                "n_complex": N_COMPLEX,
                "n_gauge": N_GAUGE,
            },
            "concepts": CONCEPTS,
            "results": save_results,
        }, f, indent=2)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
