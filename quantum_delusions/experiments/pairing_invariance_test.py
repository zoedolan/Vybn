#!/usr/bin/env python3
"""
Two falsification tests for the polar holonomy v3 result.
Imports directly from v3 to ensure identical prompt bank and functions.
"""

import numpy as np
import cmath
import json
import random
import torch
from pathlib import Path
from datetime import datetime, timezone
from scipy import stats
import sys
sys.path.insert(0, str(Path(__file__).parent))
import polar_holonomy_gpt2_v3 as v3

N_PERMUTATIONS = 20
K_LOOPS = 200
N_SHUFFLES = 200


def to_complex_permuted(h, pca, n_complex, pairing):
    """Like v3.to_complex_vector but with arbitrary PCA component pairing."""
    proj = pca.transform(h.reshape(1, -1))[0]
    z = np.array([complex(proj[pairing[2*i]], proj[pairing[2*i+1]]) for i in range(n_complex)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    if norm < 1e-10:
        z = np.zeros(n_complex, dtype=complex); z[0] = 1.0
    else:
        z = z / norm
    return z


def run_trial_permuted(all_states, gauge_used, pca, n_complex, corners, n_points, rng, pairing, occurrence=1, shuffle=False):
    """v3.run_trial but with permuted pairing."""
    hs = v3.sample_loop_states(all_states, gauge_used, corners, n_points, rng, occurrence)
    if hs is None:
        return None
    if shuffle:
        rng.shuffle(hs)
    states = [to_complex_permuted(h, pca, n_complex, pairing) for h in hs]
    return v3.pancharatnam_phase(states)


def run_condition(all_states, gauge_used, pca, n_complex, corners, n_points, k, master_rng, pairing, occurrence=1, shuffle=False):
    """Run k trials. Uses master_rng to create a fresh sub-rng (for reproducibility matching v3)."""
    phases = []
    for _ in range(k):
        ph = run_trial_permuted(all_states, gauge_used, pca, n_complex, corners, n_points, master_rng, pairing, occurrence, shuffle)
        if ph is not None:
            phases.append(ph)
    return phases


def main():
    rng = np.random.default_rng(42)
    random.seed(42)
    torch.manual_seed(42)
    
    tok, mdl = v3.load_model()
    all_states = v3.precompute_all_states(tok, mdl)
    
    n_complex = 16
    n_real = 2 * n_complex
    
    # Replicate v3's approach: iterate through dims to advance rng to the C^16 state.
    # v3 processes [2, 4, 8, 16] sequentially with shared rng.
    # We need to advance the rng through the first 3 dims to match v3's C^16 gauge.
    for nc in [2, 4, 8]:
        nr = 2 * nc
        _pca, _gu = v3.fit_gauge(all_states, v3.N_GAUGE_SAMPLES, nr, rng)
        # Advance rng through K_LOOPS trials for each condition
        for _ in range(v3.K_LOOPS):
            v3.run_trial(all_states, _gu, _pca, nc, v3.CORNERS_CCW, v3.N_LOOP_POINTS, rng)
        corners_cw = list(reversed(v3.CORNERS_CCW))
        for _ in range(v3.K_LOOPS):
            v3.run_trial(all_states, _gu, _pca, nc, corners_cw, v3.N_LOOP_POINTS, rng)
        for _ in range(v3.K_LOOPS):
            v3.run_trial(all_states, _gu, _pca, nc, v3.CORNERS_TALL, v3.N_LOOP_POINTS, rng)
        for _ in range(v3.K_LOOPS):
            v3.run_trial(all_states, _gu, _pca, nc, v3.CORNERS_CCW, 4, rng)
        for _ in range(v3.N_SHUFFLES):
            v3.run_trial(all_states, _gu, _pca, nc, v3.CORNERS_CCW, v3.N_LOOP_POINTS, rng, shuffle=True)
        for _ in range(v3.K_LOOPS):
            v3.run_trial(all_states, _gu, _pca, nc, v3.CORNERS_CCW, v3.N_LOOP_POINTS, rng, occurrence=0)
    
    # Now rng is at the same state as v3 when it reaches C^16
    pca, gauge_used = v3.fit_gauge(all_states, v3.N_GAUGE_SAMPLES, n_real, rng)
    var_exp = pca.explained_variance_ratio_
    print(f"\nPCA: {n_real} components, {sum(var_exp)*100:.1f}% variance")
    
    for (al, bl), cell in all_states.items():
        avail = [i for i in range(len(cell)) if i not in gauge_used.get((al, bl), set())]
        print(f"  ({al},{bl}): {len(cell)} total, {len(avail)} available")
    
    corners_cw = list(reversed(v3.CORNERS_CCW))
    canonical = list(range(n_real))
    
    # First: reproduce v3's canonical result to verify
    print("\nReproducing v3 canonical result...")
    # Save rng state so we can replay
    rng_state = rng.bit_generator.state
    
    phases_ccw_v3 = run_condition(all_states, gauge_used, pca, n_complex, v3.CORNERS_CCW, v3.N_LOOP_POINTS, K_LOOPS, rng, canonical)
    phases_cw_v3 = run_condition(all_states, gauge_used, pca, n_complex, corners_cw, v3.N_LOOP_POINTS, K_LOOPS, rng, canonical)
    
    print(f"  Φ_CCW = {np.mean(phases_ccw_v3):+.4f} ± {np.std(phases_ccw_v3):.4f} (v3 reported: -0.0965 ± 0.1533)")
    print(f"  Φ_CW  = {np.mean(phases_cw_v3):+.4f} (v3 reported: +0.0803)")
    print(f"  n_ccw = {len(phases_ccw_v3)}, n_cw = {len(phases_cw_v3)}")
    
    # =====================================================================
    # TEST 1: PCA PAIRING INVARIANCE
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 1: PCA PAIRING INVARIANCE")
    print(f"{'='*70}\n")
    
    # For each permutation, use fresh rng seeds to get comparable (but independent) samples
    perm_rng = np.random.default_rng(2026)
    permutations = [canonical]
    for _ in range(N_PERMUTATIONS):
        p = list(range(n_real))
        perm_rng.shuffle(p)
        permutations.append(p)
    
    results_t1 = []
    for pi, pairing in enumerate(permutations):
        label = "CANONICAL" if pi == 0 else f"PERM-{pi:02d}"
        
        # Use fresh rngs per permutation for fair comparison
        r1 = np.random.default_rng(1000 + pi)
        ph_ccw = run_condition(all_states, gauge_used, pca, n_complex, v3.CORNERS_CCW, v3.N_LOOP_POINTS, K_LOOPS, r1, pairing)
        r2 = np.random.default_rng(2000 + pi)
        ph_cw = run_condition(all_states, gauge_used, pca, n_complex, corners_cw, v3.N_LOOP_POINTS, K_LOOPS, r2, pairing)
        r3 = np.random.default_rng(3000 + pi)
        ph_null = run_condition(all_states, gauge_used, pca, n_complex, v3.CORNERS_CCW, v3.N_LOOP_POINTS, N_SHUFFLES, r3, pairing, shuffle=True)
        
        m_ccw = np.mean(ph_ccw) if ph_ccw else float('nan')
        m_cw = np.mean(ph_cw) if ph_cw else float('nan')
        s_ccw = np.std(ph_ccw) if ph_ccw else float('nan')
        orient = m_ccw + m_cw
        
        mw_p = stats.mannwhitneyu(ph_ccw, ph_null, alternative='two-sided').pvalue if ph_ccw and ph_null else 1.0
        t_p = stats.ttest_1samp(ph_ccw, 0).pvalue if len(ph_ccw) > 1 else 1.0
        
        results_t1.append({"label": label, "mean_ccw": float(m_ccw), "std_ccw": float(s_ccw),
                           "mean_cw": float(m_cw), "orient": float(orient), 
                           "mw_p": float(mw_p), "t_p": float(t_p), "n": len(ph_ccw)})
        
        o_ok = "✓" if abs(orient) < abs(m_ccw) * 0.5 else "✗"
        s_ok = "✓" if mw_p < 0.05 else "✗"
        print(f"  {label:12s}  Φ={m_ccw:+.4f}±{s_ccw:.4f}  CW={m_cw:+.4f}  "
              f"ori={orient:+.4f}{o_ok}  p={mw_p:.2e}{s_ok}  p(0)={t_p:.2e}")
    
    all_m = [r["mean_ccw"] for r in results_t1]
    canon = all_m[0]
    perm_m = all_m[1:]
    n_sig = sum(1 for r in results_t1[1:] if r["mw_p"] < 0.05)
    n_ori = sum(1 for r in results_t1[1:] if abs(r["orient"]) < abs(r["mean_ccw"]) * 0.5 and abs(r["mean_ccw"]) > 0.01)
    n_sign = sum(1 for m in perm_m if np.sign(m) == np.sign(canon)) if not np.isnan(canon) and canon != 0 else 0
    
    print(f"\n  Canonical: {canon:+.4f}")
    print(f"  Permutations: mean={np.mean(perm_m):+.4f}, std={np.std(perm_m):.4f}")
    print(f"  Sig p<0.05: {n_sig}/{N_PERMUTATIONS}")
    print(f"  Orient flip: {n_ori}/{N_PERMUTATIONS}")
    print(f"  Same sign: {n_sign}/{N_PERMUTATIONS}")
    
    if n_sig >= N_PERMUTATIONS * 0.7:
        v1 = "INVARIANT"
    elif n_sig >= N_PERMUTATIONS * 0.3:
        v1 = "PARTIAL"
    else:
        v1 = "ARTIFACT"
    print(f"\n  *** VERDICT: {v1} ***\n")
    
    # =====================================================================
    # TEST 2: FIRST-OCCURRENCE ORIENTATION SYMMETRY
    # =====================================================================
    print(f"{'='*70}")
    print("TEST 2: FIRST-OCCURRENCE ORIENTATION SYMMETRY")
    print(f"{'='*70}\n")
    
    r1 = np.random.default_rng(5000)
    ph_2ccw = run_condition(all_states, gauge_used, pca, n_complex, v3.CORNERS_CCW, v3.N_LOOP_POINTS, K_LOOPS, r1, canonical, occurrence=1)
    r2 = np.random.default_rng(5001)
    ph_2cw = run_condition(all_states, gauge_used, pca, n_complex, corners_cw, v3.N_LOOP_POINTS, K_LOOPS, r2, canonical, occurrence=1)
    r3 = np.random.default_rng(5002)
    ph_1ccw = run_condition(all_states, gauge_used, pca, n_complex, v3.CORNERS_CCW, v3.N_LOOP_POINTS, K_LOOPS, r3, canonical, occurrence=0)
    r4 = np.random.default_rng(5003)
    ph_1cw = run_condition(all_states, gauge_used, pca, n_complex, corners_cw, v3.N_LOOP_POINTS, K_LOOPS, r4, canonical, occurrence=0)
    r5 = np.random.default_rng(5004)
    ph_1null = run_condition(all_states, gauge_used, pca, n_complex, v3.CORNERS_CCW, v3.N_LOOP_POINTS, N_SHUFFLES, r5, canonical, occurrence=0, shuffle=True)
    
    m2c, m2w = np.mean(ph_2ccw), np.mean(ph_2cw)
    m1c, m1w = np.mean(ph_1ccw), np.mean(ph_1cw)
    m1n = np.mean(ph_1null)
    s2, s1, sn = np.std(ph_2ccw), np.std(ph_1ccw), np.std(ph_1null)
    o2, o1 = m2c + m2w, m1c + m1w
    
    mw1p = stats.mannwhitneyu(ph_1ccw, ph_1null, alternative='two-sided').pvalue if ph_1ccw and ph_1null else 1.0
    mw1ap = stats.mannwhitneyu(np.abs(ph_1ccw), np.abs(ph_1null), alternative='two-sided').pvalue if ph_1ccw and ph_1null else 1.0
    
    print(f"  2nd occ: CCW={m2c:+.4f}±{s2:.4f}  CW={m2w:+.4f}  orient={o2:+.4f}  flip={(1-abs(o2)/(abs(m2c)+1e-10))*100:.0f}%")
    print(f"  1st occ: CCW={m1c:+.4f}±{s1:.4f}  CW={m1w:+.4f}  orient={o1:+.4f}  flip={(1-abs(o1)/(abs(m1c)+1e-10))*100:.0f}%" if abs(m1c)>0.01 else f"  1st occ: CCW={m1c:+.4f}±{s1:.4f}  CW={m1w:+.4f}  orient={o1:+.4f}")
    print(f"  1st null: {m1n:+.4f}±{sn:.4f}")
    print(f"  |Φ_1st| = {np.mean(np.abs(ph_1ccw)):.4f}  |Φ_2nd| = {np.mean(np.abs(ph_2ccw)):.4f}  |Φ_null| = {np.mean(np.abs(ph_1null)):.4f}")
    print(f"  1st vs null: p={mw1p:.2e}  |1st| vs |null|: p={mw1ap:.2e}")
    print(f"  std: 1st={s1:.4f}  2nd={s2:.4f}  null={sn:.4f}  ratio={s1/(s2+1e-10):.2f}")
    
    orient_ok = abs(o1) < abs(m1c) * 0.5 if abs(m1c) > 0.01 else False
    if orient_ok and mw1p < 0.05:
        v2 = "GEOMETRIC"
    elif mw1p < 0.05:
        v2 = "AMBIGUOUS"
    elif s1 > s2 * 1.5:
        v2 = "VARIANCE"
    else:
        v2 = "NULL"
    print(f"\n  *** VERDICT: {v2} ***\n")
    
    # Save
    out = Path(__file__).parent / "results" / f"pairing_invariance_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(out, "w") as f:
        json.dump({"timestamp": datetime.now(timezone.utc).isoformat(),
                    "test_1": {"canonical": float(canon), "perm_mean": float(np.mean(perm_m)),
                               "perm_std": float(np.std(perm_m)), "n_sig": n_sig,
                               "n_orient": n_ori, "verdict": v1, "details": results_t1},
                    "test_2": {"m2ccw": float(m2c), "m2cw": float(m2w), "o2": float(o2),
                               "m1ccw": float(m1c), "m1cw": float(m1w), "o1": float(o1),
                               "m1null": float(m1n), "s1": float(s1), "s2": float(s2),
                               "sn": float(sn), "p_signed": float(mw1p),
                               "p_abs": float(mw1ap), "verdict": v2}}, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
