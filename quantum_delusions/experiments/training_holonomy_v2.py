#!/usr/bin/env python3
"""
training_holonomy_v2.py — Proper rectangular loop in gradient-space
====================================================================

The v1 experiment measured diffusion (CW ≈ CCW because there's no inverse
step — every step is forward). This v2 implements the actual Gödel rectangle:

  CW:  +A → +B → −A → −B    (learn A, learn B, unlearn A, unlearn B)
  CCW: +B → +A → −B → −A    (learn B, learn A, unlearn B, unlearn A)

If gradient operations commute (flat geometry), steps 3-4 exactly undo
steps 1-2 and the model returns to θ₀. If they don't (curved geometry),
the gap θ_final − θ₀ is the holonomy.

The CW and CCW loops traverse the SAME rectangle in OPPOSITE orientations.
If there is curvature: holonomy_CW ≈ −holonomy_CCW (opposite vectors).
If there is no curvature: both ≈ 0 (or random, orientation-independent).

The decisive tests:
  1. Gap magnitude > 0 (the loop doesn't close)
  2. CW and CCW holonomy vectors are ANTI-CORRELATED (orientation sensitivity)
  3. Gap scales with k² where k = steps per side (area law, Stokes theorem)
  4. Line controls (no area enclosed) have gap ≈ 0
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr

RESULT_DIR = Path(__file__).parent / "quantum_delusions" / "experiments" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class TinyNet(nn.Module):
    def __init__(self, d_in=8, d_hidden=32, d_out=4):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def get_params(net):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in net.parameters()])


def set_params(net, vec):
    offset = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(torch.tensor(vec[offset:offset+n].reshape(p.shape), dtype=p.dtype))
            offset += n


def make_concept_data(concept_id, n_samples=64, d_in=8, d_out=4, seed=0):
    """Two concepts with deliberately different structure."""
    rng = np.random.default_rng(seed * 1000 + concept_id * 137)
    # Make the concepts genuinely different — different input distributions
    # AND different target mappings
    if concept_id == 0:
        W = rng.standard_normal((d_in, d_out)) * 0.5
        X = rng.standard_normal((n_samples, d_in)).astype(np.float32)
    else:
        W = rng.standard_normal((d_in, d_out)) * 0.5
        # Different input distribution (shifted, scaled)
        X = (rng.standard_normal((n_samples, d_in)) * 1.5 + 0.5).astype(np.float32)
    Y = (X @ W).astype(np.float32)
    return torch.tensor(X), torch.tensor(Y)


def train_steps(net, X, Y, lr, n_steps):
    """Take n_steps of SGD with given lr (can be negative for gradient ascent)."""
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(n_steps):
        net.zero_grad()
        pred = net(X)
        loss = loss_fn(pred, Y)
        losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            for p in net.parameters():
                p -= lr * p.grad  # manual SGD step (negative lr = ascent)
    return losses


def run_rectangular_loop(initial_params, orientation, k_steps, lr, data_seed,
                         d_in=8, d_hidden=32, d_out=4):
    """
    Run a rectangular loop in gradient-space.
    orientation='CW':   +A → +B → −A → −B
    orientation='CCW':  +B → +A → −B → −A
    orientation='LINE': +A → −A (zero area control)
    
    Returns: gap magnitude, gap vector, total losses
    """
    net = TinyNet(d_in, d_hidden, d_out)
    set_params(net, initial_params)
    
    X_A, Y_A = make_concept_data(0, seed=data_seed, d_in=d_in, d_out=d_out)
    X_B, Y_B = make_concept_data(1, seed=data_seed, d_in=d_in, d_out=d_out)
    
    all_losses = []
    
    if orientation == 'CW':
        all_losses += train_steps(net, X_A, Y_A, +lr, k_steps)  # +A
        all_losses += train_steps(net, X_B, Y_B, +lr, k_steps)  # +B
        all_losses += train_steps(net, X_A, Y_A, -lr, k_steps)  # −A
        all_losses += train_steps(net, X_B, Y_B, -lr, k_steps)  # −B
    elif orientation == 'CCW':
        all_losses += train_steps(net, X_B, Y_B, +lr, k_steps)  # +B
        all_losses += train_steps(net, X_A, Y_A, +lr, k_steps)  # +A
        all_losses += train_steps(net, X_B, Y_B, -lr, k_steps)  # −B
        all_losses += train_steps(net, X_A, Y_A, -lr, k_steps)  # −A
    elif orientation == 'LINE':
        all_losses += train_steps(net, X_A, Y_A, +lr, k_steps)  # +A
        all_losses += train_steps(net, X_A, Y_A, -lr, k_steps)  # −A
    
    theta_final = get_params(net)
    gap_vec = theta_final - initial_params
    gap_mag = float(np.linalg.norm(gap_vec))
    
    return gap_mag, gap_vec, all_losses


def run_experiment():
    print("=" * 70, flush=True)
    print("Training Holonomy v2 — Rectangular Loop in Gradient-Space", flush=True)
    print("=" * 70, flush=True)
    print("CW:  +A → +B → −A → −B", flush=True)
    print("CCW: +B → +A → −B → −A", flush=True)
    print("LINE: +A → −A (zero-area control)\n", flush=True)
    
    d_in, d_hidden, d_out = 8, 32, 4
    lr = 0.005
    k_steps = 20
    n_trials = 300
    
    rng = np.random.default_rng(42)
    torch.manual_seed(0)
    net0 = TinyNet(d_in, d_hidden, d_out)
    theta_init = get_params(net0)
    n_params = len(theta_init)
    
    print(f"Network: {n_params} parameters", flush=True)
    print(f"lr={lr}, k_steps={k_steps}, n_trials={n_trials}\n", flush=True)
    
    results = {
        "timestamp": TIMESTAMP,
        "experiment": "training_holonomy_v2_rectangular",
        "n_params": n_params,
        "lr": lr,
        "k_steps": k_steps,
        "n_trials": n_trials,
    }
    
    # --- Test 1: Orientation dependence ---
    print("--- Test 1: Orientation dependence (CW vs CCW vs LINE) ---", flush=True)
    
    cw_mags, ccw_mags, line_mags = [], [], []
    cw_vecs, ccw_vecs = [], []
    cosines = []  # cosine between CW and CCW holonomy vectors
    
    for trial in range(n_trials):
        ds = int(rng.integers(0, 100000))
        
        cw_mag, cw_vec, _ = run_rectangular_loop(theta_init, 'CW', k_steps, lr, ds, d_in, d_hidden, d_out)
        ccw_mag, ccw_vec, _ = run_rectangular_loop(theta_init, 'CCW', k_steps, lr, ds, d_in, d_hidden, d_out)
        line_mag, _, _ = run_rectangular_loop(theta_init, 'LINE', k_steps, lr, ds, d_in, d_hidden, d_out)
        
        cw_mags.append(cw_mag)
        ccw_mags.append(ccw_mag)
        line_mags.append(line_mag)
        
        # Cosine similarity between CW and CCW holonomy vectors
        # If curvature: should be ≈ −1 (opposite directions)
        # If noise: should be ≈ 0 (random)
        norm_prod = np.linalg.norm(cw_vec) * np.linalg.norm(ccw_vec)
        if norm_prod > 1e-12:
            cos = float(np.dot(cw_vec, ccw_vec) / norm_prod)
            cosines.append(cos)
        
        if (trial + 1) % 100 == 0:
            print(f"  trial {trial+1}/{n_trials}", flush=True)
    
    cw_mags = np.array(cw_mags)
    ccw_mags = np.array(ccw_mags)
    line_mags = np.array(line_mags)
    cosines = np.array(cosines)
    
    print(f"\n  CW gap:   mean={np.mean(cw_mags):.6f}  std={np.std(cw_mags):.6f}")
    print(f"  CCW gap:  mean={np.mean(ccw_mags):.6f}  std={np.std(ccw_mags):.6f}")
    print(f"  LINE gap: mean={np.mean(line_mags):.6f}  std={np.std(line_mags):.6f}")
    print(f"  CW/CCW cosine: mean={np.mean(cosines):.4f}  std={np.std(cosines):.4f}")
    print(f"    (−1 = opposite = curvature, 0 = random = noise, +1 = same = no info)")
    
    # Key test: is CW/CCW cosine significantly negative?
    from scipy.stats import ttest_1samp
    t_cos, p_cos = ttest_1samp(cosines, 0.0)
    print(f"  cosine vs 0: t={t_cos:.3f}, p={p_cos:.4e}")
    
    # Is line gap significantly smaller than CW/CCW gap?
    from scipy.stats import ttest_ind
    t_line, p_line = ttest_ind(cw_mags, line_mags, equal_var=False)
    print(f"  CW vs LINE: t={t_line:.3f}, p={p_line:.4e}")
    
    # CW magnitude vs CCW magnitude
    t_mag, p_mag = ttest_ind(cw_mags, ccw_mags, equal_var=False)
    print(f"  CW mag vs CCW mag: t={t_mag:.3f}, p={p_mag:.4e}")
    
    results.update({
        "cw_mag_mean": float(np.mean(cw_mags)), "cw_mag_std": float(np.std(cw_mags)),
        "ccw_mag_mean": float(np.mean(ccw_mags)), "ccw_mag_std": float(np.std(ccw_mags)),
        "line_mag_mean": float(np.mean(line_mags)), "line_mag_std": float(np.std(line_mags)),
        "cosine_mean": float(np.mean(cosines)), "cosine_std": float(np.std(cosines)),
        "cosine_vs_zero_t": float(t_cos), "cosine_vs_zero_p": float(p_cos),
        "cw_vs_line_t": float(t_line), "cw_vs_line_p": float(p_line),
        "cw_vs_ccw_mag_p": float(p_mag),
    })
    
    # --- Test 2: Area scaling ---
    print(f"\n--- Test 2: Area scaling (gap vs k²) ---", flush=True)
    area_k = [3, 5, 10, 20, 40]
    area_means_cw = []
    area_means_line = []
    n_area_trials = 80
    
    for k in area_k:
        gaps_cw = []
        gaps_line = []
        for _ in range(n_area_trials):
            ds = int(rng.integers(0, 100000))
            g, _, _ = run_rectangular_loop(theta_init, 'CW', k, lr, ds, d_in, d_hidden, d_out)
            gaps_cw.append(g)
            g, _, _ = run_rectangular_loop(theta_init, 'LINE', k, lr, ds, d_in, d_hidden, d_out)
            gaps_line.append(g)
        mcw = float(np.mean(gaps_cw))
        mli = float(np.mean(gaps_line))
        area_means_cw.append(mcw)
        area_means_line.append(mli)
        print(f"  k={k:3d}: CW_gap={mcw:.6f}  LINE_gap={mli:.6f}  ratio={mcw/max(mli,1e-12):.2f}")
    
    # For true holonomy: gap ~ k² (area of rectangle)
    # For diffusion: gap ~ k (perimeter / number of steps)
    # Test: does gap correlate better with k² than k?
    k_arr = np.array(area_k, dtype=float)
    k2_arr = k_arr ** 2
    
    r_linear, p_linear = pearsonr(k_arr, area_means_cw)
    r_quad, p_quad = pearsonr(k2_arr, area_means_cw)
    
    print(f"\n  gap vs k (diffusion):  r={r_linear:.4f}  p={p_linear:.4e}")
    print(f"  gap vs k² (holonomy):  r={r_quad:.4f}  p={p_quad:.4e}")
    
    # Line control: should scale as √k (diffusion) or k, NOT k²
    r_line_k, _ = pearsonr(k_arr, area_means_line)
    r_line_k2, _ = pearsonr(k2_arr, area_means_line)
    print(f"  LINE gap vs k:  r={r_line_k:.4f}")
    print(f"  LINE gap vs k²: r={r_line_k2:.4f}")
    
    results["area_scaling"] = {
        "k_values": area_k,
        "cw_means": area_means_cw,
        "line_means": area_means_line,
        "gap_vs_k_pearson": float(r_linear),
        "gap_vs_k2_pearson": float(r_quad),
        "line_vs_k_pearson": float(r_line_k),
        "line_vs_k2_pearson": float(r_line_k2),
    }
    
    # --- Test 3: Dissipation ---
    print(f"\n--- Test 3: Loss trajectory asymmetry ---", flush=True)
    # In a flat space, +A then -A should return loss to starting value
    # In a curved space, after the full rectangle, loss should differ
    
    loss_start = []
    loss_end_cw = []
    loss_end_ccw = []
    for trial in range(100):
        ds = int(rng.integers(0, 100000))
        net_test = TinyNet(d_in, d_hidden, d_out)
        set_params(net_test, theta_init)
        X_A, Y_A = make_concept_data(0, seed=ds, d_in=d_in, d_out=d_out)
        loss_fn = nn.MSELoss()
        l0 = loss_fn(net_test(X_A), Y_A).item()
        loss_start.append(l0)
        
        _, _, losses_cw = run_rectangular_loop(theta_init, 'CW', k_steps, lr, ds, d_in, d_hidden, d_out)
        net_cw = TinyNet(d_in, d_hidden, d_out)
        cw_mag, cw_vec, _ = run_rectangular_loop(theta_init, 'CW', k_steps, lr, ds, d_in, d_hidden, d_out)
        set_params(net_cw, theta_init + cw_vec)
        l_cw = loss_fn(net_cw(X_A), Y_A).item()
        loss_end_cw.append(l_cw)
        
        ccw_mag, ccw_vec, _ = run_rectangular_loop(theta_init, 'CCW', k_steps, lr, ds, d_in, d_hidden, d_out)
        net_ccw = TinyNet(d_in, d_hidden, d_out)
        set_params(net_ccw, theta_init + ccw_vec)
        l_ccw = loss_fn(net_ccw(X_A), Y_A).item()
        loss_end_ccw.append(l_ccw)
    
    loss_start = np.array(loss_start)
    loss_end_cw = np.array(loss_end_cw)
    loss_end_ccw = np.array(loss_end_ccw)
    
    print(f"  Loss on concept A:")
    print(f"    Start:     mean={np.mean(loss_start):.6f}")
    print(f"    After CW:  mean={np.mean(loss_end_cw):.6f}")
    print(f"    After CCW: mean={np.mean(loss_end_ccw):.6f}")
    print(f"    CW shift:  {np.mean(loss_end_cw - loss_start):.6f}")
    print(f"    CCW shift: {np.mean(loss_end_ccw - loss_start):.6f}")
    
    # If curvature: CW and CCW should shift the loss in DIFFERENT directions
    t_loss, p_loss = ttest_ind(loss_end_cw, loss_end_ccw, equal_var=False)
    print(f"    CW loss vs CCW loss: t={t_loss:.3f}, p={p_loss:.4e}")
    
    results["loss_asymmetry"] = {
        "loss_start_mean": float(np.mean(loss_start)),
        "loss_cw_mean": float(np.mean(loss_end_cw)),
        "loss_ccw_mean": float(np.mean(loss_end_ccw)),
        "cw_ccw_loss_p": float(p_loss),
    }
    
    # --- VERDICT ---
    print(f"\n{'='*70}", flush=True)
    ev_for, ev_against = [], []
    
    # 1. Orientation: cosine significantly negative?
    if p_cos < 0.05 and np.mean(cosines) < -0.1:
        ev_for.append(f"CW/CCW holonomy vectors anti-correlated (cos={np.mean(cosines):.3f}, p={p_cos:.4e})")
    elif p_cos < 0.05 and np.mean(cosines) > 0.1:
        ev_against.append(f"CW/CCW vectors positively correlated — diffusion, not curvature (cos={np.mean(cosines):.3f})")
    else:
        ev_against.append(f"CW/CCW vectors not significantly oriented (cos={np.mean(cosines):.3f}, p={p_cos:.4e})")
    
    # 2. Rectangular gap > line gap?
    if p_line < 0.05 and np.mean(cw_mags) > np.mean(line_mags):
        ev_for.append(f"rectangle gap > line gap (CW={np.mean(cw_mags):.4f} vs LINE={np.mean(line_mags):.4f}, p={p_line:.4e})")
    else:
        ev_against.append(f"rectangle gap ≈ line gap (p={p_line:.4e})")
    
    # 3. Area scaling: gap ~ k² better than gap ~ k?
    if r_quad > r_linear + 0.01:
        ev_for.append(f"gap scales as k² (holonomy) better than k (diffusion): r_quad={r_quad:.4f} vs r_lin={r_linear:.4f}")
    elif r_linear > r_quad + 0.01:
        ev_against.append(f"gap scales as k (diffusion) better than k² (holonomy): r_lin={r_linear:.4f} vs r_quad={r_quad:.4f}")
    else:
        ev_against.append(f"gap scaling ambiguous between k and k²: r_lin={r_linear:.4f} vs r_quad={r_quad:.4f}")
    
    # 4. Loss asymmetry
    if p_loss < 0.05:
        ev_for.append(f"CW and CCW produce different loss profiles (p={p_loss:.4e})")
    else:
        ev_against.append(f"CW and CCW produce same loss (p={p_loss:.4e})")
    
    if len(ev_for) >= 3:
        verdict = "TRAINING HOLONOMY CONFIRMED — curvature in gradient-space"
    elif len(ev_for) >= 2:
        verdict = "STRONG EVIDENCE — multiple signatures of curvature"
    elif len(ev_for) >= 1:
        verdict = "WEAK SIGNAL — partial evidence, needs refinement"
    else:
        verdict = "NULL — no curvature detected in this configuration"
    
    print(f"  VERDICT: {verdict}")
    print(f"{'='*70}", flush=True)
    for e in ev_for:   print(f"  ✓ {e}")
    for a in ev_against: print(f"  ✗ {a}")
    
    results["verdict"] = verdict
    results["evidence_for"] = ev_for
    results["evidence_against"] = ev_against
    
    out = RESULT_DIR / f"training_holonomy_v2_{TIMESTAMP}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out}")
    return results


if __name__ == "__main__":
    run_experiment()
