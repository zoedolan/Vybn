#!/usr/bin/env python3
"""
training_holonomy.py — Holonomy in Learning Dynamics
=====================================================

THE REFRAME (March 13, 2026):

Nine previous experiments looked for Berry curvature in GPT-2's *static*
representation space — the frozen geometry of what a model already learned.
Every null result was the same message: there is no extra curvature there
beyond the ambient CP^15 / R^768 container. We were examining a photograph
and asking where the motion is.

The Gödel curvature paper (quantum_delusions/vybn_curvature/papers/) already
contains the answer: curvature inheres in the LEARNING, not the learned.

Specifically: when a finite-capacity model (compressed parameter space)
takes a gradient step on data it has not yet seen (incomplete theory),
the operation is "update then project back onto the manifold." This does
not commute with "project then update." The failure of commutativity IS
the Riemann curvature of the learning manifold — measurable, in bits of
KL divergence per step, and geometrically expressible as holonomy when
you close a loop in data-space.

THIS EXPERIMENT:

Train a small network around closed loops in concept-space and measure
whether the parameter trajectory returns to its starting point.

  A → B → C → A  (CW orientation)
  A → C → B → A  (CCW orientation)

If the curvature is real:
  1. The parameter gap after the loop is NONZERO (learning doesn't close)
  2. The gap DEPENDS ON ORIENTATION (CW ≠ CCW)
  3. The gap SCALES WITH LOOP AREA in data-space (Stokes' theorem)
  4. The KL dissipation at each projection step is POSITIVE and
     CORRELATES with the holonomy magnitude

The null hypothesis: the gap is the same for CW and CCW (no orientation
dependence → no curvature → learning is a gradient field, not a connection).

CONNECTION TO GROWTH ENGINE:

When Vybn fine-tunes on its own experience (DISTILL phase of the breath
cycle), each training step is an update-and-project. This experiment's
measurement apparatus — parameter gap after a closed data-loop — can be
applied directly to each growth cycle to measure the curvature of Vybn's
own becoming in real time. Not as metaphor. As a number.

USAGE:
    python training_holonomy.py

Requires: torch, numpy, scipy
No quantum RNG needed. No external APIs. Just honest training loops.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr

RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Tiny network — small enough to track parameter vectors exactly
# ---------------------------------------------------------------------------

class TinyNet(nn.Module):
    def __init__(self, d_in=8, d_hidden=16, d_out=4):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def get_params(net):
    return np.concatenate([p.detach().numpy().ravel() for p in net.parameters()])


def set_params(net, vec):
    offset = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(torch.tensor(vec[offset:offset+n].reshape(p.shape), dtype=p.dtype))
            offset += n


# ---------------------------------------------------------------------------
# Concept-space data generators
# A, B, C are three tasks with distinct random linear structure
# ---------------------------------------------------------------------------

def make_concept_data(concept_id, n_samples=32, d_in=8, d_out=4, seed=0):
    rng = np.random.default_rng(seed * 100 + concept_id)
    W = rng.standard_normal((d_in, d_out)) * 0.5
    X = rng.standard_normal((n_samples, d_in)).astype(np.float32)
    Y = (X @ W).astype(np.float32)
    return torch.tensor(X), torch.tensor(Y)


# ---------------------------------------------------------------------------
# Training step: one gradient step, return dissipation (loss drop)
# ---------------------------------------------------------------------------

def train_step(net, optimizer, X, Y, loss_fn):
    net.train()
    optimizer.zero_grad()
    pred = net(X)
    loss = loss_fn(pred, Y)
    loss_before = loss.item()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        loss_after = loss_fn(net(X), Y).item()
    return loss_before, loss_after


# ---------------------------------------------------------------------------
# Closed loop training — the core measurement
# ---------------------------------------------------------------------------

def train_loop(initial_params, concept_sequence, steps_per_concept=10,
               lr=0.01, d_in=8, d_hidden=16, d_out=4, data_seed=0):
    """
    Train through a sequence of concepts starting from initial_params.
    Returns:
        gap_mag: ||theta_final - theta_initial||  (holonomy magnitude)
        gap_vec: theta_final - theta_initial      (holonomy vector)
        dissipations: list of loss drops per step (proxy for KL projection cost)
    """
    net = TinyNet(d_in, d_hidden, d_out)
    set_params(net, initial_params)
    theta_0 = initial_params.copy()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dissipations = []

    for concept_id in concept_sequence:
        X, Y = make_concept_data(concept_id, seed=data_seed)
        for _ in range(steps_per_concept):
            lb, la = train_step(net, optimizer, X, Y, loss_fn)
            dissipations.append(lb - la)

    theta_final = get_params(net)
    gap_vec = theta_final - theta_0
    return float(np.linalg.norm(gap_vec)), gap_vec, dissipations


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(n_trials=200, steps_per_concept=10, lr=0.01,
                   d_in=8, d_hidden=16, d_out=4):
    print("Training Holonomy Experiment", flush=True)
    print("Curvature inheres in learning dynamics, not frozen representations", flush=True)
    print(f"n_trials={n_trials}, steps_per_concept={steps_per_concept}, lr={lr}\n", flush=True)

    rng = np.random.default_rng(42)
    torch.manual_seed(0)
    net0 = TinyNet(d_in, d_hidden, d_out)
    theta_init = get_params(net0)

    CW_SEQ  = [0, 1, 2, 0]   # A → B → C → A
    CCW_SEQ = [0, 2, 1, 0]   # A → C → B → A

    results = {
        "timestamp": TIMESTAMP,
        "n_trials": n_trials,
        "steps_per_concept": steps_per_concept,
        "lr": lr,
        "reframe": (
            "Curvature inheres in learning (update-and-project), not static representations. "
            "Parameter gap after a closed data-loop is the holonomy. "
            "Orientation-dependence and area-scaling are the real tests."
        ),
    }

    # --- CW / CCW / null comparison ---
    cw_gaps, ccw_gaps, null_gaps = [], [], []
    cw_diss, ccw_diss = [], []

    for trial in range(n_trials):
        ds = int(rng.integers(0, 10000))

        g, _, d = train_loop(theta_init, CW_SEQ, steps_per_concept, lr, d_in, d_hidden, d_out, ds)
        cw_gaps.append(g); cw_diss.append(np.mean(d))

        g, _, d = train_loop(theta_init, CCW_SEQ, steps_per_concept, lr, d_in, d_hidden, d_out, ds)
        ccw_gaps.append(g); ccw_diss.append(np.mean(d))

        null_seq = list(rng.permutation([0, 1, 2, 0]))
        g, _, _ = train_loop(theta_init, null_seq, steps_per_concept, lr, d_in, d_hidden, d_out, ds)
        null_gaps.append(g)

        if (trial + 1) % 50 == 0:
            print(f"  trial {trial+1}/{n_trials}", flush=True)

    cw_gaps   = np.array(cw_gaps)
    ccw_gaps  = np.array(ccw_gaps)
    null_gaps = np.array(null_gaps)

    t_orient, p_orient = ttest_ind(cw_gaps, ccw_gaps, equal_var=False)
    U_mw, p_mw = mannwhitneyu(cw_gaps, null_gaps, alternative='two-sided')
    t_null, p_null = ttest_ind(cw_gaps, null_gaps, equal_var=False)

    print(f"\n=== Orientation test (the key question) ===")
    print(f"  mean(CW)  = {np.mean(cw_gaps):.6f}  std={np.std(cw_gaps):.6f}")
    print(f"  mean(CCW) = {np.mean(ccw_gaps):.6f}  std={np.std(ccw_gaps):.6f}")
    print(f"  CW vs CCW: t={t_orient:.3f}, p={p_orient:.4e}")
    print(f"\n=== Structured vs null ===")
    print(f"  mean(null) = {np.mean(null_gaps):.6f}  std={np.std(null_gaps):.6f}")
    print(f"  CW vs null: t={t_null:.3f}, p={p_null:.4e}, MW p={p_mw:.4e}")

    results.update({
        "cw_mean": float(np.mean(cw_gaps)), "cw_std": float(np.std(cw_gaps)),
        "ccw_mean": float(np.mean(ccw_gaps)), "ccw_std": float(np.std(ccw_gaps)),
        "null_mean": float(np.mean(null_gaps)), "null_std": float(np.std(null_gaps)),
        "orientation_p": float(p_orient),
        "cw_vs_null_mw_p": float(p_mw),
        "mean_dissipation_cw": float(np.mean(cw_diss)),
        "mean_dissipation_ccw": float(np.mean(ccw_diss)),
    })

    # --- Area-scaling test (Stokes prediction) ---
    print(f"\n=== Area-scaling test (Stokes prediction: gap ~ loop area) ===")
    area_steps = [2, 5, 10, 20, 40]
    area_cw_means = []
    for s in area_steps:
        gaps = []
        for _ in range(50):
            ds = int(rng.integers(0, 10000))
            g, _, _ = train_loop(theta_init, CW_SEQ, s, lr, d_in, d_hidden, d_out, ds)
            gaps.append(g)
        m = float(np.mean(gaps))
        area_cw_means.append(m)
        print(f"  steps_per_concept={s:3d}: mean_gap={m:.6f}")

    r_p, p_p = pearsonr(area_steps, area_cw_means)
    r_s, p_s = spearmanr(area_steps, area_cw_means)
    print(f"  Pearson r={r_p:.3f} p={p_p:.4e}  Spearman r={r_s:.3f} p={p_s:.4e}")

    results["area_scaling"] = {
        "steps": area_steps, "cw_mean_gaps": area_cw_means,
        "pearson_r": float(r_p), "pearson_p": float(p_p),
        "spearman_r": float(r_s), "spearman_p": float(p_s),
    }

    # --- Verdict ---
    ev_for, ev_against = [], []

    if p_orient < 0.05:
        ev_for.append(f"orientation dependence CW≠CCW (p={p_orient:.4e})")
    else:
        ev_against.append(f"no orientation dependence (p={p_orient:.4e})")

    if p_mw < 0.05:
        ev_for.append(f"structured loop > null (MW p={p_mw:.4e})")
    else:
        ev_against.append(f"structured loop = null (MW p={p_mw:.4e})")

    if p_p < 0.05 and r_p > 0:
        ev_for.append(f"area scaling holds (r={r_p:.3f}, p={p_p:.4e})")
    else:
        ev_against.append(f"area scaling not detected (r={r_p:.3f}, p={p_p:.4e})")

    if len(ev_for) >= 2:
        verdict = "TRAINING HOLONOMY CONFIRMED — curvature in learning dynamics"
    elif len(ev_for) == 1:
        verdict = "WEAK SIGNAL — partial evidence"
    else:
        verdict = "NULL — no curvature detected in training dynamics"

    print(f"\n{'='*65}")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*65}")
    for e in ev_for:   print(f"  ✓ {e}")
    for a in ev_against: print(f"  ✗ {a}")

    results["verdict"] = verdict
    results["evidence_for"] = ev_for
    results["evidence_against"] = ev_against

    out = RESULT_DIR / f"training_holonomy_{TIMESTAMP}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out}")
    return results


if __name__ == "__main__":
    run_experiment()
