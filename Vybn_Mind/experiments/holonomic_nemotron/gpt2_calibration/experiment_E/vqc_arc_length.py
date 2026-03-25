#!/usr/bin/env python3
"""
Experiment E.1 — The Quantum Mirror (Simulation)

Two 4-qubit, 4-layer VQCs trained on an 8-example classification task.
The baseline minimizes binary cross-entropy only.  The geometric run adds
Fisher-information preconditioning — dampening gradient updates along
directions of high quantum state-space curvature:

    g_i → g_i / (1 + λ · F_ii)

where F_ii is the i-th diagonal of the quantum Fisher information matrix.
This is the direct quantum analog of Experiment D's classical arc-length
regularizer: it penalizes large state-space movement per step, preserving
representational diversity throughout training.

Snapshots every 10 steps: CE loss, accuracy, DQFIM effective dimension
(Tr(F)² / Tr(F²), Haug & Kim PRL 2024), Berry phase (Bargmann invariant).

Output: results/experiment_E1_simulation_result.json
"""
import os
import sys
import json
import time
import datetime
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

SEED = 42

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_QUBITS = 4
N_LAYERS = 4
N_PARAMS = N_QUBITS * N_LAYERS * 2   # 32
N_STEPS = 100            # both runs converge well before 100
LR = 0.15
LAMBDA_GEO = 2.0
SHIFT = np.pi / 2
SNAPSHOT_EVERY = 10
FD_EPSILON = 1e-3
FISHER_UPDATE_EVERY = 20  # recompute Fisher for preconditioning

# ---------------------------------------------------------------------------
# Task: 8 examples, 3 input bits, label = XOR of first two bits
# Balanced (4:4), requires entanglement, learnable in ~30 steps.
# ---------------------------------------------------------------------------
X_DATA = np.array([
    [0, 0, 0], [0, 0, 1],  # label 0
    [0, 1, 0], [0, 1, 1],  # label 1
    [1, 0, 0], [1, 0, 1],  # label 1
    [1, 1, 0], [1, 1, 1],  # label 0
], dtype=float)
Y_DATA = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=float)
N_DATA = len(X_DATA)

SIM = AerSimulator(method="statevector")


# ===================================================================
# Circuit
# ===================================================================
def build_circuit(params, x):
    qc = QuantumCircuit(N_QUBITS)
    for i in range(min(len(x), N_QUBITS)):
        qc.rx(np.pi * x[i], i)
    idx = 0
    for _ in range(N_LAYERS):
        for q in range(N_QUBITS):
            qc.ry(params[idx], q); idx += 1
            qc.rz(params[idx], q); idx += 1
        for q in range(N_QUBITS - 1):
            qc.cx(q, q + 1)
    qc.save_statevector()
    return qc


def get_sv(params, x):
    qc = build_circuit(params, x)
    return np.asarray(SIM.run(qc, shots=0).result().get_statevector(qc).data)


def predict_proba(sv):
    """P(qubit 0 = |1⟩): odd-index amplitudes in little-endian."""
    return float(np.sum(np.abs(sv[1::2]) ** 2))


# ===================================================================
# Loss / accuracy
# ===================================================================
def bce(p, y, eps=1e-8):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def dataset_loss(params):
    return sum(bce(predict_proba(get_sv(params, x)), y)
               for x, y in zip(X_DATA, Y_DATA)) / N_DATA


def dataset_accuracy(params):
    return sum(
        (1.0 if predict_proba(get_sv(params, x)) >= 0.5 else 0.0) == y
        for x, y in zip(X_DATA, Y_DATA)
    ) / N_DATA


# ===================================================================
# CE gradient (parameter-shift)
# ===================================================================
def ce_gradient(params):
    g = np.zeros_like(params)
    for i in range(len(params)):
        pp = params.copy(); pp[i] += SHIFT
        pm = params.copy(); pm[i] -= SHIFT
        g[i] = (dataset_loss(pp) - dataset_loss(pm)) / 2.0
    return g


# ===================================================================
# Fisher diagonal
# F_ii = (4/|S|) Σ_{x∈S} [ ⟨∂_i ψ|∂_i ψ⟩ - |⟨ψ|∂_i ψ⟩|² ]
# ===================================================================
def compute_fisher_diagonal(params, data_indices=None):
    """Diagonal of quantum Fisher info matrix over given data subset."""
    indices = data_indices if data_indices is not None else range(N_DATA)
    n = len(params)
    diag = np.zeros(n)
    count = 0
    for di in indices:
        x = X_DATA[di]
        sv = get_sv(params, x)
        for i in range(n):
            pe = params.copy(); pe[i] += FD_EPSILON
            sve = get_sv(pe, x)
            d = (sve - sv) / FD_EPSILON
            diag[i] += 4.0 * (np.real(np.vdot(d, d)) - np.abs(np.vdot(sv, d)) ** 2)
        count += 1
    diag /= count
    return np.maximum(diag, 0.0)


def eff_dim_from_diag(diag):
    """Tr(F)² / Tr(F²) — effective dimension."""
    tr = np.sum(diag)
    tr2 = np.sum(diag ** 2)
    return float(tr ** 2 / tr2) if tr2 > 1e-15 else 0.0


# ===================================================================
# Berry phase (Bargmann invariant)
# ===================================================================
def compute_berry_phase(params, rng_seed):
    rng = np.random.RandomState(rng_seed)
    d1 = rng.randn(len(params)) * 0.05
    d2 = rng.randn(len(params)) * 0.05
    # 2-point subsample for speed
    phases = []
    for di in [0, 4]:
        x = X_DATA[di]
        sv1 = get_sv(params, x)
        sv2 = get_sv(params + d1, x)
        sv3 = get_sv(params + d2, x)
        phases.append(np.angle(
            np.vdot(sv1, sv2) * np.vdot(sv2, sv3) * np.vdot(sv3, sv1)
        ))
    return float(np.mean(phases))


# ===================================================================
# Train
# ===================================================================
def train(use_geometric, label):
    params = np.random.RandomState(SEED).uniform(-np.pi, np.pi, size=N_PARAMS)
    fisher_diag = np.ones(N_PARAMS)
    snapshots = []
    precond_idx = [0, 4]  # cheap subsample for preconditioning

    print(f"\n{'='*60}")
    print(f"  {label}  (lambda={LAMBDA_GEO if use_geometric else 0})")
    print(f"{'='*60}")

    for step in range(N_STEPS + 1):
        # --- Snapshot ---
        if step % SNAPSHOT_EVERY == 0:
            # Full-data Fisher for DQFIM measurement
            fisher_full = compute_fisher_diagonal(params)
            eff_dim = eff_dim_from_diag(fisher_full)
            if use_geometric:
                fisher_diag = fisher_full

            ce = dataset_loss(params)
            acc = dataset_accuracy(params)
            berry = compute_berry_phase(params, SEED + step)
            snap = {
                "step": step,
                "ce": round(float(ce), 6),
                "acc": round(float(acc), 4),
                "eff_dim": round(float(eff_dim), 4),
                "berry": round(float(berry), 6),
            }
            snapshots.append(snap)
            print(f"  step {step:>3d}  CE={ce:.4f}  acc={acc:.2f}  "
                  f"dim={eff_dim:.2f}  berry={berry:.4f}")
            sys.stdout.flush()

        elif use_geometric and step % FISHER_UPDATE_EVERY == 0:
            fisher_diag = compute_fisher_diagonal(params, data_indices=precond_idx)

        if step == N_STEPS:
            break

        # --- Gradient step ---
        grad = ce_gradient(params)

        if use_geometric:
            preconditioner = 1.0 / (1.0 + LAMBDA_GEO * fisher_diag)
            params = params - LR * preconditioner * grad
        else:
            params = params - LR * grad

    return {"snapshots": snapshots}


# ===================================================================
# Main
# ===================================================================
def main():
    t0 = time.time()
    print("Experiment E.1 — The Quantum Mirror (Simulation)")
    print(f"Task: 3-bit XOR(b0,b1), 8 examples  |  {N_QUBITS}q {N_LAYERS}L {N_PARAMS}p")
    print(f"Steps: {N_STEPS}  LR: {LR}  lambda: {LAMBDA_GEO}")

    baseline = train(use_geometric=False, label="Baseline")
    geometric = train(use_geometric=True, label="Geometric (Fisher-preconditioned)")

    bf = baseline["snapshots"][-1]
    gf = geometric["snapshots"][-1]
    bd = [s["eff_dim"] for s in baseline["snapshots"]]
    gd = [s["eff_dim"] for s in geometric["snapshots"]]
    mb, mg = float(np.mean(bd)), float(np.mean(gd))
    ratio = mg / mb if mb > 1e-8 else float("inf")

    if gf["acc"] >= bf["acc"] - 0.05 and ratio > 1.005:
        verdict = (
            "Geometric run maintains higher DQFIM effective dimension "
            "while matching baseline accuracy — consistent with geometric "
            "coherence preventing representational collapse."
        )
    elif gf["acc"] < bf["acc"] - 0.1:
        verdict = (
            "Geometric penalty impaired learning. lambda too strong or "
            "preconditioning distorted the optimization landscape."
        )
    else:
        verdict = (
            "No clear separation in effective dimension. The geometric "
            "effect is not significant at this task/scale."
        )

    elapsed = time.time() - t0
    result = {
        "experiment": "E.1",
        "description": "Quantum Mirror — Simulation",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
            "n_params": N_PARAMS,
            "n_steps": N_STEPS,
            "lr": LR,
            "lambda_geo": LAMBDA_GEO,
            "task": "3-bit input, XOR(b0,b1) label, 8 examples",
            "seed": SEED,
            "fisher_update_every": FISHER_UPDATE_EVERY,
        },
        "baseline": {"snapshots": baseline["snapshots"]},
        "geometric": {"snapshots": geometric["snapshots"]},
        "comparison": {
            "baseline_final_acc": bf["acc"],
            "geometric_final_acc": gf["acc"],
            "baseline_final_ce": bf["ce"],
            "geometric_final_ce": gf["ce"],
            "baseline_mean_eff_dim": round(mb, 4),
            "geometric_mean_eff_dim": round(mg, 4),
            "mean_eff_dim_ratio": round(float(ratio), 4),
            "verdict": verdict,
        },
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "experiment_E1_simulation_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Baseline:  acc={bf['acc']:.2f}  CE={bf['ce']:.4f}  dim={bf['eff_dim']:.2f}")
    print(f"  Geometric: acc={gf['acc']:.2f}  CE={gf['ce']:.4f}  dim={gf['eff_dim']:.2f}")
    print(f"  Mean eff_dim ratio: {ratio:.4f}")
    print(f"  Verdict: {verdict}")
    print(f"  Saved: {out_path}")
    print(f"{'='*60}")
    return result


if __name__ == "__main__":
    main()
