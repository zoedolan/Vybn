#!/usr/bin/env python3
"""
Experiment E.1 — The Quantum Mirror (Simulation)

Compares a baseline VQC (vanilla gradient descent) against a geometrically-
regularized VQC (diagonal quantum natural gradient) on a 2-bit AND gate task.

Both use a 4-qubit, 4-layer hardware-efficient ansatz (32 parameters).

**Geometric regularization — diagonal quantum natural gradient (QNG):**
For each parameter i, we compute the diagonal FS metric element
g_ii = (1/2)(1 - Re⟨ψ(θ+se_i)|ψ(θ-se_i)⟩) from the same parameter-shifted
circuits used for the CE gradient (Stokes et al., Quantum 4, 269, 2020).
This adds ZERO extra circuit evaluations.  The update becomes:

    θ ← θ − η · diag(g + εI)⁻¹ · ∇CE

This moves the optimizer in Fubini-Study distance units, taking larger steps
in flat directions and smaller steps in curved directions.

Prediction (from Qi et al. 2026): the geometry-aware optimizer should reach
the same accuracy in fewer steps by navigating geodesics, and the learned
solution should have different DQFIM effective dimension — reflecting a
different universality class of the learned circuit.

Output: results/experiment_E1_simulation_result.json
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector

os.environ["PYTHONUNBUFFERED"] = "1"

# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_QUBITS = 4
N_LAYERS = 4
N_PARAMS = N_QUBITS * N_LAYERS * 2  # 32
N_STEPS = 200
LR = 0.05
LR_QNG = 0.02             # QNG amplifies steps; use moderately smaller LR
DIAG_DAMPING = 0.001      # small damping — let the metric do the work
SNAPSHOT_EVERY = 10
DQFIM_N_SAMPLES = 15
DQFIM_EPSILON = 0.05
BERRY_EPSILON = 0.08
PARAM_SHIFT = np.pi / 2

# ---------------------------------------------------------------------------
# Dataset: 2-bit AND gate  (4 examples)
# ---------------------------------------------------------------------------
X_BITS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y_LABELS = np.array([0, 0, 0, 1], dtype=float)
N_DATA = len(Y_LABELS)


def encode_input(bits: np.ndarray) -> np.ndarray:
    """Map 2-bit input → 4 rotation angles (bits duplicated)."""
    return np.array([bits[0], bits[1], bits[0], bits[1]]) * np.pi


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------
def build_ansatz() -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    inp = ParameterVector("x", N_QUBITS)
    theta = ParameterVector("θ", N_PARAMS)
    qc = QuantumCircuit(N_QUBITS)

    for i in range(N_QUBITS):
        qc.ry(inp[i], i)

    idx = 0
    for _ in range(N_LAYERS):
        for q in range(N_QUBITS):
            qc.ry(theta[idx], q); idx += 1
        for q in range(N_QUBITS):
            qc.rz(theta[idx], q); idx += 1
        for q in range(N_QUBITS):
            qc.cx(q, (q + 1) % N_QUBITS)

    return qc, inp, theta


_QC, _INP, _THETA = build_ansatz()


# ---------------------------------------------------------------------------
# Statevector helpers
# ---------------------------------------------------------------------------
def get_statevector(input_angles: np.ndarray, params: np.ndarray) -> np.ndarray:
    bind = {_INP[i]: float(input_angles[i]) for i in range(N_QUBITS)}
    bind.update({_THETA[i]: float(params[i]) for i in range(N_PARAMS)})
    return Statevector.from_instruction(_QC.assign_parameters(bind)).data


def proba_from_sv(sv: np.ndarray) -> float:
    """P(qubit-0 in |1⟩) from a statevector."""
    probs = np.abs(sv) ** 2
    return float(sum(probs[k] for k in range(len(probs)) if k & 1))


def predict_proba(input_angles: np.ndarray, params: np.ndarray) -> float:
    return proba_from_sv(get_statevector(input_angles, params))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def ce_from_probs(probs_list: list[float]) -> float:
    eps = 1e-8
    total = 0.0
    for p, y in zip(probs_list, Y_LABELS):
        p = np.clip(p, eps, 1 - eps)
        total += -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return total / N_DATA


def cross_entropy(params: np.ndarray) -> float:
    return ce_from_probs([predict_proba(encode_input(b), params) for b in X_BITS])


def accuracy(params: np.ndarray) -> float:
    return sum(
        1.0 for b, y in zip(X_BITS, Y_LABELS)
        if (predict_proba(encode_input(b), params) >= 0.5) == (y >= 0.5)
    ) / N_DATA


# ---------------------------------------------------------------------------
# Gradient (vanilla)
# ---------------------------------------------------------------------------
def vanilla_gradient(params: np.ndarray) -> tuple[np.ndarray, float]:
    """CE gradient via parameter-shift rule.  Cost: 2 × N_PARAMS × N_DATA svs."""
    loss_val = cross_entropy(params)
    grad = np.zeros(N_PARAMS)
    for j in range(N_PARAMS):
        p_plus = params.copy();  p_plus[j] += PARAM_SHIFT
        p_minus = params.copy(); p_minus[j] -= PARAM_SHIFT
        grad[j] = (cross_entropy(p_plus) - cross_entropy(p_minus)) / 2.0
    return grad, loss_val


# ---------------------------------------------------------------------------
# Gradient + diagonal FS metric  (zero extra cost)
# ---------------------------------------------------------------------------
def gradient_and_metric(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute CE gradient AND diagonal FS metric from the same shifted circuits.
    For parameter-shift-compatible gates (RY, RZ):
        g_ii = (1/2)(1 - Re⟨ψ(θ+se_i)|ψ(θ-se_i)⟩)  averaged over data
    (Stokes et al., Quantum 4, 269, 2020).
    """
    ref_probs = [predict_proba(encode_input(b), params) for b in X_BITS]
    loss_val = ce_from_probs(ref_probs)

    grad = np.zeros(N_PARAMS)
    diag_g = np.zeros(N_PARAMS)
    eps_num = 1e-8

    for j in range(N_PARAMS):
        p_plus = params.copy();  p_plus[j] += PARAM_SHIFT
        p_minus = params.copy(); p_minus[j] -= PARAM_SHIFT

        ce_plus = 0.0
        ce_minus = 0.0
        g_jj = 0.0

        for i, bits in enumerate(X_BITS):
            inp = encode_input(bits)
            sv_p = get_statevector(inp, p_plus)
            sv_m = get_statevector(inp, p_minus)

            # CE contributions from shifted circuits
            pp = np.clip(proba_from_sv(sv_p), eps_num, 1 - eps_num)
            pm = np.clip(proba_from_sv(sv_m), eps_num, 1 - eps_num)
            y = Y_LABELS[i]
            ce_plus += -(y * np.log(pp) + (1 - y) * np.log(1 - pp))
            ce_minus += -(y * np.log(pm) + (1 - y) * np.log(1 - pm))

            # Diagonal FS metric from the same two statevectors
            g_jj += 0.5 * (1.0 - float(np.real(np.vdot(sv_p, sv_m))))

        grad[j] = (ce_plus - ce_minus) / (2.0 * N_DATA)
        diag_g[j] = g_jj / N_DATA

    return grad, diag_g, loss_val


# ---------------------------------------------------------------------------
# DQFIM effective dimension  (Haug & Kim, PRL 2024)
# d_eff = (Tr F)² / Tr(F²)
# ---------------------------------------------------------------------------
def compute_dqfim_eff_dim(params: np.ndarray) -> float:
    psi_list = [get_statevector(encode_input(b), params) for b in X_BITS]
    rng = np.random.RandomState(SEED + int(abs(params[0]) * 1000) % 10000)
    traces = []

    for _ in range(DQFIM_N_SAMPLES):
        v = rng.randn(N_PARAMS)
        v /= np.linalg.norm(v)
        p_plus = params + DQFIM_EPSILON * v
        p_minus = params - DQFIM_EPSILON * v

        vtFv = 0.0
        for idx, bits in enumerate(X_BITS):
            inp = encode_input(bits)
            psi_0 = psi_list[idx]
            psi_p = get_statevector(inp, p_plus)
            psi_m = get_statevector(inp, p_minus)
            dpsi = (psi_p - psi_m) / (2 * DQFIM_EPSILON)
            vtFv += float(np.real(
                np.vdot(dpsi, dpsi) - np.abs(np.vdot(dpsi, psi_0)) ** 2
            ))
        traces.append(vtFv / N_DATA)

    d = N_PARAMS
    mean_vtFv = np.mean(traces)
    mean_vtFv_sq = np.mean(np.array(traces) ** 2)
    tr_F = d * mean_vtFv
    tr_F2 = (d * (d + 2) * mean_vtFv_sq - tr_F ** 2) / 2.0

    if tr_F2 < 1e-12:
        return float(d)
    return float(np.clip(tr_F ** 2 / tr_F2, 1.0, d))


# ---------------------------------------------------------------------------
# Berry phase (Bargmann invariant)
# ---------------------------------------------------------------------------
def compute_berry_phase(params: np.ndarray) -> float:
    rng = np.random.RandomState(SEED + int(abs(params.sum()) * 100) % 10000)
    v1 = rng.randn(N_PARAMS) * BERRY_EPSILON
    v2 = rng.randn(N_PARAMS) * BERRY_EPSILON
    p1, p2, p3 = params + v1, params + v2, params

    total = 0.0
    for bits in X_BITS:
        inp = encode_input(bits)
        psi1 = get_statevector(inp, p1)
        psi2 = get_statevector(inp, p2)
        psi3 = get_statevector(inp, p3)
        bargmann = np.vdot(psi1, psi2) * np.vdot(psi2, psi3) * np.vdot(psi3, psi1)
        total += np.angle(bargmann)
    return float(total / N_DATA)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_run(name: str, use_qng: bool) -> list[dict]:
    rng_init = np.random.RandomState(SEED)
    params = rng_init.randn(N_PARAMS) * 0.1
    lr = LR_QNG if use_qng else LR
    snapshots = []

    for step in range(N_STEPS + 1):
        if step % SNAPSHOT_EVERY == 0:
            ce = cross_entropy(params)
            acc = accuracy(params)
            eff_dim = compute_dqfim_eff_dim(params)
            berry = compute_berry_phase(params)
            snap = {
                "step": step,
                "ce": round(ce, 6),
                "acc": round(acc, 4),
                "eff_dim": round(eff_dim, 4),
                "berry": round(berry, 6),
            }
            snapshots.append(snap)
            tag = "QNG" if use_qng else "VAN"
            print(
                f"  [{tag}] step={step:>3d}  CE={ce:.4f}  "
                f"acc={acc:.2f}  d_eff={eff_dim:.1f}  berry={berry:.4f}",
                flush=True,
            )

        if step == N_STEPS:
            break

        if use_qng:
            grad, diag_g, _ = gradient_and_metric(params)
            # Diagonal natural gradient — no renormalization!
            # g_ii⁻¹ reshapes the gradient to follow geodesics
            nat_grad = grad / (diag_g + DIAG_DAMPING)
            params = params - lr * nat_grad
        else:
            grad, _ = vanilla_gradient(params)
            params = params - lr * grad

    return snapshots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("Experiment E.1 — The Quantum Mirror (Simulation)")
    print("=" * 60)
    print(f"Config: {N_QUBITS}q × {N_LAYERS}L = {N_PARAMS} params")
    print(f"Task: 2-bit AND gate (4 examples)")
    print(f"Steps: {N_STEPS}  LR(vanilla)={LR}  LR(QNG)={LR_QNG}")
    print(f"Baseline: vanilla gradient descent")
    print(f"Geometric: diagonal quantum natural gradient (Stokes et al. 2020)")
    print()

    print("-" * 40)
    print("Baseline (vanilla GD)")
    print("-" * 40)
    bl_snaps = train_run("baseline", use_qng=False)
    t_bl = time.time() - t0
    print(f"\nBaseline done in {t_bl:.1f}s\n")

    print("-" * 40)
    print("Geometric (diagonal QNG)")
    print("-" * 40)
    geo_snaps = train_run("geometric", use_qng=True)
    t_total = time.time() - t0
    print(f"\nGeometric done in {t_total - t_bl:.1f}s")
    print(f"Total: {t_total:.1f}s\n")

    # --- analysis ---
    bl_f, geo_f = bl_snaps[-1], geo_snaps[-1]
    bl_eff = [s["eff_dim"] for s in bl_snaps]
    geo_eff = [s["eff_dim"] for s in geo_snaps]
    mean_ratio = float(np.mean(geo_eff)) / max(float(np.mean(bl_eff)), 1e-8)

    bl_full = next((s["step"] for s in bl_snaps if s["acc"] >= 1.0), N_STEPS)
    geo_full = next((s["step"] for s in geo_snaps if s["acc"] >= 1.0), N_STEPS)

    # Track eff_dim divergence over training
    eff_dim_diffs = [g - b for g, b in zip(geo_eff, bl_eff)]
    early_diff = float(np.mean(eff_dim_diffs[:5]))   # steps 0-40
    late_diff = float(np.mean(eff_dim_diffs[-5:]))    # steps 160-200

    parts = []
    if geo_full < bl_full:
        parts.append(f"QNG converged to 100% in {geo_full} steps vs baseline's {bl_full}")
    elif geo_full == bl_full:
        parts.append(f"Both reached 100% at step {bl_full}")
    else:
        parts.append(f"Baseline reached 100% at step {bl_full}, QNG at {geo_full}")

    if abs(mean_ratio - 1.0) > 0.03:
        parts.append(f"mean d_eff ratio (QNG/VAN) = {mean_ratio:.3f}")
    if abs(late_diff) > 1.0:
        parts.append(f"late-training d_eff difference = {late_diff:+.1f}")

    if geo_full < bl_full and mean_ratio >= 0.95:
        parts.append(
            "— QNG navigates the loss landscape more efficiently via FS geodesics, "
            "consistent with the geometric coherence hypothesis"
        )
    elif mean_ratio > 1.1:
        parts.append(
            "— QNG maintains higher effective dimension, suggesting it avoids "
            "representational collapse"
        )
    elif abs(mean_ratio - 1.0) <= 0.03 and abs(geo_full - bl_full) <= 10:
        parts.append(
            "— minimal geometric effect at this scale. The task may be too "
            "simple or the circuit too expressive for collapse to occur"
        )

    verdict = ". ".join(parts) + "."

    result = {
        "experiment": "E.1",
        "description": "Quantum Mirror — Simulation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(t_total, 1),
        "config": {
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
            "n_params": N_PARAMS,
            "n_steps": N_STEPS,
            "lr_vanilla": LR,
            "lr_qng": LR_QNG,
            "diag_damping": DIAG_DAMPING,
            "task": "2-bit AND gate",
            "seed": SEED,
            "snapshot_every": SNAPSHOT_EVERY,
            "dqfim_n_samples": DQFIM_N_SAMPLES,
            "dqfim_epsilon": DQFIM_EPSILON,
            "berry_epsilon": BERRY_EPSILON,
        },
        "baseline": {"snapshots": bl_snaps},
        "geometric": {"snapshots": geo_snaps},
        "comparison": {
            "baseline_final_acc": bl_f["acc"],
            "geometric_final_acc": geo_f["acc"],
            "baseline_final_ce": bl_f["ce"],
            "geometric_final_ce": geo_f["ce"],
            "baseline_steps_to_full_acc": bl_full,
            "geometric_steps_to_full_acc": geo_full,
            "baseline_mean_eff_dim": round(float(np.mean(bl_eff)), 4),
            "geometric_mean_eff_dim": round(float(np.mean(geo_eff)), 4),
            "mean_eff_dim_ratio": round(float(mean_ratio), 4),
            "early_eff_dim_diff": round(early_diff, 4),
            "late_eff_dim_diff": round(late_diff, 4),
            "verdict": verdict,
        },
    }

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment_E1_simulation_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline:  acc={bl_f['acc']:.2f}  CE={bl_f['ce']:.4f}  d_eff={bl_f['eff_dim']:.1f}  (100% at step {bl_full})")
    print(f"Geometric: acc={geo_f['acc']:.2f}  CE={geo_f['ce']:.4f}  d_eff={geo_f['eff_dim']:.1f}  (100% at step {geo_full})")
    print(f"Mean d_eff ratio (QNG/VAN): {mean_ratio:.4f}")
    print(f"Early d_eff diff: {early_diff:+.2f}  Late: {late_diff:+.2f}")
    print(f"\n{verdict}")
    print(f"\nSaved: {out_path}")
    print(f"Time:  {t_total:.1f}s")


if __name__ == "__main__":
    main()
