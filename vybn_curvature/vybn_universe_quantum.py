#!/usr/bin/env python3
"""
vybn_universe_quantum.py

Vybn rectangular universe on hardware: patch-wise curvature visibility.

This script probes the same rectangular loops as the Vybn-2 classical universe,
but across many disjoint 3-qubit "patches" of a real IBM device. For each patch
it extracts:

    boundary orientation slope  b  in  <Y0> ≈ a + b * A
    bulk swelling slope         α  in  S_bulk(|A|) ≈ S0 + α * |A|

where A ∈ {−2, −1, +1, +2} is the loop area in the (t1, t2) time plane.

Design:
    - Build seven circuits in one job:
          id, zero_short, zero_long, plus1, minus1, plus2, minus2.
    - Tile the device with disjoint 3-qubit patches: (3p,3p+1,3p+2).
    - Implement the same H1/H2 dynamics as in the classical model via native gates.
    - Rotate q0 and q2 into the Y basis; measure all qubits in Z.
    - From counts, estimate per-patch Ey0, Ez1 and turn Ez1 into bulk entropy.
    - Fit b and α per patch and report |b|/stderr(b) as a visibility score.

Outputs per run:
    - console table of per-patch b, α, visibilities
    - curvature_summary.json (all patch data)
    - PNGs in a run directory:
        boundary_slopes.png
        boundary_visibility.png
        bulk_alphas.png
        manifest.txt
"""

import argparse
import json
import math
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# ----------------------------------------------------------------------
# Patch-level gate builders (mirror H1/H2)
# ----------------------------------------------------------------------

def apply_u1_on_patch(qc: QuantumCircuit, theta: float, q0: int, q1: int) -> None:
    """U1(θ) = exp(-i θ X_q0 X_q1 / 2) via (H⊗H) ZZ (H⊗H)."""
    qc.h(q0)
    qc.h(q1)
    qc.cx(q0, q1)
    qc.rz(theta, q1)
    qc.cx(q0, q1)
    qc.h(q0)
    qc.h(q1)


def apply_u2_on_patch(qc: QuantumCircuit, theta: float, q1: int, q2: int) -> None:
    """U2(θ) = exp(-i θ Y_q1 X_q2 / 2) via basis map Rx(π/2) ⊗ H → ZZ → back."""
    qc.rx(math.pi / 2, q1)
    qc.h(q2)
    qc.cx(q1, q2)
    qc.rz(theta, q2)
    qc.cx(q1, q2)
    qc.rx(-math.pi / 2, q1)
    qc.h(q2)


def apply_loop_on_patch(qc: QuantumCircuit, steps, theta: float, p: int) -> None:
    """Apply a loop given as a list of (dt1, dt2) steps on patch p."""
    q0 = 3 * p
    q1 = q0 + 1
    q2 = q0 + 2
    for dt1, dt2 in steps:
        if dt1 == 1 and dt2 == 0:
            apply_u1_on_patch(qc, +theta, q0, q1)
        elif dt1 == -1 and dt2 == 0:
            apply_u1_on_patch(qc, -theta, q0, q1)
        elif dt1 == 0 and dt2 == 1:
            apply_u2_on_patch(qc, +theta, q1, q2)
        elif dt1 == 0 and dt2 == -1:
            apply_u2_on_patch(qc, -theta, q1, q2)
        else:
            raise ValueError(f"Non-unit step: {(dt1, dt2)}")


def rotate_patches_to_measure(qc: QuantumCircuit, num_patches: int) -> None:
    """Rotate q0 and q2 of each patch into the Y basis (S†H); leave q1 in Z."""
    for p in range(num_patches):
        q0 = 3 * p
        q2 = q0 + 2
        qc.sdg(q0)
        qc.h(q0)
        qc.sdg(q2)
        qc.h(q2)


# ----------------------------------------------------------------------
# Counts handling and simple stats
# ----------------------------------------------------------------------

def counts_dict(pub) -> dict:
    """Robustly get a counts dict from a SamplerV2 primitive result."""
    data = getattr(pub, "data", pub)
    m = getattr(data, "m", None)
    if m is not None and hasattr(m, "get_counts"):
        return m.get_counts()
    meas = getattr(data, "meas", None)
    if meas is not None and hasattr(meas, "get_counts"):
        return meas.get_counts()
    raise AttributeError("Sampler result has neither .data.m nor .data.meas with get_counts()")


def counts_to_patch_expectations(counts: dict, num_patches: int):
    """
    Return per-patch (Ey0, Ez1, Ey2) from a multi-qubit bitstring count dict.

    Assumes one classical bit per qubit, with rightmost char = lsb (c[0]).
    """
    totals = [0] * num_patches
    n0_0 = [0] * num_patches
    n1_0 = [0] * num_patches
    n0_1 = [0] * num_patches
    n1_1 = [0] * num_patches
    n0_2 = [0] * num_patches
    n1_2 = [0] * num_patches

    bits = 3 * num_patches
    for bitstr, n in counts.items():
        s = bitstr.zfill(bits)
        for p in range(num_patches):
            b0 = int(s[-1 - (3 * p + 0)])  # q0
            b1 = int(s[-1 - (3 * p + 1)])  # q1
            b2 = int(s[-1 - (3 * p + 2)])  # q2
            totals[p] += n
            if b0 == 0:
                n0_0[p] += n
            else:
                n1_0[p] += n
            if b1 == 0:
                n0_1[p] += n
            else:
                n1_1[p] += n
            if b2 == 0:
                n0_2[p] += n
            else:
                n1_2[p] += n

    out = []
    for p in range(num_patches):
        if totals[p] == 0:
            out.append((0.0, 0.0, 0.0))
            continue
        Ey0 = (n0_0[p] - n1_0[p]) / totals[p]
        Ez1 = (n0_1[p] - n1_1[p]) / totals[p]
        Ey2 = (n0_2[p] - n1_2[p]) / totals[p]
        out.append((Ey0, Ez1, Ey2))
    return out


def binary_entropy_bits(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log(p, 2) - (1.0 - p) * math.log(1.0 - p, 2)


def fit_line(xs, ys):
    """
    Unweighted least-squares fit y ≈ a + b x.
    Returns (a, b, r2, se_b).
    """
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0, float("inf")
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    Sxx = sum((x - xbar) ** 2 for x in xs)
    if Sxx == 0.0:
        return ybar, 0.0, 0.0, float("inf")
    Sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = Sxy / Sxx
    a = ybar - b * xbar
    yhat = [a + b * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    sigma2 = ss_res / max(n - 2, 1)
    se_b = math.sqrt(sigma2 / Sxx) if Sxx > 0 else float("inf")
    return a, b, r2, se_b


# ----------------------------------------------------------------------
# Main curvature-visibility protocol
# ----------------------------------------------------------------------

def run(theta: float, backend_name: str, shots: int, num_patches: int, tag: str):
    # Loops matching Vybn-2 rectangles.
    rect_plus = [(1, 0), (0, 1), (-1, 0), (0, -1)]      # +1
    rect_minus = [(0, 1), (1, 0), (0, -1), (-1, 0)]     # -1
    zero_short = [(1, 0), (-1, 0), (0, 1), (0, -1)]     #  0 short
    zero_long = rect_plus + rect_minus                  #  0 long
    plus2 = rect_plus * 2                               # +2
    minus2 = rect_minus * 2                             # -2

    loops = [
        ("id",         0.0, []),
        ("zero_short", 0.0, zero_short),
        ("zero_long",  0.0, zero_long),
        ("plus1",     +1.0, rect_plus),
        ("minus1",    -1.0, rect_minus),
        ("plus2",     +2.0, plus2),
        ("minus2",    -2.0, minus2),
    ]

    # Shuffle loop order to soften slow drift.
    random.shuffle(loops)

    qr = QuantumRegister(3 * num_patches, "q")
    cr = ClassicalRegister(3 * num_patches, "m")
    labelled = []
    for loop_id, area, steps in loops:
        qc = QuantumCircuit(qr, cr, name=f"loop_{loop_id}")
        for p in range(num_patches):
            apply_loop_on_patch(qc, steps, theta, p)
        rotate_patches_to_measure(qc, num_patches)
        qc.measure(qr, cr)
        labelled.append((loop_id, area, qc))

    print(f"Building circuits for {num_patches} patches ({3*num_patches} qubits total)")
    for loop_id, area, qc in labelled:
        print(f"  {loop_id:10s}: area = {area:+.1f}, depth ≈ {qc.depth()}")

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa_circuits = pm.run([qc for (_, _, qc) in labelled])

    sampler = Sampler(backend)
    job = sampler.run(isa_circuits, shots=shots)
    result = job.result()

    per = {}  # loop_id -> list of (Ey0, Ez1, Ey2)
    for (loop_id, _, _), pub in zip(labelled, result):
        counts = counts_dict(pub)
        per[loop_id] = counts_to_patch_expectations(counts, num_patches)

    summary = {
        "theta": theta,
        "backend": backend_name,
        "shots": shots,
        "patches": num_patches,
        "tag": tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "per_patch": [],
    }

    rows = []

    print("\npatch  a(bound)   b(bound)   R2_b   vis(|b|/se)    S0(bulk)   alpha(bulk)  R2_α    vis1    vis2")

    for p in range(num_patches):
        def ey(loop_id: str) -> float:
            return per[loop_id][p][0]

        def ez(loop_id: str) -> float:
            return per[loop_id][p][1]

        # Boundary orientation slope from A ∈ {−2, −1, +1, +2}
        A_vals = []
        Y_vals = []
        for loop_id, area, _ in loops:
            if loop_id in ("minus2", "minus1", "plus1", "plus2"):
                A_vals.append(area)
                Y_vals.append(ey(loop_id))

        a_b, b_b, r2_b, se_b = fit_line(A_vals, Y_vals)
        vis = abs(b_b) / se_b if se_b > 0 and math.isfinite(se_b) else 0.0

        # Paired visibilities at |A| = 1,2
        vis1 = 0.0
        vis2 = 0.0
        if "plus1" in per and "minus1" in per:
            vis1 = 0.5 * (ey("plus1") - ey("minus1"))
        if "plus2" in per and "minus2" in per:
            vis2 = 0.5 * (ey("plus2") - ey("minus2"))

        # Bulk entropy vs |A|
        def s_of(loop_id: str) -> float:
            p1 = (1.0 - ez(loop_id)) / 2.0
            return binary_entropy_bits(p1)

        S0_short = s_of("zero_short")
        S1 = 0.5 * (s_of("plus1") + s_of("minus1"))
        S2 = 0.5 * (s_of("plus2") + s_of("minus2"))

        xs = [0.0, 1.0, 2.0]
        ys = [S0_short, S1, S2]
        a_alpha, alpha, r2_alpha, _ = fit_line(xs, ys)

        rows.append((p, a_b, b_b, r2_b, vis, a_alpha, alpha, r2_alpha, vis1, vis2))

        summary["per_patch"].append({
            "patch": p,
            "boundary_intercept": a_b,
            "boundary_slope": b_b,
            "boundary_R2": r2_b,
            "boundary_visibility": vis,
            "bulk_S0": a_alpha,
            "bulk_alpha": alpha,
            "bulk_R2": r2_alpha,
            "vis1": vis1,
            "vis2": vis2,
            "loops": {
                loop_id: {"Ey0": ey(loop_id), "Ez1": ez(loop_id)}
                for loop_id, _, _ in loops
            },
        })

    for (p, a_b, b_b, r2_b, vis, a_alpha, alpha, r2_alpha, vis1, vis2) in rows:
        print(
            f"{p:5d}  {a_b:+.6f}  {b_b:+.6f}  {r2_b: .3f}  {vis: .2f}     "
            f"{a_alpha:.6f}   {alpha:.6f}   {r2_alpha: .3f}   {vis1:+.4f}  {vis2:+.4f}"
        )

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    outdir = f"vybn2_curvature_{tag}_{ts}"
    os.makedirs(outdir, exist_ok=True)
    summary["outdir"] = outdir

    json_path = os.path.join(outdir, "curvature_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {json_path}")

    patches = [r[0] for r in rows]
    boundary_slopes = [r[2] for r in rows]
    boundary_vis = [r[4] for r in rows]
    bulk_alphas = [r[6] for r in rows]

    plt.figure()
    plt.bar(patches, boundary_slopes)
    plt.xlabel("patch index")
    plt.ylabel("boundary slope b")
    plt.title(f"Vybn rectangular: boundary orientation slope per patch (theta={theta:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "boundary_slopes.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.bar(patches, boundary_vis)
    plt.xlabel("patch index")
    plt.ylabel("|b| / stderr(b)")
    plt.title("Vybn rectangular: boundary curvature visibility per patch")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "boundary_visibility.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.bar(patches, bulk_alphas)
    plt.xlabel("patch index")
    plt.ylabel("bulk slope α")
    plt.title("Vybn rectangular: bulk entropy growth vs |area| per patch")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bulk_alphas.png"), dpi=200)
    plt.close()

    manifest_path = os.path.join(outdir, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("Vybn rectangular curvature visibility run\n")
        f.write(f"backend = {backend_name}\n")
        f.write(f"theta   = {theta:.6f}\n")
        f.write(f"shots   = {shots}\n")
        f.write(f"patches = {num_patches}\n")
        f.write("\nImages:\n")
        f.write("  boundary_slopes.png      – per-patch boundary slope b\n")
        f.write("  boundary_visibility.png  – per-patch |b|/stderr(b)\n")
        f.write("  bulk_alphas.png          – per-patch bulk swelling α\n")


def main():
    ap = argparse.ArgumentParser(
        description="Vybn rectangular universe on IBM Quantum: curvature visibility."
    )
    ap.add_argument("--theta", type=float, default=0.4,
                    help="rotation angle θ per time-step (radians)")
    ap.add_argument("--ibm-backend", type=str, default="ibm_fez",
                    help="backend name (e.g. ibm_fez)")
    ap.add_argument("--shots", type=int, default=4096,
                    help="shots per circuit")
    ap.add_argument("--patches", type=int, default=8,
                    help="number of 3-qubit patches")
    ap.add_argument("--tag", type=str, default="run",
                    help="freeform tag for output filenames")
    args = ap.parse_args()
    run(theta=args.theta,
        backend_name=args.ibm_backend,
        shots=args.shots,
        num_patches=args.patches,
        tag=args.tag)


if __name__ == "__main__":
    main()
