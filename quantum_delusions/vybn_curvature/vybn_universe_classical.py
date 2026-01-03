#!/usr/bin/env python3
"""
vybn_universe_classical.py

Vybn-2: rectangular holonomy universe in at least two-dimensional time (classical).

This is the laptop-side universe we want *fully* legible before touching the chip.

Model:
    q0, q2 = "boundary" qubits
    q1     = "bulk" qubit

    H1 = X_0 X_1 ⊗ I_2   (left boundary coupled to bulk)
    H2 = I_0 ⊗ Y_1 X_2   (bulk coupled to right boundary)

Time lives in a 2D control plane (t1, t2). A "history" is a closed loop built from
steps (±1,0) or (0,±1). Each step applies exp(-i θ H1 / 2) or exp(-i θ H2 / 2)
with ±θ depending on direction.

Vybn-2 restricts to a tiny family of rectangular loops with known areas:

    id          : no steps           (area 0)
    zero_short  : (1,0),(-1,0),(0,1),(0,-1)                    (area 0, short)
    rect_plus   : (1,0),(0,1),(-1,0),(0,-1)                    (area +1)
    rect_minus  : (0,1),(1,0),(0,-1),(-1,0)                    (area -1)
    plus2       : rect_plus repeated twice                     (area +2)
    minus2      : rect_minus repeated twice                    (area -2)
    zero_long   : rect_plus followed by rect_minus             (area 0, long)

Two observables per loop:

    • boundary:  ⟨Y_boundary⟩ in a 1-qubit toy driven by the same loop
                 (start in |+x>, apply Rx/Ry by θ along the path)
    • bulk:      S(q1) in bits, from the 3-qubit chain evolved under H1/H2.

This cleanly realizes the law we care about:

    ⟨Y_boundary⟩ is odd in signed area A (slope k near A = 0)
    S(q1) is mainly a function of |A| (slope α in S ≈ S0 + α|A|)

Outputs for each run:

    - console table of areas, depths, ⟨Y_boundary⟩, S(q1)
    - PNGs:
        loop_paths.png
        boundary_vs_area.png         (with labels)
        bulk_entropy_vs_abs_area.png (with labels)
    - manifest.txt (human summary)
    - rectangles_summary.json (θ, loop data, k, α, etc.)
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

Point = Tuple[int, int]

# ----------------------------------------------------------------------
# 1-qubit boundary toy
# ----------------------------------------------------------------------

SIGMA_X_1Q = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y_1Q = np.array([[0, -1j], [1j, 0]], dtype=complex)
IDENTITY_1Q = np.eye(2, dtype=complex)


def rx(theta: float) -> np.ndarray:
    return math.cos(theta / 2.0) * IDENTITY_1Q - 1j * math.sin(theta / 2.0) * SIGMA_X_1Q


def ry(theta: float) -> np.ndarray:
    return math.cos(theta / 2.0) * IDENTITY_1Q - 1j * math.sin(theta / 2.0) * SIGMA_Y_1Q


def expectation_y_1q(psi: np.ndarray) -> float:
    return float(np.real(np.conjugate(psi).T @ (SIGMA_Y_1Q @ psi)))


def path_from_steps(steps: List[Point]) -> List[Point]:
    t1 = t2 = 0
    path: List[Point] = [(t1, t2)]
    for dt1, dt2 in steps:
        t1 += dt1
        t2 += dt2
        path.append((t1, t2))
    if (t1, t2) != (0, 0):
        raise ValueError(f"Loop not closed; final point = {(t1, t2)}")
    if path[-1] != path[0]:
        path.append(path[0])
    return path


def boundary_y_for_path_1q(path: List[Point], theta: float) -> float:
    """Drive a single qubit with Rx/Ry along the loop and measure ⟨Y⟩."""
    psi = (1.0 / math.sqrt(2.0)) * np.array([1.0, 1.0], dtype=complex)  # |+x>

    for (t10, t20), (t11, t21) in zip(path[:-1], path[1:]):
        dt1 = t11 - t10
        dt2 = t21 - t20
        if dt1 == 1 and dt2 == 0:
            U = rx(+theta)
        elif dt1 == -1 and dt2 == 0:
            U = rx(-theta)
        elif dt1 == 0 and dt2 == 1:
            U = ry(+theta)
        elif dt1 == 0 and dt2 == -1:
            U = ry(-theta)
        else:
            raise ValueError(f"Non-unit step in 1q path: {(dt1, dt2)}")
        psi = U @ psi

    return expectation_y_1q(psi)


# ----------------------------------------------------------------------
# 3-qubit bulk model
# ----------------------------------------------------------------------

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)
IDENTITY_8 = np.eye(8, dtype=complex)


def kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(a, b), c)


H1 = kron3(SIGMA_X, SIGMA_X, IDENTITY_2)
H2 = kron3(IDENTITY_2, SIGMA_Y, IDENTITY_2 @ SIGMA_X)


def unitary_from_H(H: np.ndarray, theta: float) -> np.ndarray:
    return math.cos(theta / 2.0) * IDENTITY_8 - 1j * math.sin(theta / 2.0) * H


def unitary_for_step_3q(dt1: int, dt2: int, theta: float) -> np.ndarray:
    if dt1 == 1 and dt2 == 0:
        return unitary_from_H(H1, theta)
    if dt1 == -1 and dt2 == 0:
        return unitary_from_H(H1, -theta)
    if dt1 == 0 and dt2 == 1:
        return unitary_from_H(H2, theta)
    if dt1 == 0 and dt2 == -1:
        return unitary_from_H(H2, -theta)
    raise ValueError(f"Non-unit step: {(dt1, dt2)}")


def reduced_density_bulk(psi: np.ndarray) -> np.ndarray:
    """Trace out q0 and q2; keep q1. Basis ordering: |q0 q1 q2>."""
    rho_full = np.outer(psi, np.conjugate(psi))
    rho_bulk = np.zeros((2, 2), dtype=complex)

    for b0 in (0, 1):
        for b1 in (0, 1):
            val = 0.0 + 0.0j
            for q0 in (0, 1):
                for q2 in (0, 1):
                    idx_i = (q0 << 2) | (b0 << 1) | q2
                    idx_j = (q0 << 2) | (b1 << 1) | q2
                    val += rho_full[idx_i, idx_j]
            rho_bulk[b0, b1] = val

    return rho_bulk


def entropy_bits(rho: np.ndarray) -> float:
    evals = np.linalg.eigvals(rho)
    evals = np.real_if_close(evals)
    evals = np.clip(evals, 0.0, 1.0)

    total = float(np.sum(evals))
    if total > 0.0:
        evals = evals / total

    S = 0.0
    for lam in evals:
        lam = float(lam)
        if lam > 0.0:
            S -= lam * math.log(lam, 2)
    return S


def signed_area(path: List[Point]) -> float:
    if path[0] != path[-1]:
        raise ValueError("Path must be closed.")
    area2 = 0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        area2 += x0 * y1 - x1 * y0
    return 0.5 * area2


def simulate_rectangular_loop(steps: List[Point], theta: float):
    """
    Given a step list, build the path and return:

        area, depth, Y_boundary, S_bulk, path_points
    """
    path = path_from_steps(steps)
    area = signed_area(path)

    # boundary 1-qubit toy
    y_boundary = boundary_y_for_path_1q(path, theta)

    # bulk 3-qubit chain
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1.0  # |000>
    depth = 0
    for (t10, t20), (t11, t21) in zip(path[:-1], path[1:]):
        dt1 = t11 - t10
        dt2 = t21 - t20
        U = unitary_for_step_3q(dt1, dt2, theta)
        psi = U @ psi
        depth += 1

    rho_bulk = reduced_density_bulk(psi)
    S_bulk = entropy_bits(rho_bulk)

    return area, depth, y_boundary, S_bulk, path


# ----------------------------------------------------------------------
# Rectangular loop definitions
# ----------------------------------------------------------------------

rect_plus = [(1, 0), (0, 1), (-1, 0), (0, -1)]          # area +1
rect_minus = [(0, 1), (1, 0), (0, -1), (-1, 0)]         # area -1
plus2 = rect_plus * 2                                   # area +2
minus2 = rect_minus * 2                                 # area -2
zero_short = [(1, 0), (-1, 0), (0, 1), (0, -1)]         # area 0, short
zero_long = rect_plus + rect_minus                      # area 0, long

loops_def: Dict[str, List[Point]] = {
    "id": [],
    "zero_short": zero_short,
    "rect_plus": rect_plus,
    "rect_minus": rect_minus,
    "plus2": plus2,
    "minus2": minus2,
    "zero_long": zero_long,
}


# ----------------------------------------------------------------------
# Small linear fit
# ----------------------------------------------------------------------

def fit_line(xs, ys):
    """
    Unweighted least-squares fit y ≈ a + b x.
    Returns (a, b, r2).
    """
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    Sxx = sum((x - xbar) ** 2 for x in xs)
    if Sxx == 0.0:
        return ybar, 0.0, 0.0
    Sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = Sxy / Sxx
    a = ybar - b * xbar
    yhat = [a + b * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, r2


# ----------------------------------------------------------------------
# Main classical run
# ----------------------------------------------------------------------

def run(theta: float, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    print("Vybn-2 rectangular universe (classical)")
    print(f"theta = {theta:.3f}")
    print(f"gallery directory: {outdir}")

    results = {}
    areas: List[float] = []
    yvals: List[float] = []
    svals: List[float] = []

    for name, steps in loops_def.items():
        if not steps:
            area = 0.0
            depth = 0
            yb = 0.0
            S = 0.0
            path = [(0, 0), (0, 0)]
        else:
            area, depth, yb, S, path = simulate_rectangular_loop(steps, theta)
        results[name] = {
            "area": area,
            "depth": depth,
            "Y_boundary": yb,
            "S_bulk": S,
            "path": path,
        }
        areas.append(area)
        yvals.append(yb)
        svals.append(S)

    print("\nLoop summary:")
    print("name         area    depth    <Y_boundary>   S(q1)")
    for name, info in results.items():
        print(
            f"{name:11s} {info['area']: .1f}   {info['depth']:5d}   "
            f"{info['Y_boundary']:+.6f}   {info['S_bulk']:.6f}"
        )

    # Paths overlay
    plt.figure()
    for name, info in results.items():
        path = info["path"]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1, label=name)
    plt.xlabel("t1")
    plt.ylabel("t2")
    plt.axis("equal")
    plt.title("Vybn-2 rectangular loops in two-dimensional time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loop_paths.png"), dpi=200)
    plt.close()

    # Boundary vs area (annotated)
    plt.figure()
    plt.scatter(areas, yvals, s=60)
    for x, y, name in zip(areas, yvals, results.keys()):
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.axhline(0.0, color="gray", linewidth=0.5)
    plt.xlabel("signed area in (t1, t2)")
    plt.ylabel("<Y_boundary> in 1-qubit toy")
    plt.title("Vybn-2: boundary signal vs loop area")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "boundary_vs_area.png"), dpi=200)
    plt.close()

    # Bulk vs |area| (annotated)
    abs_areas = [abs(a) for a in areas]
    plt.figure()
    plt.scatter(abs_areas, svals, s=60)
    for x, y, name in zip(abs_areas, svals, results.keys()):
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("|area| in (t1, t2)")
    plt.ylabel("S(q1) in bits")
    plt.title("Vybn-2: bulk entropy vs |area|")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bulk_entropy_vs_abs_area.png"), dpi=200)
    plt.close()

    # Fits: boundary and bulk
    A_fit = [
        results["minus2"]["area"],
        results["rect_minus"]["area"],
        results["rect_plus"]["area"],
        results["plus2"]["area"],
    ]
    Y_fit = [
        results["minus2"]["Y_boundary"],
        results["rect_minus"]["Y_boundary"],
        results["rect_plus"]["Y_boundary"],
        results["plus2"]["Y_boundary"],
    ]
    a_b, k, r2_b = fit_line(A_fit, Y_fit)

    S0 = results["zero_short"]["S_bulk"]
    S1 = 0.5 * (results["rect_plus"]["S_bulk"] + results["rect_minus"]["S_bulk"])
    S2 = 0.5 * (results["plus2"]["S_bulk"] + results["minus2"]["S_bulk"])
    xs_bulk = [0.0, 1.0, 2.0]
    ys_bulk = [S0, S1, S2]
    a_alpha, alpha, r2_alpha = fit_line(xs_bulk, ys_bulk)

    print("\nBoundary fit <Y_boundary> ≈ a + k * A (A ∈ {−2,−1,+1,+2}):")
    print(f"  a ≈ {a_b:+.6f}")
    print(f"  k ≈ {k:+.6f}")
    print(f"  R^2 ≈ {r2_b:.6f}")

    print("\nBulk fit S(q1) ≈ S0 + α |A| (|A| ∈ {0,1,2}):")
    print(f"  S0 ≈ {a_alpha:.6f}")
    print(f"  α  ≈ {alpha:.6f}")
    print(f"  R^2 ≈ {r2_alpha:.6f}")

    # Manifest + JSON
    manifest_path = os.path.join(outdir, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("Vybn-2 rectangular holonomy universe (classical)\n")
        f.write(f"theta = {theta:.6f}\n\n")
        f.write("Loops:\n")
        for name, info in results.items():
            f.write(
                f"  {name:11s} area={info['area']: .1f}, depth={info['depth']:3d}, "
                f"<Y_boundary>={info['Y_boundary']:+.6f}, S(q1)={info['S_bulk']:.6f}\n"
            )
        f.write("\nBoundary fit:\n")
        f.write(f"  a = {a_b:+.6f}\n")
        f.write(f"  k = {k:+.6f}\n")
        f.write(f"  R^2 = {r2_b:.6f}\n")
        f.write("\nBulk fit:\n")
        f.write(f"  S0 = {a_alpha:.6f}\n")
        f.write(f"  α  = {alpha:.6f}\n")
        f.write(f"  R^2 = {r2_alpha:.6f}\n")

    summary = {
        "theta": theta,
        "loops": {
            name: {
                "area": info["area"],
                "depth": info["depth"],
                "Y_boundary": info["Y_boundary"],
                "S_bulk": info["S_bulk"],
            }
            for name, info in results.items()
        },
        "boundary_fit": {"a": a_b, "k": k, "R2": r2_b},
        "bulk_fit": {"S0": a_alpha, "alpha": alpha, "R2": r2_alpha},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(outdir, "rectangles_summary.json"), "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    print(f"\nWrote {os.path.join(outdir, 'rectangles_summary.json')}")


if __name__ == "__main__":
    theta = 0.4
    ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    outdir = f"vybn2_rectangles_{ts}"
    run(theta, outdir)
