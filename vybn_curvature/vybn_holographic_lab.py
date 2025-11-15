#!/usr/bin/env python3
"""
vybn_holographic_lab.py

Three-qubit toy for bulk–boundary holonomy.

Qubit roles:
    q0 and q2 are the "boundary"
    q1 is the "bulk"

Dynamics:
    We give the chain two fixed interaction generators:

        H1 = X_0 X_1 ⊗ I_2   (left boundary coupled to bulk)
        H2 = I_0 ⊗ Y_1 X_2   (bulk coupled to right boundary)

    From these we define two "time flows" on the three-qubit system:

        U1(θ) = exp(-i θ H1 / 2)
        U2(θ) = exp(-i θ H2 / 2)

    A point (t1, t2) in a 2D "time plane" labels how far we have flowed
    along each direction. A closed loop is a walk on the integer lattice
    in (t1, t2) that returns to the origin, built from unit steps:

        +t1  → U1(+θ)
        -t1  → U1(-θ) = U1(+θ)†
        +t2  → U2(+θ)
        -t2  → U2(-θ) = U2(+θ)†

    Because H1 and H2 do not commute, different closed loops with the
    same net displacement can leave different "twists" (holonomy) on
    the three-qubit state.

Experiment:
    Start each run in |000> (boundary and bulk unexcited).
    Draw a random closed loop in the (t1, t2) plane with a fixed number
    of steps. Apply the corresponding sequence of U1 / U2 gates.

    For each loop we record:
        - the signed area A of the loop in the time plane,
        - Z expectation on q0 (left boundary),
        - Z expectation on q1 (bulk),
        - Z expectation on q2 (right boundary),
        - the von Neumann entropy of the bulk qubit S(q1) in bits.

We then group loops by area and print the mean values in each group.
Everything is done with numpy on your laptop; no hardware calls.
"""

import argparse
import math
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np

# Single-qubit matrices

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)

# Three-qubit convenience

def kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(a, b), c)


IDENTITY_8 = np.eye(8, dtype=complex)

# Interaction generators

H1 = kron3(SIGMA_X, SIGMA_X, IDENTITY_2)   # boundary-left ↔ bulk
H2 = kron3(IDENTITY_2, SIGMA_Y, SIGMA_X)   # bulk ↔ boundary-right


def unitary_from_H(H: np.ndarray, theta: float) -> np.ndarray:
    """
    Compute U = exp(-i theta H / 2) for H^2 = I, using:
        U = cos(theta/2) I - i sin(theta/2) H
    """
    return math.cos(theta / 2.0) * IDENTITY_8 - 1j * math.sin(theta / 2.0) * H


def unitary_for_step(dt1: int, dt2: int, theta: float) -> np.ndarray:
    """
    Map a single lattice step in (t1, t2) to the three-qubit unitary.
    """
    if dt1 == 1 and dt2 == 0:
        return unitary_from_H(H1, theta)
    if dt1 == -1 and dt2 == 0:
        return unitary_from_H(H1, -theta)
    if dt1 == 0 and dt2 == 1:
        return unitary_from_H(H2, theta)
    if dt1 == 0 and dt2 == -1:
        return unitary_from_H(H2, -theta)
    raise ValueError(f"Non-unit step encountered: ({dt1}, {dt2})")


# Time-plane paths

Point = Tuple[int, int]  # (t1, t2)


def generate_closed_loop(num_steps: int, rng: random.Random) -> List[Point]:
    """
    Generate a random closed loop on Z^2 with num_steps steps.

    Returns the vertices [p0, p1, ..., pN], with p0 = pN = (0, 0).
    """
    if num_steps < 4 or num_steps % 2 == 1:
        raise ValueError("num_steps should be even and >= 4 for closure to be plausible.")

    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    max_tries = 10000

    for _ in range(max_tries):
        t1, t2 = 0, 0
        path: List[Point] = [(t1, t2)]
        for _ in range(num_steps):
            dt1, dt2 = rng.choice(steps)
            t1 += dt1
            t2 += dt2
            path.append((t1, t2))
        if (t1, t2) == (0, 0):
            if path[-1] != path[0]:
                path.append(path[0])
            return path

    raise RuntimeError(f"Failed to find closed loop of {num_steps} steps after {max_tries} tries.")


def signed_area(path: List[Point]) -> float:
    """Signed area of a closed polygon in the (t1, t2) plane via the shoelace formula."""
    if path[0] != path[-1]:
        raise ValueError("Path is not closed for area computation.")

    area2 = 0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        area2 += x0 * y1 - x1 * y0
    return 0.5 * area2


# Observables

Z_ON_Q0 = kron3(SIGMA_Z, IDENTITY_2, IDENTITY_2)
Z_ON_Q1 = kron3(IDENTITY_2, SIGMA_Z, IDENTITY_2)
Z_ON_Q2 = kron3(IDENTITY_2, IDENTITY_2, SIGMA_Z)


def expectation(op: np.ndarray, state: np.ndarray) -> float:
    """Return real expectation value <op> for a normalized state."""
    return float(np.real(np.vdot(state, op @ state)))


def reduced_density_bulk(state: np.ndarray) -> np.ndarray:
    """
    Reduced density matrix of the bulk qubit q1 after tracing out q0 and q2.

    state is an 8-component pure state vector in the basis |q0 q1 q2>.
    """
    rho_full = np.outer(state, np.conjugate(state))
    rho_bulk = np.zeros((2, 2), dtype=complex)

    for b0 in (0, 1):  # bulk index
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
    """Von Neumann entropy S(ρ) in bits for a 2x2 density matrix."""
    evals = np.linalg.eigvals(rho)
    evals = np.real_if_close(evals)
    evals = np.clip(evals, 0.0, 1.0)
    total = float(np.sum(evals))
    if total > 0:
        evals = evals / total

    S = 0.0
    for lam in evals:
        lam = float(lam)
        if lam > 0.0:
            S -= lam * math.log(lam, 2)
    return S


# Main loop

def run_holographic_experiment(theta: float,
                               num_steps: int,
                               num_loops: int,
                               seed: Optional[int]) -> None:
    rng = random.Random(seed)
    if seed is not None:
        print(f"rng seed = {seed}")

    print(f"theta = {theta:.3f} rad, steps per loop = {num_steps}, loops = {num_loops}")

    # Initial three-qubit state |000>
    psi0 = np.zeros(8, dtype=complex)
    psi0[0] = 1.0

    print("idx\tarea\t\tZ(q0)\t\tZ(q1)\t\tZ(q2)\t\tS_bulk")

    records_by_area: Dict[float, List[Tuple[float, float, float, float]]] = defaultdict(list)

    for idx in range(num_loops):
        path = generate_closed_loop(num_steps, rng)
        area = signed_area(path)

        psi = psi0.copy()
        for (t1_0, t2_0), (t1_1, t2_1) in zip(path[:-1], path[1:]):
            dt1 = t1_1 - t1_0
            dt2 = t2_1 - t2_0
            U = unitary_for_step(dt1, dt2, theta)
            psi = U @ psi

        z_q0 = expectation(Z_ON_Q0, psi)
        z_q1 = expectation(Z_ON_Q1, psi)
        z_q2 = expectation(Z_ON_Q2, psi)

        rho_bulk = reduced_density_bulk(psi)
        s_bulk = entropy_bits(rho_bulk)

        records_by_area[area].append((z_q0, z_q1, z_q2, s_bulk))

        print(f"{idx:3d}\t{area: .3f}\t\t{z_q0: .6f}\t{z_q1: .6f}\t{z_q2: .6f}\t{s_bulk: .6f}")

    print("\nSummary by area (means over loops with the same area):")
    print("area\tcount\tZ(q0)\t\tZ(q1)\t\tZ(q2)\t\tS_bulk")
    for area in sorted(records_by_area.keys()):
        vals = records_by_area[area]
        count = len(vals)
        mean_z0 = sum(v[0] for v in vals) / count
        mean_z1 = sum(v[1] for v in vals) / count
        mean_z2 = sum(v[2] for v in vals) / count
        mean_s = sum(v[3] for v in vals) / count
        print(f"{area: .3f}\t{count:5d}\t{mean_z0: .6f}\t{mean_z1: .6f}\t{mean_z2: .6f}\t{mean_s: .6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-qubit bulk–boundary holonomy lab.")
    parser.add_argument("--theta", type=float, default=0.4,
                        help="rotation angle θ for each time-step (radians)")
    parser.add_argument("--steps", type=int, default=20,
                        help="number of steps per closed loop (even)")
    parser.add_argument("--loops", type=int, default=200,
                        help="number of random loops to sample")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for reproducibility")
    args = parser.parse_args()

    run_holographic_experiment(theta=args.theta,
                               num_steps=args.steps,
                               num_loops=args.loops,
                               seed=args.seed)


if __name__ == "__main__":
    main()
