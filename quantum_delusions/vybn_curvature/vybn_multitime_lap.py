#!/usr/bin/env python3
"""
vybn_multitime_lab.py

Brownian holonomy baseline for a single qubit.

We pretend there are two "time directions", t1 and t2. A path through
this 2D time-plane is a walk on the integer lattice in (t1, t2).

Each unit step in the time-plane is mapped to a unitary on a single qubit:

    +t1  →  Rx(+θ)
    -t1  →  Rx(-θ)
    +t2  →  Ry(+θ)
    -t2  →  Ry(-θ)

So a closed loop in the time-plane becomes a product of small rotations
on the Bloch sphere. Even though the loop is "closed" in (t1, t2),
the quantum state need not return to where it started; the mismatch
is the holonomy.

This file does only classical linear algebra, no IBM, no Qiskit:

  1. Generate random closed loops on Z^2 with a given number of steps.
  2. Compute the signed area of each loop using the shoelace formula.
  3. Evolve |+x> through the corresponding SU(2) rotations.
  4. Compute <Y> for each loop and summarize how <Y> depends on area.

This is the SU(2) baseline. If we later want a "polar time" rule, we
swap out or extend the mapping from steps → unitaries or the way we
combine them and see whether the statistics change.
"""

import argparse
import math
import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np


# ----- SU(2) machinery ----- #

# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)


def rx(theta: float) -> np.ndarray:
    """Rotation about X by angle theta."""
    return math.cos(theta / 2) * IDENTITY - 1j * math.sin(theta / 2) * SIGMA_X


def ry(theta: float) -> np.ndarray:
    """Rotation about Y by angle theta."""
    return math.cos(theta / 2) * IDENTITY - 1j * math.sin(theta / 2) * SIGMA_Y


def expectation_y(state: np.ndarray) -> float:
    """Return <Y> = ψ† σ_y ψ for a normalized 2-component state vector."""
    return float(np.real(np.conjugate(state).T @ (SIGMA_Y @ state)))


# ----- Time-plane paths ----- #


Step = Tuple[int, int]      # (Δt1, Δt2)
Point = Tuple[int, int]     # (t1, t2)


def generate_closed_loop(num_steps: int, rng: random.Random) -> List[Point]:
    """
    Generate a random closed loop with num_steps steps on Z^2.

    Returns the vertices [p0, p1, ..., pN], with p0 = pN = (0, 0).
    """
    if num_steps < 4 or num_steps % 2 == 1:
        raise ValueError("num_steps should be even and >= 4 for closure to be plausible.")

    moves: List[Step] = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    max_tries = 10000
    for _ in range(max_tries):
        t1, t2 = 0, 0
        path: List[Point] = [(t1, t2)]
        for _ in range(num_steps):
            dt1, dt2 = rng.choice(moves)
            t1 += dt1
            t2 += dt2
            path.append((t1, t2))
        if (t1, t2) == (0, 0):
            # ensure explicit closure for area formula
            if path[-1] != path[0]:
                path.append(path[0])
            return path

    raise RuntimeError(f"Failed to find closed loop of {num_steps} steps after {max_tries} tries.")


def signed_area(path: List[Point]) -> float:
    """
    Signed area of a closed polygon path using the shoelace formula.

    path should have first vertex = last vertex.
    """
    if path[0] != path[-1]:
        raise ValueError("Path is not closed for area computation.")

    area2 = 0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        area2 += x0 * y1 - x1 * y0
    return 0.5 * area2


def path_to_gates(path: List[Point], theta: float) -> List[np.ndarray]:
    """
    Map edges of the time-plane path to SU(2) rotations.

    For each step (Δt1, Δt2):
      ( 1,  0) → Rx(+θ)
      (-1,  0) → Rx(-θ)
      ( 0,  1) → Ry(+θ)
      ( 0, -1) → Ry(-θ)
    """
    gates: List[np.ndarray] = []
    for (t1_0, t2_0), (t1_1, t2_1) in zip(path[:-1], path[1:]):
        dt1 = t1_1 - t1_0
        dt2 = t2_1 - t2_0
        if dt1 == 1 and dt2 == 0:
            gates.append(rx(+theta))
        elif dt1 == -1 and dt2 == 0:
            gates.append(rx(-theta))
        elif dt1 == 0 and dt2 == 1:
            gates.append(ry(+theta))
        elif dt1 == 0 and dt2 == -1:
            gates.append(ry(-theta))
        else:
            raise ValueError(f"Non-unit step encountered: {(dt1, dt2)}")
    return gates


# ----- Experiment driver ----- #


def evolve_plus_x(gates: List[np.ndarray]) -> np.ndarray:
    """
    Start in |+x> = (|0> + |1>)/sqrt(2) and apply gates in order.
    """
    plus_x = (1 / math.sqrt(2)) * np.array([1.0, 1.0], dtype=complex)
    psi = plus_x
    for G in gates:
        psi = G @ psi
    return psi


def run_brownian_experiment(theta: float, num_steps: int, num_loops: int, seed: int | None) -> None:
    """
    Generate num_loops random closed loops, compute area and <Y> for each,
    and print both per-loop data and a summary aggregated by area.
    """
    rng = random.Random(seed)

    print(f"theta = {theta:.3f} rad, steps per loop = {num_steps}, loops = {num_loops}")
    if seed is not None:
        print(f"rng seed = {seed}")
    print("loop_idx\tarea\t\t<Y>")

    records: list[tuple[float, float]] = []

    for idx in range(num_loops):
        path = generate_closed_loop(num_steps, rng)
        area = signed_area(path)
        gates = path_to_gates(path, theta)
        psi_final = evolve_plus_x(gates)
        ey = expectation_y(psi_final)
        records.append((area, ey))
        print(f"{idx:3d}\t\t{area: .3f}\t\t{ey: .6f}")

    # aggregate by area
    buckets: dict[float, list[float]] = defaultdict(list)
    for area, ey in records:
        buckets[area].append(ey)

    print("\nSummary by area:")
    print("area\tcount\tmean(<Y>)\tstd(<Y>)")
    for area in sorted(buckets.keys()):
        vals = buckets[area]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = math.sqrt(var)
        print(f"{area: .3f}\t{n:5d}\t{mean: .6f}\t{std: .6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Brownian holonomy baseline for a single qubit.")
    parser.add_argument("--theta", type=float, default=0.4, help="rotation angle θ in radians")
    parser.add_argument("--steps", type=int, default=20, help="number of time-plane steps per loop (even)")
    parser.add_argument("--loops", type=int, default=200, help="number of random loops to sample")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = parser.parse_args()

    run_brownian_experiment(theta=args.theta, num_steps=args.steps, num_loops=args.loops, seed=args.seed)


if __name__ == "__main__":
    main()
