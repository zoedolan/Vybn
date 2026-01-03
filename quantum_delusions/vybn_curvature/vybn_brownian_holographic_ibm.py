#!/usr/bin/env python3
"""
vybn_brownian_holographic_ibm.py

Plain English summary:
Brownian holonomy in two-dimensional time on IBM Quantum.

This file runs a single, very specific experiment: we build the tiniest "universe" that can feel curved time—a three-qubit strip with two edges and a middle—and stir it along closed loops in a two-dimensional time plane. Each loop is a random walk in two time directions; we treat the walk as a shape in time-space, compute its signed area, and send the corresponding circuit to an IBM backend.

For every loop, we measure the Y spin on the left edge qubit and ask how that edge signal depends on the loop’s signed area. The main output is a single number, lambda_hat, the best-fit slope relating area and boundary response. In ideal simulations this slope is our discrete “time curvature.” On real hardware it becomes a faint but telling whisper of that curvature, riding on top of the chip’s noise.

In plain language: this script checks whether a three-qubit line on a real device behaves like a tiny hologram of time. If lambda_hat keeps showing up with the predicted sign and similar size across runs and backends, it is evidence that the shape of how we move through time leaves a measurable fingerprint on the edge—and a hint that larger versions of this idea could link quantum information flow, spacetime-like geometry, and more robust, geometry-based quantum gates.

Technical details:
One-shot Brownian holonomy experiment on IBM Quantum.

We use the same three-qubit bulk–boundary geometry as in vybn_holographic_ibm.py:
    q0 and q2 are "boundary" qubits
    q1 is the "bulk" qubit

We drive the chain with random closed loops in a two-dimensional time plane (t1, t2).
Each step in the loop applies either U1(±θ) or U2(±θ):

    U1(θ) = exp(-i θ X0 X1 / 2)
    U2(θ) = exp(-i θ Y1 X2 / 2)

For each randomly generated closed loop, we:
    - compute its signed area A in the (t1, t2) plane,
    - compile the corresponding circuit on a single three-qubit chain,
    - measure Y on the left boundary qubit q0 (via a final basis rotation).

The main quantity of interest is the correlation between loop area and boundary
measurement, estimated as

    lambda_hat = sum_j A_j * <Y0>_j / sum_j A_j^2

over all loops with non-zero area. In the ideal simulator this converges to the
curvature coefficient you measured previously; on hardware it tells you how much
of that odd-in-area signal survives the device noise.
"""

import argparse
import math
import random
from typing import List, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# ---------------------------------------------------------------------------
# Gate decompositions for H1 and H2, copied from vybn_holographic_ibm.py
# ---------------------------------------------------------------------------

def apply_u1(qc: QuantumCircuit, theta: float) -> None:
    """
    Apply U1(theta) = exp(-i theta X0 X1 / 2) to qubits q0, q1.

    Decomposition:
        X ⊗ X = (H ⊗ H) Z ⊗ Z (H ⊗ H)
        exp(-i θ Z⊗Z / 2) = CX RZ(θ) CX
    """
    q0, q1 = 0, 1

    qc.h(q0)
    qc.h(q1)

    qc.cx(q0, q1)
    qc.rz(theta, q1)
    qc.cx(q0, q1)

    qc.h(q0)
    qc.h(q1)


def apply_u2(qc: QuantumCircuit, theta: float) -> None:
    """
    Apply U2(theta) = exp(-i theta Y1 X2 / 2) to qubits q1, q2.

    Choose V1, V2 so that:
        V1 Z V1† = Y
        V2 Z V2† = X

    With V1 = Rx(-π/2), V2 = H:
        Y ⊗ X = (V1 ⊗ V2) Z ⊗ Z (V1 ⊗ V2)†
    """
    q1, q2 = 1, 2

    # pre-rotation to map Z⊗Z → Y⊗X
    qc.rx(math.pi / 2, q1)
    qc.h(q2)

    # ZZ entangler in that rotated basis
    qc.cx(q1, q2)
    qc.rz(theta, q2)
    qc.cx(q1, q2)

    # rotate back
    qc.rx(-math.pi / 2, q1)
    qc.h(q2)


# ---------------------------------------------------------------------------
# Brownian loop generation in time-space
# ---------------------------------------------------------------------------

Step = Tuple[int, int]   # (dt1, dt2)


def generate_closed_loop_steps(num_steps: int, rng: random.Random) -> List[Step]:
    """
    Generate a random closed loop in (t1, t2) with num_steps steps.

    Each step is one of (±1, 0) or (0, ±1). We reject and retry until the walk
    returns to the origin after num_steps.
    """
    if num_steps < 4 or num_steps % 2 != 0:
        raise ValueError("num_steps should be an even integer >= 4.")

    step_set: List[Step] = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    max_tries = 10000

    for _ in range(max_tries):
        t1 = 0
        t2 = 0
        steps: List[Step] = []
        for _ in range(num_steps):
            dt1, dt2 = rng.choice(step_set)
            t1 += dt1
            t2 += dt2
            steps.append((dt1, dt2))
        if t1 == 0 and t2 == 0:
            return steps

    raise RuntimeError(f"Failed to find closed loop of {num_steps} steps after {max_tries} tries.")


def signed_area_from_steps(steps: List[Step]) -> float:
    """
    Compute signed area in (t1, t2) plane from dt1, dt2 steps using the shoelace formula.
    """
    t1 = 0
    t2 = 0
    path = [(t1, t2)]
    for dt1, dt2 in steps:
        t1 += dt1
        t2 += dt2
        path.append((t1, t2))

    if path[-1] != path[0]:
        raise ValueError(f"Loop not closed; final point {path[-1]} instead of {path[0]}.")

    area2 = 0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        area2 += x0 * y1 - x1 * y0

    return 0.5 * area2


# ---------------------------------------------------------------------------
# Measurement utilities
# ---------------------------------------------------------------------------

def counts_to_expectation_z(counts: dict) -> float:
    """
    For a single-qubit Z-basis measurement, compute <Z> = p0 - p1 from counts.
    """
    n0 = counts.get("0", 0)
    n1 = counts.get("1", 0)
    total = n0 + n1
    if total == 0:
        return 0.0
    return (n0 - n1) / total


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

def build_loop_circuit(loop_id: str, steps: List[Step], theta: float) -> QuantumCircuit:
    """
    Build a 3-qubit circuit for a given Brownian loop in (t1, t2) space.

    We start in |000>, apply U1 or U2 for each step, and measure Y on the
    left boundary qubit q0 via a final S†–H basis rotation.
    """
    qr = QuantumRegister(3, "q")
    cr = ClassicalRegister(1, "m")
    qc = QuantumCircuit(qr, cr, name=f"loop_{loop_id}")

    for dt1, dt2 in steps:
        if dt1 == 1 and dt2 == 0:
            apply_u1(qc, theta)
        elif dt1 == -1 and dt2 == 0:
            apply_u1(qc, -theta)
        elif dt1 == 0 and dt2 == 1:
            apply_u2(qc, theta)
        elif dt1 == 0 and dt2 == -1:
            apply_u2(qc, -theta)
        else:
            raise ValueError(f"Non-unit step encountered: ({dt1}, {dt2})")

    # rotate into Y basis on q0 so that Z-measurement gives <Y0>
    qc.sdg(qr[0])
    qc.h(qr[0])
    qc.measure(qr[0], cr[0])

    return qc


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_brownian_holographic_ibm(
    theta: float,
    backend_name: str,
    shots: int,
    num_loops: int,
    num_steps: int,
    seed: int,
) -> None:
    """
    Build a Brownian ensemble of loops, run them on IBM hardware, and
    print the area–boundary correlation.
    """
    rng = random.Random(seed)

    loops = []
    for j in range(num_loops):
        steps = generate_closed_loop_steps(num_steps, rng)
        area = signed_area_from_steps(steps)
        loop_id = f"L{j:03d}"
        loops.append((loop_id, area, steps))

    print("Generated loops (loop_id, area, num_steps, dt path snippet):")
    for loop_id, area, steps in loops:
        preview = " ".join(f"({dt1},{dt2})" for dt1, dt2 in steps[:6])
        if len(steps) > 6:
            preview += " ..."
        print(f"  {loop_id}: area = {area:+.1f}, steps = {len(steps)}, path ≈ {preview}")

    labelled_circuits = []
    for loop_id, area, steps in loops:
        qc = build_loop_circuit(loop_id, steps, theta)
        labelled_circuits.append((loop_id, area, qc))

    print("\nCircuit depths:")
    for loop_id, area, qc in labelled_circuits:
        print(f"  {loop_id}: area = {area:+.1f}, depth = {qc.depth()}")

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    circuits = [qc for (_, _, qc) in labelled_circuits]
    isa_circuits = pm.run(circuits)

    sampler = Sampler(backend)
    job = sampler.run(isa_circuits, shots=shots)
    result = job.result()

    print(f"\nIBM backend '{backend_name}', shots per circuit = {shots}")
    print("loop_id\tarea\t\t<Y0>")

    areas = []
    y_values = []

    for (loop_id, area, _), pub_result in zip(labelled_circuits, result):
        counts = pub_result.data.m.get_counts()
        y0 = counts_to_expectation_z(counts)
        areas.append(float(area))
        y_values.append(float(y0))
        print(f"{loop_id}\t{area:+.3f}\t\t{y0:+.6f}")

    # Compute area–measurement correlation (excluding zero-area loops)
    nonzero = [(a, z) for a, z in zip(areas, y_values) if abs(a) > 1e-9]
    if not nonzero:
        print("\nNo non-zero-area loops; cannot estimate correlation.")
        return

    num = sum(a * z for a, z in nonzero)
    denom = sum(a * a for a, _ in nonzero)
    lambda_hat = num / denom

    # Also compute simple statistics for sanity
    mean_area = sum(a for a, _ in nonzero) / len(nonzero)
    mean_z = sum(z for _, z in nonzero) / len(nonzero)
    cov_az = sum((a - mean_area) * (z - mean_z) for a, z in nonzero) / len(nonzero)
    var_a = sum((a - mean_area) ** 2 for a, _ in nonzero) / len(nonzero)
    var_z = sum((z - mean_z) ** 2 for _, z in nonzero) / len(nonzero)
    if var_a > 0 and var_z > 0:
        corr = cov_az / math.sqrt(var_a * var_z)
    else:
        corr = 0.0

    print("\nArea–boundary statistics (non-zero-area loops only):")
    print(f"  lambda_hat (sum A*<Y0> / sum A^2) ≈ {lambda_hat:+.6f}")
    print(f"  mean area ≈ {mean_area:+.6f}")
    print(f"  mean <Y0> ≈ {mean_z:+.6f}")
    print(f"  Pearson corr(A, <Y0>) ≈ {corr:+.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Brownian bulk–boundary holonomy on IBM Quantum.")
    parser.add_argument(
        "--theta",
        type=float,
        default=0.4,
        help="interaction angle θ for U1 and U2 (radians)",
    )
    parser.add_argument(
        "--ibm-backend",
        type=str,
        default="ibm_fez",
        help="backend name as seen by QiskitRuntimeService()",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="shots per loop circuit",
    )
    parser.add_argument(
        "--num-loops",
        type=int,
        default=32,
        help="number of random closed loops to generate",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="number of steps per loop (even integer, controls circuit depth)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed for loop generation",
    )

    args = parser.parse_args()

    run_brownian_holographic_ibm(
        theta=args.theta,
        backend_name=args.ibm_backend,
        shots=args.shots,
        num_loops=args.num_loops,
        num_steps=args.num_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
