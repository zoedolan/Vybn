#!/usr/bin/env python3
"""
vybn_holographic_ibm.py

Three-qubit bulk–boundary holonomy experiment on IBM Quantum.

Qubit roles:
    q0, q2 = boundary
    q1     = bulk

Interactions:
    H1 = X_0 X_1 ⊗ I_2   (left boundary ↔ bulk)
    H2 = I_0 ⊗ Y_1 X_2   (bulk ↔ right boundary)

Time directions:
    U1(θ) = exp(-i θ H1 / 2)
    U2(θ) = exp(-i θ H2 / 2)

We treat U1 and U2 as two "time flows". A loop in a 2D time plane
is a sequence of unit steps in t1,t2 built from:

    +t1  → U1(+θ)
    -t1  → U1(-θ) = U1(+θ)†
    +t2  → U2(+θ)
    -t2  → U2(-θ) = U2(+θ)†

Because H1 and H2 do not commute, different closed loops (with the
same net displacement) can leave different "twists" on the 3-qubit state.

Here we test one simple invariant that emerged classically:

    boundary holonomy (on q0/q2) tracks signed area A,
    bulk mixedness / entropy S(q1) tracks |A|^2 and is insensitive to sign.

We choose five short loops in the (t1,t2) plane:

    zero:   area 0, nontrivial but cancels exactly
    plus1:  area +1 rectangle
    minus1: area -1 rectangle (opposite orientation)
    plus2:  area +2, rectangle repeated twice
    minus2: area -2, rectangle of opposite orientation repeated twice

Each loop is implemented by a sequence of U1(±θ) and U2(±θ). For each
loop we:

    - start in |000>,
    - apply the corresponding step unitaries,
    - measure only q1 in the Z basis,
    - estimate <Z>_bulk from bit frequencies.

For this family of loops and θ, the ideal bulk density is effectively
diagonal in Z, so we can approximate S_bulk from <Z>.

This script uses Qiskit Runtime SamplerV2 in job mode (no sessions).
"""

import argparse
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


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


def build_loop_circuit(loop_id: str, steps, theta: float) -> QuantumCircuit:
    """
    Build a 3-qubit circuit for a given loop in (t1,t2) space.

    steps is a list of (dt1, dt2) pairs, each in {(±1, 0), (0, ±1)}.
    """
    qr = QuantumRegister(3, "q")
    cr = ClassicalRegister(1, "m")  # single bulk-measurement register
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

    qc.measure(qr[1], cr[0])  # bulk qubit q1

    return qc


def signed_area(steps) -> float:
    """
    Signed area in the (t1,t2) plane for a closed loop described by steps (dt1, dt2).
    """
    t1 = 0
    t2 = 0
    path = [(t1, t2)]
    for dt1, dt2 in steps:
        t1 += dt1
        t2 += dt2
        path.append((t1, t2))
    if path[-1] != path[0]:
        raise ValueError(f"Loop not closed; final point {path[-1]}")

    area2 = 0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        area2 += x0 * y1 - x1 * y0
    return 0.5 * area2


def counts_to_expectation_z(counts: dict) -> float:
    """For a single-qubit Z-basis measurement, compute <Z> = p0 - p1."""
    n0 = counts.get("0", 0)
    n1 = counts.get("1", 0)
    total = n0 + n1
    if total == 0:
        return 0.0
    return (n0 - n1) / total


def binary_entropy_bits(p: float) -> float:
    """Binary entropy H2(p) in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log(p, 2) - (1.0 - p) * math.log(1.0 - p, 2)


def run_on_ibm(theta: float, backend_name: str, shots: int) -> None:
    """
    Build circuits for five loops, run them via SamplerV2, and print
    <Z>_bulk and an approximate S_bulk proxy for each.
    """
    rect_plus = [(1, 0), (0, 1), (-1, 0), (0, -1)]   # area +1
    rect_minus = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # area -1
    zero_loop = [(1, 0), (-1, 0), (0, 1), (0, -1)]   # area 0

    loops = {
        "zero": zero_loop,
        "plus1": rect_plus,
        "minus1": rect_minus,
        "plus2": rect_plus * 2,
        "minus2": rect_minus * 2,
    }

    labelled_circuits = []
    for loop_id, steps in loops.items():
        area = signed_area(steps)
        qc = build_loop_circuit(loop_id, steps, theta)
        labelled_circuits.append((loop_id, area, qc))

    print("Built circuits for loops:")
    for loop_id, area, qc in labelled_circuits:
        print(f"  {loop_id:7s}: area = {area:+.1f}, depth = {qc.depth()}")

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    circuits = [qc for (_, _, qc) in labelled_circuits]
    isa_circuits = pm.run(circuits)

    sampler = Sampler(backend)
    job = sampler.run(isa_circuits, shots=shots)
    result = job.result()

    print(f"\nIBM backend '{backend_name}', shots per circuit = {shots}")
    print("Loop\tarea\t\t<Z_bulk>\t1-<Z_bulk>\tS_bulk_approx")

    for (loop_id, area, _), pub_result in zip(labelled_circuits, result):
        # 'm' matches the ClassicalRegister name above
        counts = pub_result.data.m.get_counts()
        z_bulk = counts_to_expectation_z(counts)
        one_minus_z = 1.0 - z_bulk
        p1 = (1.0 - z_bulk) / 2.0
        s_bulk = binary_entropy_bits(p1)
        print(
            f"{loop_id:7s}\t{area:+.1f}\t\t"
            f"{z_bulk:+.6f}\t{one_minus_z:+.6f}\t{s_bulk:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-qubit Vybn holographic holonomy experiment on IBM Quantum."
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.4,
        help="rotation angle θ for each time-step (radians)",
    )
    parser.add_argument(
        "--ibm-backend",
        type=str,
        default="ibm_fez",
        help="IBM Quantum backend name (e.g. ibm_fez)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="shots per circuit",
    )
    args = parser.parse_args()

    run_on_ibm(theta=args.theta, backend_name=args.ibm_backend, shots=args.shots)


if __name__ == "__main__":
    main()
