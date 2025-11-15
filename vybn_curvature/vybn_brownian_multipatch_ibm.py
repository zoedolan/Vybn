#!/usr/bin/env python3
"""
vybn_brownian_multipatch_ibm.py

Multi-patch Brownian-time / holographic experiment on IBM Quantum.

We tile the chip with disjoint 3-qubit patches:

    patch p uses qubits (3p, 3p+1, 3p+2)

Within each patch:
    q0, q2 = boundary (measured in Y basis)
    q1     = bulk     (measured in Z basis)

We apply the SAME Brownian-time loop in each patch in parallel, then
measure all qubits. This lets us see, in one job:

  - boundary <Y> vs loop orientation (plus vs minus) across patches
  - bulk <Z> and S_bulk vs |area| and "texture" across patches
  - whether these behaviors are spatially universal or patch-dependent

Loops in the (t1, t2) plane:

  id          : no loop (baseline)
  zero_short  : small area-0 loop
  zero_long   : longer area-0 loop (more steps / "texture")
  plus1       : area +1 rectangle
  minus1      : area -1 rectangle (opposite orientation)
  plus2       : two copies of plus1 (area +2)
  minus2      : two copies of minus1 (area -2)
  fig8        : "figure-eight" style area-0 loop with more structure

Each loop is described by a list of (dt1, dt2) steps with dt1,dt2 in {±1,0}
and |(dt1,dt2)| = 1, and turned into U1/U2 applications as in your earlier
holographic scripts.

Run:

  python vybn_brownian_multipatch_ibm.py --ibm-backend ibm_fez --shots 4096 --patches 8

Adjust --patches if you want fewer/more 3-qubit universes in parallel.
"""

import argparse
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# ----- Single-patch U1 / U2 (same structure as before) ----- #

def apply_u1_on_patch(qc: QuantumCircuit, theta: float, q0: int, q1: int) -> None:
    """
    Apply U1(theta) = exp(-i theta X_q0 X_q1 / 2) on a given patch.

    Decomposition:

      X⊗X = (H⊗H) Z⊗Z (H⊗H)
      exp(-i θ Z⊗Z / 2) = CX RZ(θ) CX
    """
    qc.h(q0)
    qc.h(q1)

    qc.cx(q0, q1)
    qc.rz(theta, q1)
    qc.cx(q0, q1)

    qc.h(q0)
    qc.h(q1)


def apply_u2_on_patch(qc: QuantumCircuit, theta: float, q1: int, q2: int) -> None:
    """
    Apply U2(theta) = exp(-i theta Y_q1 X_q2 / 2) on a given patch.

    As before, choose V1,V2 such that:
      V1 Z V1† = Y  (V1 = Rx(-π/2))
      V2 Z V2† = X  (V2 = H)

    Then:
      Y⊗X = (V1⊗V2) Z⊗Z (V1⊗V2)†
    """
    qc.rx(math.pi / 2, q1)
    qc.h(q2)

    qc.cx(q1, q2)
    qc.rz(theta, q2)
    qc.cx(q1, q2)

    qc.rx(-math.pi / 2, q1)
    qc.h(q2)


def apply_loop_on_patch(qc: QuantumCircuit, steps, theta: float, patch_index: int) -> None:
    """
    Apply a given (dt1,dt2) loop to patch 'patch_index'.

    patch p uses qubits:
        q0 = 3p, q1 = 3p+1, q2 = 3p+2
    """
    q0 = 3 * patch_index
    q1 = q0 + 1
    q2 = q0 + 2

    for dt1, dt2 in steps:
        if dt1 == 1 and dt2 == 0:
            apply_u1_on_patch(qc, theta, q0, q1)
        elif dt1 == -1 and dt2 == 0:
            apply_u1_on_patch(qc, -theta, q0, q1)
        elif dt1 == 0 and dt2 == 1:
            apply_u2_on_patch(qc, theta, q1, q2)
        elif dt1 == 0 and dt2 == -1:
            apply_u2_on_patch(qc, -theta, q1, q2)
        else:
            raise ValueError(f"Non-unit step encountered: {(dt1, dt2)}")


def rotate_patches_to_measure(qc: QuantumCircuit, num_patches: int) -> None:
    """
    For each patch:
      - rotate boundary qubits (q0,q2) into Y basis: S† H
      - leave bulk qubit (q1) in Z basis
    """
    for p in range(num_patches):
        q0 = 3 * p
        q1 = q0 + 1
        q2 = q0 + 2
        qc.sdg(q0)
        qc.h(q0)
        qc.sdg(q2)
        qc.h(q2)
        # q1 untouched (Z basis)


def signed_area_from_steps(steps) -> float:
    """
    Compute signed area in (t1,t2) plane from dt1,dt2 steps.
    """
    t1 = 0
    t2 = 0
    path = [(t1, t2)]
    for dt1, dt2 in steps:
        t1 += dt1
        t2 += dt2
        path.append((t1, t2))
    if path[-1] != path[0]:
        raise ValueError(f"Loop not closed; final point = {path[-1]}")

    area2 = 0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        area2 += x0 * y1 - x1 * y0
    return 0.5 * area2


def counts_to_patch_expectations(counts: dict, num_patches: int):
    """
    Given counts over all 3*num_patches bits (one ClassicalRegister),
    compute <Y0>, <Z1>, <Y2> for each patch.

    Returns a list of length num_patches:
        [(Ey0, Ez1, Ey2), ...]
    """
    totals = [0] * num_patches
    n0_0 = [0] * num_patches
    n1_0 = [0] * num_patches
    n0_1 = [0] * num_patches
    n1_1 = [0] * num_patches
    n0_2 = [0] * num_patches
    n1_2 = [0] * num_patches

    # total bits = 3 * num_patches
    num_bits = 3 * num_patches

    for bitstr, n in counts.items():
        # bitstr[0] is msb; bitstr[-1] is lsb (c[0])
        if len(bitstr) != num_bits:
            # paddings or leading zeros are possible; pad on the left
            bitstr = bitstr.zfill(num_bits)
        # loop patches
        for p in range(num_patches):
            # classical bit indices for this patch: 3p, 3p+1, 3p+2
            b0_idx = 3 * p      # bulk bit indices in "lsb is rightmost" sense
            b1_idx = 3 * p + 1
            b2_idx = 3 * p + 2

            # convert to positions in the string
            # bit index i (0-based, lsb) is at position -1 - i
            s0 = bitstr[-1 - b0_idx]
            s1 = bitstr[-1 - b1_idx]
            s2 = bitstr[-1 - b2_idx]

            b0 = int(s0)  # q0
            b1 = int(s1)  # q1
            b2 = int(s2)  # q2

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

    results = []
    for p in range(num_patches):
        if totals[p] == 0:
            results.append((0.0, 0.0, 0.0))
            continue

        Ey0 = (n0_0[p] - n1_0[p]) / totals[p]
        Ez1 = (n0_1[p] - n1_1[p]) / totals[p]
        Ey2 = (n0_2[p] - n1_2[p]) / totals[p]
        results.append((Ey0, Ez1, Ey2))

    return results


def binary_entropy_bits(p: float) -> float:
    """
    Binary entropy H2(p) in bits.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log(p, 2) - (1.0 - p) * math.log(1.0 - p, 2)


def run_on_ibm(theta: float, backend_name: str, shots: int, num_patches: int) -> None:
    """
    Build and run multipatch Brownian/holographic experiment on IBM Quantum.
    """

    # elementary loops
    rect_plus = [(1, 0), (0, 1), (-1, 0), (0, -1)]   # area +1
    rect_minus = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # area -1
    zero_short = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # area 0
    zero_long = rect_plus + rect_minus               # area 0, more texture
    # figure-eight: two +1 lobes and two -1 lobes in a sequence (area 0, high texture)
    fig8 = rect_plus + rect_plus + rect_minus + rect_minus

    loops = {
        "id":         [],
        "zero_short": zero_short,
        "zero_long":  zero_long,
        "plus1":      rect_plus,
        "minus1":     rect_minus,
        "plus2":      rect_plus * 2,
        "minus2":     rect_minus * 2,
        "fig8":       fig8,
    }

    labelled_circuits = []

    num_qubits = 3 * num_patches
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "m")

    for loop_id, steps in loops.items():
        qc = QuantumCircuit(qr, cr, name=f"loop_{loop_id}")
        # apply same loop to each patch
        for p in range(num_patches):
            apply_loop_on_patch(qc, steps, theta, p)
        rotate_patches_to_measure(qc, num_patches)
        qc.measure(qr, cr)

        if steps:
            area = signed_area_from_steps(steps)
        else:
            area = 0.0

        labelled_circuits.append((loop_id, area, qc))

    print(f"Building circuits for {num_patches} patches "
          f"({3*num_patches} qubits total)")
    for loop_id, area, qc in labelled_circuits:
        print(f"  {loop_id:10s}: area = {area:+.1f}, depth ≈ {qc.depth()}")

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    circuits = [qc for (_, _, qc) in labelled_circuits]
    isa_circuits = pm.run(circuits)

    sampler = Sampler(backend)
    job = sampler.run(isa_circuits, shots=shots)
    result = job.result()

    print(f"\nIBM backend '{backend_name}', shots per circuit = {shots}")
    print("loop_id    area   patch  <Y0>\t\t<Z1>\t\t<Y2>\t\tS_bulk_approx")

    for (loop_id, area, _), pub_result in zip(labelled_circuits, result):
        counts = pub_result.data.m.get_counts()
        patch_vals = counts_to_patch_expectations(counts, num_patches)

        for p, (Ey0, Ez1, Ey2) in enumerate(patch_vals):
            p1 = (1.0 - Ez1) / 2.0
            S_bulk = binary_entropy_bits(p1)
            print(
                f"{loop_id:10s} {area:+.1f}   {p:2d}   "
                f"{Ey0:+.6f}\t{Ez1:+.6f}\t{Ey2:+.6f}\t{S_bulk:.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multipatch Brownian/holographic experiment on IBM Quantum."
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
    parser.add_argument(
        "--patches",
        type=int,
        default=8,
        help="number of 3-qubit patches to tile (uses 3*patches qubits)",
    )
    args = parser.parse_args()

    run_on_ibm(
        theta=args.theta,
        backend_name=args.ibm_backend,
        shots=args.shots,
        num_patches=args.patches,
    )


if __name__ == "__main__":
    main()
