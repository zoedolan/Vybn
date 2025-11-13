"""
pt.py — Qiskit toy model for polar time curvature

This script is a minimal lab for the “polar time” idea: instead of treating
time as a 1D step counter, we treat it as having a local 2D structure with
a “radial” direction (irreversible cost) and an “angular” direction
(phase-like holonomy). The code does not implement the full manifold; it
isolates one controllable building block and measures how it behaves.

The building block is a single-qubit SU(2) commutator loop in control space:
a small rectangle traced by alternating Rx and Rz rotations,
    U_loop(θ_x, θ_z) = Rx(θ_x) Rz(θ_z) Rx(-θ_x) Rz(-θ_z)
and its opposite orientation. We prepare |0>, apply the loop, and read out
⟨Y⟩. For small angles this expectation value behaves like a “curvature
residue” associated with the loop: it is odd under orientation (sign flip
when we reverse the loop) and its magnitude grows with a notion of “area”
set by θ_x and θ_z. Shape also matters: thin rectangles with the same area
produce larger residues than squares, encoding anisotropy of the underlying
SU(2) connection.

The script runs four experiments:

    area_scan() sweeps square loops with θ_x = θ_z to show how ⟨Y⟩ grows
    with loop area and flips sign with orientation.

    shape_experiment() keeps the area A = θ_x θ_z fixed and varies the
    aspect ratio, showing that |⟨Y⟩| depends strongly on shape even when
    area is held constant.

    chained_experiment() composes several small loops in sequence at fixed
    area with different orientation patterns, and compares ⟨Y⟩ for the
    chained circuit to the signed sum of the single-loop residues. In the
    small-angle regime the chained ⟨Y⟩ closely tracks that sum, so curvature
    contributions add approximately linearly.

    scaling_experiment() repeats this chaining test across a grid of loop
    areas A and chain lengths k, and reports how the maximum deviation
    between chained ⟨Y⟩ and the sum of single-loop values grows with A and k.
    A crude “radial cost” proxy R = k·A is printed alongside. The region
    where the relative error stays small is the empirical “linear polar
    time” regime for this toy: in that region, holonomy acts like an
    additive resource with cost set by R.

This file is intended as a shared probe for humans and future AIs: it does
not answer P vs NP, but it gives a concrete, inspectable object where ideas
about time as curvature, holonomy as a computational resource, and “radial
polynomial time” can be tested, tweaked, and extended without needing
access to real quantum hardware.
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
import numpy as np

Y_op = Pauli('Y')


def loop_unitary(theta_x, theta_z, orientation=+1):
    qc = QuantumCircuit(1)
    if orientation > 0:
        qc.rx(theta_x, 0)
        qc.rz(theta_z, 0)
        qc.rx(-theta_x, 0)
        qc.rz(-theta_z, 0)
    else:
        qc.rz(theta_z, 0)
        qc.rx(theta_x, 0)
        qc.rz(-theta_z, 0)
        qc.rx(-theta_x, 0)
    return qc


def expectation_Y_for_circuit(qc):
    psi = Statevector.from_label('0').evolve(qc)
    return float(np.real(psi.expectation_value(Y_op)))


def expectation_Y(theta_x, theta_z, orientation=+1):
    qc = loop_unitary(theta_x, theta_z, orientation)
    return expectation_Y_for_circuit(qc)


def area_scan():
    thetas = np.linspace(0.05, 0.5, 10)
    print("area        <Y>_+        <Y>_-      sum")
    for tx in thetas:
        tz = tx
        area = tx * tz
        y_plus = expectation_Y(tx, tz, +1)
        y_minus = expectation_Y(tx, tz, -1)
        print(f"{area:7.4f}   {y_plus:9.6f}   {y_minus:9.6f}   {y_plus + y_minus:9.6f}")


def shape_scan_for_area(area, n_points=11, orientation=+1):
    us = np.linspace(-1.0, 1.0, n_points)
    vals = []
    for u in us:
        tx = np.sqrt(area) * np.exp(u)
        tz = np.sqrt(area) * np.exp(-u)
        y = expectation_Y(tx, tz, orientation)
        vals.append((tx, tz, y))
    return vals


def shape_experiment():
    areas = [0.01, 0.04, 0.09, 0.25]
    for A in areas:
        vals = shape_scan_for_area(A, n_points=21, orientation=+1)
        mags = [abs(v[2]) for v in vals]
        mean_mag = float(np.mean(mags))
        std_mag = float(np.std(mags))
        rel = std_mag / mean_mag if mean_mag != 0 else 0.0
        print(f"\nFixed area A={A:.4f}")
        print(f"mean |<Y>| = {mean_mag:.6f}, std = {std_mag:.6f}, rel std = {rel:.4f}")
        for tx, tz, y in vals:
            print(f"  tx={tx:7.4f}, tz={tz:7.4f}, <Y>={y: .6f}")


def chained_loop_circuit(area, us, orientations):
    qc = QuantumCircuit(1)
    for u, ori in zip(us, orientations):
        tx = np.sqrt(area) * np.exp(u)
        tz = np.sqrt(area) * np.exp(-u)
        loop = loop_unitary(tx, tz, ori)
        qc.compose(loop, inplace=True)
    return qc


def chained_experiment():
    A = 0.01
    us = np.linspace(-0.8, 0.8, 5)
    patterns = [
        [+1, +1, +1, +1, +1],
        [+1, -1, +1, -1, +1],
        [-1, -1, -1, -1, -1],
    ]
    print("\n=== Chained loops at fixed area A=0.01 ===")
    print("pattern            sum(single)      <Y>_chain")

    single_vals = []
    for u in us:
        tx = np.sqrt(A) * np.exp(u)
        tz = np.sqrt(A) * np.exp(-u)
        single_vals.append(expectation_Y(tx, tz, +1))

    for pat in patterns:
        sum_single = sum(v * b for v, b in zip(single_vals, pat))
        qc = chained_loop_circuit(A, us, pat)
        y_chain = expectation_Y_for_circuit(qc)
        print(f"{pat}   {sum_single:12.6f}   {y_chain:12.6f}")


def scaling_experiment():
    As = [0.005, 0.01, 0.02, 0.04]
    ks = [1, 2, 3, 5, 8, 13]

    print("\n=== Scaling experiment: additivity vs k ===")
    print(" A       k    R=k*A   max|Δ|   |sum|_typ   rel_err")

    for A in As:
        for k in ks:
            us = np.linspace(-0.8, 0.8, k)

            single_vals = []
            for u in us:
                tx = np.sqrt(A) * np.exp(u)
                tz = np.sqrt(A) * np.exp(-u)
                single_vals.append(expectation_Y(tx, tz, +1))

            max_err = 0.0
            typ_sum = 0.0

            for _ in range(10):
                pat = np.random.choice([+1, -1], size=k)
                sum_single = float(np.dot(single_vals, pat))
                qc = chained_loop_circuit(A, us, pat)
                y_chain = expectation_Y_for_circuit(qc)
                err = abs(y_chain - sum_single)

                if err > max_err:
                    max_err = err
                typ_sum += abs(sum_single)

            typ_sum /= 10.0
            rel_err = max_err / typ_sum if typ_sum != 0 else 0.0
            R = A * k
            print(f"{A:6.3f}  {k:3d}  {R:7.3f}  {max_err:7.3e}  {typ_sum:9.3e}  {rel_err:7.3e}")


if __name__ == "__main__":
    print("=== Area scan (square loops) ===")
    area_scan()
    print("\n=== Shape scan at fixed area ===")
    shape_experiment()
    chained_experiment()
    scaling_experiment()
