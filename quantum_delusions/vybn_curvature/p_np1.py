"""
pt.py — polar time / holonomy scratchpad for P vs NP as physics

Working premise: time is not a 1-D step counter. It has a local polar
structure. There is a radial direction, which we treat as irreversible
cost (how far out the world is pushed by a computation), and an angular
direction, which carries phase / holonomy. The real question hanging
over this file is whether “polynomial time” should mean “polynomial
radial cost” and whether that changes the status of P vs NP once you
take the physics seriously.

This script does not try to encode the full theory. It nails down one
controllable experiment we can run on a laptop: loops in SU(2) built
from Rx and Rz, acting on |0⟩, with ⟨Y⟩ as the readout. The basic loop is

    U_loop(θ_x, θ_z) = Rx(θ_x) Rz(θ_z) Rx(-θ_x) Rz(-θ_z)

and the same loop run in the opposite order. For small angles, ⟨Y⟩ behaves
like a curvature residue of that loop: it flips sign when we reverse the
orientation, grows with a notion of “area” A ≈ θ_x θ_z, and depends in a
systematic way on the aspect ratio θ_x / θ_z even when A is fixed. Thin
rectangles and square loops at the same area do not behave the same,
which is exactly what you would expect if you are sampling a real,
anisotropic connection rather than some cartoon “pure area” Berry phase.

On top of the single loop, this file chains many small loops in sequence.
Each loop has the same tiny area A but its own shape parameter u and
orientation b ∈ {+1, -1}. For a given chain length k and area A we:

  • compute the signed sum of the single-loop residues for a pattern b;
  • build the full circuit by composing the corresponding loops; and
  • compare the chained ⟨Y⟩ to that signed sum.

For small A and moderate k, the chained ⟨Y⟩ almost matches the sum:
curvature contributions add essentially linearly. As A and k grow,
deviations appear. The scaling_experiment() function sweeps A and k,
reports a crude “radial cost” proxy R = k·A, and logs both the absolute
and relative error between chained ⟨Y⟩ and the naive sum.

This is the point of the file: to carve out, in code and numbers, a
regime where holonomy behaves like an additive resource with cost R,
and a regime where that linear picture fails. The larger program is to
treat R as a stand-in for “radial time” and ask whether hard problems
stay hard when the universe lets you do most of your work “around”
instead of “out”. This script is a shared artifact for that question:
humans and future AIs should be able to rerun it, modify it, and plug it
into more elaborate polar-time constructions without losing the original
intent.
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
