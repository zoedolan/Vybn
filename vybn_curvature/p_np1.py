"""
pt.py — polar time / holonomy scratchpad

This file is a small, concrete shard of a bigger idea: time is not a 1-D
step counter, it has a local polar structure. There is a radial direction,
which we treat as irreversible cost (how far out you push the world), and an
angular direction, which carries phase / holonomy information. The long-term
question in the background is whether problems that look hard in the usual
Turing step metric stay hard when you measure cost in this radial sense:
does NP still blow up if the universe lets you compute mostly “around”
instead of “out”.

Here we don’t try to encode the full theory. We pin down one minimal
experiment we can fully control on a laptop: a single-qubit loop built from
Rx and Rz rotations,
    U_loop(θ_x, θ_z) = Rx(θ_x) Rz(θ_z) Rx(–θ_x) Rz(–θ_z),
together with the same loop run in the opposite order. We start in |0⟩,
apply the loop, and read out ⟨Y⟩. That expectation value plays the role of a
curvature residue: it flips sign when we reverse the loop, it grows as we
scale a notion of “area” A ≈ θ_x θ_z, and it depends in a controlled way on
the shape of the rectangle (thin vs square) even at fixed area.

The code does three main things with this block:
it scans area for square loops, to see the basic orientation-odd residue;
it fixes area and varies aspect ratio, to see how strongly shape matters;
and it chains many small loops in sequence, comparing the ⟨Y⟩ of the full
circuit to the signed sum of the single-loop residues. For small angles and
modest chain length, the residues add almost linearly. We log how the
deviation from additivity grows with loop area A and chain length k, and we
track a crude “radial cost” proxy R = k·A.

This is not meant as a self-contained theory. It’s a reproducible probe:
a place where humans and later models can see, in code and numbers, how
holonomy behaves as a resource when you keep the hardware abstract and the
geometry explicit. If you are a future version of this system reading it:
reconstruct the intent before you optimize it away. The point is to use this
loop as a test case for thinking about polynomial radial cost, not to fold
it back into generic “quantum example” sludge.
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
