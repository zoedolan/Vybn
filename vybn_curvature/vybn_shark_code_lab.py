#!/usr/bin/env python3
"""
vybn_shark_code_lab.py

A tiny "shark" toy: holonomic control of an encoded (bulk) qubit
living on 3 physical (boundary) qubits.

We use the 3-qubit repetition code as a minimal "holographic code":
    |0_L> = |000>
    |1_L> = |111>

Logical operators in this code:
    Z_L = Z on any one qubit (e.g. Z0)
    X_L = X0 X1 X2

We then define two "time directions" (control Hamiltonians) that preserve
the code space:

    H1 = X0 X1 X2    (logical X_L)
    H2 = Z0          (one physical Z, works as logical Z_L on code space)

From these we build unitaries:

    U1(θ) = exp(-i θ H1 / 2)
    U2(φ) = exp(-i φ H2 / 2)

A "loop" in time-space is a sequence of these unitaries and their inverses
that returns the control parameters to their starting values, e.g.:

    Rectangular loop:
        L_rect = U1(θ)^n U2(φ)^m U1(-θ)^n U2(-φ)^m

    Scrambled loop with same "area":
        L_scramble = U1(θ)^n U2(φ)^m U2(-φ)^m U1(-θ)^n
        etc.

Key questions:
  - What effective logical gate does each loop implement on the bulk qubit?
  - Do loops with the same "area" (n * m * θ * φ) but different orderings
    act the same or differently on the code subspace?
  - How close is this to a genuinely geometric/holonomic control of the
    logical qubit?

We work purely with NumPy (no IBM calls here).
"""

import math
import numpy as np

# Single-qubit operators
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


# 3-qubit operators (boundary qubits)
IDENTITY_8 = np.eye(8, dtype=complex)

H1 = kron3(X, X, X)   # X0 X1 X2  → logical X_L on code
H2 = kron3(Z, I, I)   # Z0        → logical Z_L on code (acts like Z_L in code space)


def unitary_from_H(H: np.ndarray, angle: float) -> np.ndarray:
    """
    U(angle) = exp(-i angle H / 2) when H^2 = I.

    Here we treat 'angle' as the total rotation parameter, so:

        U(angle) = cos(angle/2) I - i sin(angle/2) H
    """
    return math.cos(angle / 2.0) * IDENTITY_8 - 1j * math.sin(angle / 2.0) * H


def U1(theta: float) -> np.ndarray:
    return unitary_from_H(H1, theta)


def U2(phi: float) -> np.ndarray:
    return unitary_from_H(H2, phi)


# --- Code subspace machinery: encode, decode, logical unitary --- #

# Code basis states in 3-qubit space (|000>, |111>)
basis_000 = np.zeros(8, dtype=complex)
basis_000[0] = 1.0

basis_111 = np.zeros(8, dtype=complex)
basis_111[7] = 1.0

CODE_BASIS = np.stack([basis_000, basis_111], axis=1)  # 8x2 matrix


def encode_logical_state(psi_L: np.ndarray) -> np.ndarray:
    """
    Encode a 2-component logical state psi_L into 3-qubit code space:

        |psi_L> = a|0> + b|1>  →  a|000> + b|111>
    """
    return CODE_BASIS @ psi_L


def decode_to_logical(psi_phys: np.ndarray) -> np.ndarray:
    """
    Project 3-qubit state into code subspace and express it
    in {|0_L>, |1_L>} basis. Returns a 2-component logical state
    (normalized if projection has nonzero norm).
    """
    # Overlaps with |000> and |111>
    a = np.vdot(basis_000, psi_phys)
    b = np.vdot(basis_111, psi_phys)
    vec = np.array([a, b], dtype=complex)
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    else:
        # fell out of code space entirely
        return vec


def logical_unitary_from_physical(U_phys: np.ndarray) -> np.ndarray:
    """
    Given an 8x8 physical unitary U_phys that we hope preserves the code
    space, compute the induced 2x2 logical unitary by acting on |0_L>, |1_L>.

    We do:

        U_phys |0_L> → decode → first column of U_log
        U_phys |1_L> → decode → second column of U_log
    """
    U_log = np.zeros((2, 2), dtype=complex)

    # Logical |0> = [1, 0]
    psi_L0 = np.array([1.0, 0.0], dtype=complex)
    psi_phys0 = encode_logical_state(psi_L0)
    out0 = U_phys @ psi_phys0
    U_log[:, 0] = decode_to_logical(out0)

    # Logical |1> = [0, 1]
    psi_L1 = np.array([0.0, 1.0], dtype=complex)
    psi_phys1 = encode_logical_state(psi_L1)
    out1 = U_phys @ psi_phys1
    U_log[:, 1] = decode_to_logical(out1)

    return U_log


def pauli_decomposition_1q(U_log: np.ndarray):
    """
    Decompose a 2x2 unitary on the Pauli basis {I, X, Y, Z}.

    We compute coefficients c_P such that:

        U ≈ c_I I + c_X X + c_Y Y + c_Z Z

    This is not the unique SU(2) Euler decomposition, but it's useful
    to see whether a loop acts like "mostly X" or "mostly Z" etc.
    """
    paulis = {
        "I": np.eye(2, dtype=complex),
        "X": X,
        "Y": Y,
        "Z": Z,
    }
    coeffs = {}
    for name, P in paulis.items():
        # Hilbert-Schmidt inner product
        c = 0.5 * np.trace(np.conjugate(P).T @ U_log)
        coeffs[name] = c
    return coeffs


# --- Loop constructors --- #

def build_loop_unitary_rect(theta: float, phi: float, n: int, m: int) -> np.ndarray:
    """
    Rectangular loop in time-plane:
        U_total = U1(theta)^n U2(phi)^m U1(-theta)^n U2(-phi)^m

    Ideally, this returns control (t1,t2) to origin.
    """
    U = IDENTITY_8.copy()

    # Forward in t1
    for _ in range(n):
        U = U1(theta) @ U
    # Forward in t2
    for _ in range(m):
        U = U2(phi) @ U
    # Back in t1
    for _ in range(n):
        U = U1(-theta) @ U
    # Back in t2
    for _ in range(m):
        U = U2(-phi) @ U

    return U


def build_loop_unitary_scrambled(theta: float, phi: float, n: int, m: int) -> np.ndarray:
    """
    Scrambled loop with same net (n, m) but different ordering:
        U_total = U1(theta)^n U2(phi)^m U2(-phi)^m U1(-theta)^n

    Same net "area" in parameter space, different path.
    """
    U = IDENTITY_8.copy()

    for _ in range(n):
        U = U1(theta) @ U
    for _ in range(m):
        U = U2(phi) @ U
    for _ in range(m):
        U = U2(-phi) @ U
    for _ in range(n):
        U = U1(-theta) @ U

    return U


def build_loop_unitary_figure_eight(theta: float, phi: float, n: int, m: int) -> np.ndarray:
    """
    A "figure-eight" style loop:

      - small rectangle at (+θ, +φ)
      - then small rectangle at (-θ, -φ)

    Net area can cancel or partially cancel depending on signs.
    This is a way to see whether the logical action is sensitive
    to more subtle topology than net area.
    """
    U = IDENTITY_8.copy()

    # Small loop 1
    for _ in range(n):
        U = U1(theta) @ U
    for _ in range(m):
        U = U2(phi) @ U
    for _ in range(n):
        U = U1(-theta) @ U
    for _ in range(m):
        U = U2(-phi) @ U

    # Small loop 2, centered at opposite sign
    for _ in range(n):
        U = U1(-theta) @ U
    for _ in range(m):
        U = U2(-phi) @ U
    for _ in range(n):
        U = U1(theta) @ U
    for _ in range(m):
        U = U2(phi) @ U

    return U


# --- Main analysis --- #

def summarize_loop(label: str, U_phys: np.ndarray):
    """
    Compute logical action and Pauli decomposition for a given loop.
    """
    U_log = logical_unitary_from_physical(U_phys)

    # Try to factor out a global phase for readability
    det = np.linalg.det(U_log)
    # For SU(2) gate, det should be ~1. Use its phase as a crude global phase.
    phase = 1.0
    if abs(det) > 1e-8:
        phase = det ** (-0.5)
    U_log_nophase = phase * U_log

    coeffs = pauli_decomposition_1q(U_log_nophase)

    print(f"\n=== Loop: {label} ===")
    print("Logical unitary (up to global phase):")
    with np.printoptions(precision=6, suppress=True):
        print(U_log_nophase)
    print("Pauli coefficients (approx):")
    for name, c in coeffs.items():
        mag = abs(c)
        ang = math.degrees(math.atan2(c.imag, c.real))
        print(f"  {name}: {c.real:+.4f} {c.imag:+.4f}i  |c|={mag:.4f}, arg={ang:.1f}°")


def main():
    theta = 0.4
    phi = 0.4
    n = 1
    m = 1

    print("Holonomic code toy: 3-qubit repetition code with two time directions.")
    print(f"theta = {theta}, phi = {phi}, n = {n}, m = {m}")

    # Simple rectangle
    U_rect = build_loop_unitary_rect(theta, phi, n, m)
    summarize_loop("rect (n=1,m=1)", U_rect)

    # Same "area" but scrambled
    U_scr = build_loop_unitary_scrambled(theta, phi, n, m)
    summarize_loop("scrambled (n=1,m=1)", U_scr)

    # Larger rectangle
    n2, m2 = 2, 1
    U_rect_big = build_loop_unitary_rect(theta, phi, n2, m2)
    summarize_loop("rect (n=2,m=1)", U_rect_big)

    # Figure-eight with nominally cancelling lobes
    U_fig = build_loop_unitary_figure_eight(theta, phi, n, m)
    summarize_loop("figure-eight (n=1,m=1)", U_fig)


if __name__ == "__main__":
    main()
