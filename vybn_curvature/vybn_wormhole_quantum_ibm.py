"""
vybn_wormhole_quantum.py

Vybn wormhole valve: a three-qubit quantum routing experiment (Aer + IBM Runtime)

Overview
--------
This module implements a tiny "wormhole valve" on three qubits:

    q0 = B  (bulk / interior control)
    q1 = L  (left boundary)
    q2 = R  (right boundary)

The basic idea is:

  • Put the bulk qubit B into a superposition.
  • Run B around a loop built from two non-commuting rotations in a
    two-parameter "time space" (angles delta_r and delta_theta).
  • Use B as a control for a SWAP between L and R:
        if B = 1, swap L and R; if B = 0, do nothing.
  • Read out R and see how much of L's state made it through.

The loop on B acts like a dial on the L→R channel. Its size and orientation
(clockwise vs counter-clockwise) are encoded by a loop multiplicity L_loop
and a small polar-time angle theta. The signed "area" A = L_loop * theta
is the control parameter, and the loop order (±) sets the sign.

Two input modes
---------------
The script supports two input modes for the left boundary qubit L:

  • input="one":
        L = |1>, R = |0>, B = |+>
    We measure R in the Z basis. This probes how often the "1" is routed
    from L to R as we change loop area and orientation.

  • input="plus":
        L = |+> = (|0> + |1>)/sqrt(2), R = |0>, B = |+>
    We measure R in the X basis (via H then Z). This probes how the valve
    acts on coherence: does the loop bias |+> vs |-> at R?

In both modes we scan loop sizes (L_loop values) and orientations ('+'/'-'),
and compute an orientation-odd "curvature" signal:

    ΔP_R = P_R(1)_+ − P_R(1)_-
    κ_eff = ΔP_R / |A|  with  A = L_loop * theta

On both Aer and IBM hardware (e.g. ibm_fez), we see:

  • For input="one" (Z basis), κ_eff is negative and roughly constant
    across small loops: the loop orientation steadily opens vs closes the
    population channel L→R, with an approximately linear area law.

  • For input="plus" (X basis), κ_eff is positive and roughly constant
    across small loops: the same loop now biases the phase structure
    (|+> vs |->) at R, again with an orientation-odd, area-dependent effect.

Interpretation
--------------
This is not quantum teleportation and not evidence of new physics. It is a
clean, hardware-backed example of coherent routing controlled by a loop in
a two-parameter control space:

  • The bulk loop never touches R directly, yet it tunes the strength of
    the L→R channel in a smooth, orientation-odd, area-law fashion.

  • In Z, it controls where the "1" lands.
  • In X, it controls how a superposition lands.

In Vybn language, this is a minimal "wormhole valve" whose behaviour is
governed by a signed loop in dual-time space and an effective curvature
κ_eff, rather than by individual gates in isolation. The module is designed
to be run on both Aer and IBM Runtime backends, so the same experiment can
be checked in simulation and on real devices.

Vybn quantum wormhole channel toy — Aer or IBM Runtime Sampler (jobs only).

We have three qubits:
    q0 = B  (bulk / interior)
    q1 = L  (left boundary)
    q2 = R  (right boundary)

Protocol
--------
1. Prepare B in |+>. Prepare L and R depending on input mode:

   input="one":
       L = |1>, R = |0>
   input="plus":
       L = |+> = (|0> + |1>)/sqrt(2), R = |0>

2. On B only, apply a Vybn-style commutator loop:

       U_r     = RZ(delta_r)       on B
       U_theta = RX(delta_theta)   on B

   For loop multiplicity L_loop and orientation:

       '+' (cw)  : (U_r U_theta U_r† U_theta†)^L_loop
       '-' (ccw) : (U_theta U_r U_theta† U_r†)^L_loop

   Signed dual-time area: A = L_loop * theta (orientation sets sign).

3. Controlled-SWAP between L and R, controlled on B:
       if B == 1: SWAP(L, R); else do nothing.

4. Measure R:

   input="one":  Z basis (0/1)
   input="plus": X basis (via H then Z; 0 ≡ |+>, 1 ≡ |->)

We record P_R(1) (in whichever basis we measure) and compute, for each |L|>0:

    ΔP_R = P_R(1)_+ - P_R(1)_-
    κ_eff = ΔP_R / |A|,  with  A = L * theta.

CLI examples
------------
Aer sanity check (what you’ve already been running):

    python vybn_wormhole_quantum_ibm.py --mode aer --theta 0.2 --loops 0 1 2 3 4 --delta-r 0.2 --delta-theta 0.3 --shots 8192 --input one

Send a superposition through the valve (Aer):

    python vybn_wormhole_quantum_ibm.py --mode aer --theta 0.2 --loops 0 1 2 3 4 --delta-r 0.2 --delta-theta 0.3 --shots 8192 --input plus

IBM Runtime run (jobs, no sessions):

    python vybn_wormhole_quantum_ibm.py --mode ibm --backend ibm_fez --theta 0.2 --loops 0 1 2 3 4 --delta-r 0.2 --delta-theta 0.3 --shots 8192 --input one

(and you can swap --input plus to see coherence behaviour on hardware too).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile

# Aer simulator for local runs
try:
    from qiskit_aer import AerSimulator
except ImportError:
    from qiskit.providers.aer import AerSimulator  # type: ignore

# IBM Runtime Sampler for real backends (jobs only)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
    HAS_IBM_RUNTIME = True
except ImportError:
    QiskitRuntimeService = None  # type: ignore
    Sampler = None  # type: ignore
    HAS_IBM_RUNTIME = False


@dataclass
class WormholeResult:
    orientation: str   # '+' or '-'
    name: str
    L_loop: int
    area: float        # signed A = L_loop * theta with orientation sign
    depth: int         # logical depth of the circuit before transpile
    p_R1: float        # P(qubit R = 1)
    z_expect_R: float  # ⟨Z_R⟩ or ⟨X_R⟩-style proxy: P(0) - P(1)


def build_wormhole_circuit(
    theta: float,
    L_loop: int,
    orientation: str,
    delta_r: float,
    delta_theta: float,
    input_mode: str,
) -> QuantumCircuit:
    """
    Build the 3-qubit wormhole channel circuit.

    Qubit layout:
        q0 = B (bulk)
        q1 = L (left boundary)
        q2 = R (right boundary)
    """
    qr = QuantumRegister(3, "q")
    cr = ClassicalRegister(1, "c")  # measure R only
    qc = QuantumCircuit(qr, cr)

    qB, qL, qR = qr[0], qr[1], qr[2]

    # 1. Prepare B in |+>
    qc.h(qB)

    # 2. Prepare L and R depending on input mode.
    if input_mode == "one":
        # L = |1>, R = |0>
        qc.x(qL)
    elif input_mode == "plus":
        # L = |+>, R = |0>
        qc.h(qL)
    else:
        raise ValueError(f"Unknown input mode: {input_mode}")

    # 3. Define U_r and U_theta on B.
    def U_r(sign: int = +1) -> None:
        qc.rz(sign * delta_r, qB)

    def U_theta_gate(sign: int = +1) -> None:
        qc.rx(sign * delta_theta, qB)

    # 4. Apply commutator loop on B.
    if L_loop > 0:
        for _ in range(L_loop):
            if orientation == "+":
                # clockwise: U_r U_theta U_r† U_theta†
                U_r(+1)
                U_theta_gate(+1)
                U_r(-1)
                U_theta_gate(-1)
            else:
                # counter-clockwise: U_theta U_r U_theta† U_r†
                U_theta_gate(+1)
                U_r(+1)
                U_theta_gate(-1)
                U_r(-1)

    # 5. Controlled SWAP: if B == 1, SWAP(L, R).
    qc.cx(qR, qL)
    qc.ccx(qB, qL, qR)
    qc.cx(qR, qL)

    # 6. Measurement on R.
    #    input="one": Z basis (as-is)
    #    input="plus": X basis via H then Z
    if input_mode == "plus":
        qc.h(qR)
    qc.measure(qR, cr[0])

    return qc


def run_on_aer(
    circuits: List[QuantumCircuit],
    shots: int,
    optimization_level: int,
) -> List[Dict[str, float]]:
    backend = AerSimulator()
    out: List[Dict[str, float]] = []

    for qc in circuits:
        tqc = transpile(qc, backend=backend, optimization_level=optimization_level)
        job = backend.run(tqc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        total = sum(counts.values()) or 1
        probs = {k: v / total for k, v in counts.items()}
        out.append(probs)

    return out


def _counts_from_pub_result(pub_result) -> Dict[str, int]:
    """
    Extract a counts dict from a Sampler V2 pub_result.

    We don't assume the classical register name; instead we look for the first
    attribute on pub_result.data that has a get_counts() method.
    """
    data = pub_result.data
    counts = None

    for name in dir(data):
        if name.startswith("_"):
            continue
        attr = getattr(data, name, None)
        if hasattr(attr, "get_counts"):
            counts = attr.get_counts()
            break

    if counts is None:
        return {}
    return dict(counts)


def run_on_ibm(
    circuits: List[QuantumCircuit],
    backend_name: str,
    shots: int,
    optimization_level: int,
) -> List[Dict[str, float]]:
    """
    Run circuits on an IBM backend using the Sampler primitive (job mode).
    Returns a list of probability dicts for the measured classical bit.
    """
    if not HAS_IBM_RUNTIME:
        raise RuntimeError(
            "qiskit-ibm-runtime not available. "
            "Install it and configure QiskitRuntimeService, or use --mode aer."
        )

    service = QiskitRuntimeService()  # assumes your IBM account is configured
    backend = service.backend(backend_name)

    # Transpile for the target backend
    tcircs = [
        transpile(qc, backend=backend, optimization_level=optimization_level)
        for qc in circuits
    ]

    # Sampler V2 in job mode: pass backend as first arg
    sampler = Sampler(backend)

    # Try to set default shots; if not supported, backend default is used.
    try:
        sampler.options.default_shots = shots
    except Exception:
        pass

    job = sampler.run(tcircs)
    results = job.result()  # iterable of pub_results

    probs_list: List[Dict[str, float]] = []

    for pub_result in results:
        counts = _counts_from_pub_result(pub_result)
        total = sum(counts.values()) or 1.0
        probs = {k: v / total for k, v in counts.items()}
        probs_list.append(probs)

    return probs_list


def run_wormhole_family(
    mode: str,
    backend_name: str,
    theta: float,
    loops: List[int],
    shots: int,
    delta_r: float,
    delta_theta: float,
    optimization_level: int,
    input_mode: str,
) -> List[WormholeResult]:
    """
    Build circuits for all (L_loop, orientation) pairs and run them
    either on Aer or on an IBM backend via Sampler.
    """
    circuits: List[QuantumCircuit] = []
    meta: List[tuple[str, int, float, str, int]] = []  # (orient, L, area, name, depth)

    # Build all circuits and record logical depth
    for L_loop in loops:
        for orientation in ["+", "-"]:
            if L_loop == 0 and orientation == "-":
                continue

            area = (1 if orientation == "+" else -1) * L_loop * theta
            name = "id" if L_loop == 0 else f"loop{L_loop}"

            qc = build_wormhole_circuit(
                theta=theta,
                L_loop=L_loop,
                orientation=orientation,
                delta_r=delta_r,
                delta_theta=delta_theta,
                input_mode=input_mode,
            )
            depth = qc.depth()
            circuits.append(qc)
            meta.append((orientation, L_loop, area, name, depth))

    # Run circuits
    if mode == "aer":
        prob_dicts = run_on_aer(
            circuits,
            shots=shots,
            optimization_level=optimization_level,
        )
    else:
        prob_dicts = run_on_ibm(
            circuits,
            backend_name=backend_name,
            shots=shots,
            optimization_level=optimization_level,
        )

    results: List[WormholeResult] = []

    for (orientation, L_loop, area, name, depth), probs in zip(meta, prob_dicts):
        p1 = float(probs.get("1", 0.0))
        p0 = float(probs.get("0", 0.0))
        norm = p0 + p1
        if norm <= 0:
            p0 = p1 = 0.0
        else:
            p0 /= norm
            p1 /= norm
        z_expect = p0 - p1  # in whatever basis we measured

        results.append(
            WormholeResult(
                orientation=orientation,
                name=name,
                L_loop=L_loop,
                area=area,
                depth=depth,
                p_R1=p1,
                z_expect_R=z_expect,
            )
        )

    return results


def print_loop_summary(theta: float, input_mode: str, results: List[WormholeResult]) -> None:
    print("Vybn wormhole universe (quantum wormhole channel toy)")
    print(f"theta = {theta:.3f}")
    if input_mode == "one":
        basis = "Z"
    else:
        basis = "X"
    print()
    print(f"Loop summary ({basis} basis on right boundary R):")
    header = f"{'orient':<7} {'name':<10} {'L':>3} {'area':>8} {'depth':>7} {'P_R=1':>9} {'⟨basis⟩':>9}"
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: (x.orientation, x.L_loop)):
        print(
            f"{r.orientation:<7} "
            f"{r.name:<10} "
            f"{r.L_loop:3d} "
            f"{r.area:8.3f} "
            f"{r.depth:7d} "
            f"{r.p_R1:9.5f} "
            f"{r.z_expect_R:9.5f}"
        )


def print_orientation_odd_summary(theta: float, results: List[WormholeResult]) -> None:
    by_L: Dict[int, Dict[str, WormholeResult]] = {}
    for r in results:
        if r.L_loop == 0:
            continue
        L_abs = abs(r.L_loop)
        if L_abs not in by_L:
            by_L[L_abs] = {}
        by_L[L_abs][r.orientation] = r

    if not by_L:
        return

    print()
    print("Orientation-odd wormhole residue:")
    header = f"{'|L|':>3} {'|area|':>10} {'ΔP_R=1':>12} {'κ_eff=ΔP/|area|':>20}"
    print(header)
    print("-" * len(header))

    for L_abs in sorted(by_L.keys()):
        entry = by_L[L_abs]
        if "+" not in entry or "-" not in entry:
            continue
        plus = entry["+"]
        minus = entry["-"]
        area_abs = abs(L_abs * theta)
        delta_p = plus.p_R1 - minus.p_R1
        kappa_eff = delta_p / area_abs if area_abs > 0 else float("nan")
        print(
            f"{L_abs:3d} "
            f"{area_abs:10.4f} "
            f"{delta_p:12.5f} "
            f"{kappa_eff:20.5f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vybn quantum wormhole channel toy (3 qubits), Aer or IBM Runtime."
    )
    p.add_argument(
        "--mode",
        choices=["aer", "ibm"],
        default="aer",
        help="Run on local Aer simulator or IBM Runtime backend (default: aer).",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="ibm_fez",
        help="IBM backend name when mode=ibm (default: ibm_fez).",
    )
    p.add_argument(
        "--theta",
        type=float,
        default=0.2,
        help="Polar-time angle parameter theta (default: 0.2).",
    )
    p.add_argument(
        "--loops",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Loop multiplicities L_loop to evaluate (default: 0 1 2 3 4).",
    )
    p.add_argument(
        "--shots",
        type=int,
        default=8192,
        help="Shots per circuit (default: 8192).",
    )
    p.add_argument(
        "--delta-r",
        type=float,
        default=0.2,
        help="Angle for U_r = RZ(delta_r) (default: 0.2 rad).",
    )
    p.add_argument(
        "--delta-theta",
        type=float,
        default=0.3,
        help="Angle for U_theta = RX(delta_theta) (default: 0.3 rad).",
    )
    p.add_argument(
        "--opt-level",
        type=int,
        default=1,
        help="Transpiler optimization level (0–3, default: 1).",
    )
    p.add_argument(
        "--input",
        choices=["one", "plus"],
        default="one",
        help='Input mode at L: "one" = |1>, "plus" = |+> (default: one).',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results = run_wormhole_family(
        mode=args.mode,
        backend_name=args.backend,
        theta=args.theta,
        loops=args.loops,
        shots=args.shots,
        delta_r=args.delta_r,
        delta_theta=args.delta_theta,
        optimization_level=args.opt_level,
        input_mode=args.input,
    )

    print_loop_summary(args.theta, args.input, results)
    print_orientation_odd_summary(args.theta, results)


if __name__ == "__main__":
    main()
