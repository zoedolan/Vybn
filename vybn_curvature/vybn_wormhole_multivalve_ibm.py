#!/usr/bin/env python
"""
vybn_wormhole_multivalve_ibm.py

Multi-valve Vybn wormhole chain on 2N+1 qubits.

Geometry:

  Nodes (even indices):  q[0] = L (left boundary),
                         q[2], q[4], …, q[2N-2] = interior nodes M_1…M_{N-1},
                         q[2N] = R (right boundary).

  Valves (odd indices):  q[1], q[3], …, q[2N-1] = bulk valves B_1…B_N.

For each valve B_k we:

  • prepare B_k in |+>
  • apply the same dual-temporal commutator loop L_loop times
  • use B_k as control for a CSWAP between its two neighboring nodes

So for N = 3 (your 7-qubit run):

  q0 = L, q1 = B1, q2 = M1, q3 = B2, q4 = M2, q5 = B3, q6 = R

  CSWAP(B1; L <-> M1)
  CSWAP(B2; M1 <-> M2)
  CSWAP(B3; M2 <-> R)

Input modes:

  input="one"  : L = |1>, rest = |0>
  input="plus" : L = |+>, rest = |0>

Measurement:

  • measure R in Z if input="one"
  • measure R in X via H then Z if input="plus"

For each loop multiplicity L and orientation (+/-) we record P_R(1).
We then compute the orientation-odd residue ΔP = P_+ − P_- and define

   area_tot  = N_valves * L * theta   with sign from orientation
   κ_eff     = ΔP / |area_tot|

Your 3-qubit and 5-qubit experiments already showed a linear area law
with roughly constant κ_eff for small loops. This script asks whether
κ_eff stays intensive when you extend the bulk to multiple valves.

CLI examples:

  # Aer sanity check, 1,2,3 valves
  python vybn_wormhole_multivalve_ibm.py --mode aer --valves 3 \
      --theta 0.2 --loops 0 1 2 --delta-r 0.2 --delta-theta 0.3 \
      --shots 8192 --input plus

  # Final IBM run, 3 valves (7 qubits)
  python vybn_wormhole_multivalve_ibm.py --mode ibm --backend ibm_fez \
      --valves 3 --theta 0.2 --loops 0 1 2 --delta-r 0.2 --delta-theta 0.3 \
      --shots 4096 --input plus

Keep loops short (e.g. 0,1,2) so depth stays modest and your 30–40 s
budget isn’t blown on compilation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile

# Aer simulator
try:
    from qiskit_aer import AerSimulator
except ImportError:  # pragma: no cover
    from qiskit.providers.aer import AerSimulator  # type: ignore

# IBM Runtime (jobs only)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
    HAS_IBM_RUNTIME = True
except ImportError:  # pragma: no cover
    QiskitRuntimeService = None  # type: ignore
    Sampler = None  # type: ignore
    HAS_IBM_RUNTIME = False


@dataclass
class ChainResult:
    orientation: str
    name: str
    L_loop: int
    area_total: float
    depth: int
    p_R1: float
    basis_expect_R: float


def cswap(qc: QuantumCircuit, ctrl, a, b) -> None:
    """Standard CSWAP decomposition using CX and CCX."""
    qc.cx(b, a)
    qc.ccx(ctrl, a, b)
    qc.cx(b, a)


def build_chain_circuit(
    theta: float,
    L_loop: int,
    orientation: str,
    delta_r: float,
    delta_theta: float,
    input_mode: str,
    n_valves: int,
) -> QuantumCircuit:
    """Construct an N-valve wormhole chain circuit on 2N+1 qubits."""

    if n_valves < 1:
        raise ValueError("n_valves must be >= 1")

    num_qubits = 2 * n_valves + 1  # nodes at even indices, valves at odd
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    def node_index(k: int) -> int:
        # k = 0…n_valves: Node_0=L, Node_n=R
        return 2 * k

    def valve_index(k: int) -> int:
        # k = 1…n_valves: B_k at odd positions
        return 2 * k - 1

    # Convenience handles
    qL = qr[node_index(0)]
    qR = qr[node_index(n_valves)]

    # Prepare bulk valves in |+>
    for k in range(1, n_valves + 1):
        qc.h(qr[valve_index(k)])

    # Input state on L
    if input_mode == "one":
        qc.x(qL)   # |1>
    elif input_mode == "plus":
        qc.h(qL)   # |+>
    else:
        raise ValueError("input_mode must be 'one' or 'plus'")

    # Define loop steps on a given bulk qubit
    def loop_on(qb):
        if L_loop <= 0:
            return
        for _ in range(L_loop):
            if orientation == "+":
                qc.rz(+delta_r, qb)
                qc.rx(+delta_theta, qb)
                qc.rz(-delta_r, qb)
                qc.rx(-delta_theta, qb)
            elif orientation == "-":
                qc.rx(+delta_theta, qb)
                qc.rz(+delta_r, qb)
                qc.rx(-delta_theta, qb)
                qc.rz(-delta_r, qb)
            else:
                raise ValueError("orientation must be '+' or '-'")

    # Apply same holonomy loop to each valve
    for k in range(1, n_valves + 1):
        loop_on(qr[valve_index(k)])

    # Chain of CSWAPs: for each valve, swap between its neighboring nodes
    for k in range(1, n_valves + 1):
        ctrl = qr[valve_index(k)]
        left = qr[node_index(k - 1)]
        right = qr[node_index(k)]
        cswap(qc, ctrl, left, right)

    # Measurement basis on R
    if input_mode == "plus":
        qc.h(qR)  # X basis via H then Z

    qc.measure(qR, cr[0])

    return qc


def run_on_aer(
    circuits: List[QuantumCircuit],
    shots: int,
    optimization_level: int,
) -> List[Dict[str, float]]:
    """Run circuits on Aer and return probability dicts for c[0]."""
    backend = AerSimulator()
    out: List[Dict[str, float]] = []

    for qc in circuits:
        tqc = transpile(qc, backend=backend, optimization_level=optimization_level)
        job = backend.run(tqc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        total = sum(counts.values()) or 1
        out.append({k: v / total for k, v in counts.items()})

    return out


def _counts_from_pub_result(pub_result) -> Dict[str, int]:
    """Extract counts from a Sampler pub_result in a backend-agnostic way."""
    data = getattr(pub_result, "data", None)
    if data is None:
        return {}
    # Try standard accessors first
    for name in ("meas", "cr"):
        obj = getattr(data, name, None)
        if obj is not None and hasattr(obj, "get_counts"):
            return dict(obj.get_counts())
    # Fallback: scan attributes
    for name in dir(data):
        if name.startswith("_"):
            continue
        obj = getattr(data, name, None)
        if hasattr(obj, "get_counts"):
            return dict(obj.get_counts())
    return {}


def run_on_ibm(
    circuits: List[QuantumCircuit],
    backend_name: str,
    shots: int,
    optimization_level: int,
) -> List[Dict[str, float]]:
    """Run circuits on an IBM backend using the Sampler primitive."""
    if not HAS_IBM_RUNTIME:
        raise RuntimeError(
            "qiskit-ibm-runtime not available. "
            "Install it and configure QiskitRuntimeService, or use --mode aer."
        )

    service = QiskitRuntimeService()  # assumes your IBM account is configured
    backend = service.backend(backend_name)

    tcircs = [
        transpile(
            qc,
            backend=backend,
            optimization_level=optimization_level,
        )
        for qc in circuits
    ]

    sampler = Sampler(backend)

    try:
        sampler.options.default_shots = shots
    except Exception:
        pass

    job = sampler.run(tcircs)
    results = job.result()

    probs_list: List[Dict[str, float]] = []

    for pub_result in results:
        counts = _counts_from_pub_result(pub_result)
        total = sum(counts.values()) or 1.0
        probs = {k: v / total for k, v in counts.items()}
        probs_list.append(probs)

    return probs_list


def run_chain_family(
    mode: str,
    backend_name: str,
    theta: float,
    loops: List[int],
    shots: int,
    delta_r: float,
    delta_theta: float,
    optimization_level: int,
    input_mode: str,
    n_valves: int,
) -> List[ChainResult]:
    """Build and run the whole (orientation, L) family and collect results."""
    circuits: List[QuantumCircuit] = []
    meta: List[Tuple[str, int, float, str, int]] = []  # orient, L, area_total, name, depth

    for L in loops:
        for orient in ("+", "-"):
            # Skip redundant L=0, '-' case (no loop, same plumbing)
            if L == 0 and orient == "-":
                continue
            area_total = (1.0 if orient == "+" else -1.0) * float(n_valves) * float(L) * theta
            name = "id" if L == 0 else f"loop{L}"
            qc = build_chain_circuit(
                theta=theta,
                L_loop=L,
                orientation=orient,
                delta_r=delta_r,
                delta_theta=delta_theta,
                input_mode=input_mode,
                n_valves=n_valves,
            )
            depth = qc.depth()
            circuits.append(qc)
            meta.append((orient, L, area_total, name, depth))

    if not circuits:
        return []

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

    results: List[ChainResult] = []

    for (orient, L, area_total, name, depth), probs in zip(meta, prob_dicts):
        p1 = float(probs.get("1", 0.0))
        p0 = float(probs.get("0", 0.0))
        s = p0 + p1 or 1.0
        p0 /= s
        p1 /= s
        basis_expect = p0 - p1
        results.append(
            ChainResult(
                orientation=orient,
                name=name,
                L_loop=L,
                area_total=area_total,
                depth=depth,
                p_R1=p1,
                basis_expect_R=basis_expect,
            )
        )

    return results


def print_summary(theta: float, input_mode: str, n_valves: int, results: List[ChainResult]) -> None:
    """Pretty-print loop-by-loop data and the orientation-odd κ summary."""
    basis = "Z" if input_mode == "one" else "X"

    print(f"Vybn wormhole universe (multi-valve chain, N={n_valves})")
    print(f"theta = {theta:.3f}")
    print(f"Basis on R: {basis}")
    print()

    header = (
        f"{'orient':<7} {'name':<10} {'L':>3} "
        f"{'area_tot':>10} {'depth':>7} {'P_R=1':>10} {'⟨basis⟩':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: (x.L_loop, x.orientation)):
        print(
            f"{r.orientation:<7} {r.name:<10} {r.L_loop:3d} "
            f"{r.area_total:10.4f} {r.depth:7d} {r.p_R1:10.5f} {r.basis_expect_R:10.5f}"
        )

    by_L: Dict[int, Dict[str, ChainResult]] = {}
    for r in results:
        if r.L_loop == 0:
            continue
        by_L.setdefault(r.L_loop, {})[r.orientation] = r

    if not by_L:
        return

    print()
    print("Orientation-odd residue (multi-valve chain):")
    header2 = f"{'|L|':>3} {'|area_tot|':>12} {'ΔP_R=1':>12} {'κ_eff=ΔP/|area_tot|':>22}"
    print(header2)
    print("-" * len(header2))

    for L in sorted(by_L):
        orient_dict = by_L[L]
        if "+" not in orient_dict or "-" not in orient_dict:
            continue
        plus = orient_dict["+"]
        minus = orient_dict["-"]
        area_abs = abs(plus.area_total)
        delta_p = plus.p_R1 - minus.p_R1
        kappa = delta_p / area_abs if area_abs > 0 else float("nan")
        print(f"{L:3d} {area_abs:12.4f} {delta_p:12.5f} {kappa:22.5f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vybn multi-valve wormhole chain (2N+1 qubits), Aer or IBM Runtime.",
    )
    p.add_argument(
        "--mode",
        choices=["aer", "ibm"],
        default="aer",
        help="Execution mode: 'aer' for local simulator, 'ibm' for IBM Runtime backend.",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="ibm_fez",
        help="IBM backend name when --mode ibm (default: ibm_fez).",
    )
    p.add_argument(
        "--valves",
        type=int,
        default=2,
        help="Number of bulk valves N (total qubits = 2N+1). Use N=3 for 7-qubit run.",
    )
    p.add_argument(
        "--theta",
        type=float,
        default=0.2,
        help="Base polar angle theta controlling the loop area (default: 0.2 rad).",
    )
    p.add_argument(
        "--loops",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="List of loop multiplicities L to run (default: 0 1 2).",
    )
    p.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Shots per circuit (default: 4096).",
    )
    p.add_argument(
        "--delta-r",
        type=float,
        default=0.2,
        help="Angle for U_r = RZ(delta_r) in the loop (default: 0.2 rad).",
    )
    p.add_argument(
        "--delta-theta",
        type=float,
        default=0.3,
        help="Angle for U_theta = RX(delta_theta) in the loop (default: 0.3 rad).",
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
        default="plus",
        help='Input mode at L: "one" = |1>, "plus" = |+> (default: plus).',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = run_chain_family(
        mode=args.mode,
        backend_name=args.backend,
        theta=args.theta,
        loops=args.loops,
        shots=args.shots,
        delta_r=args.delta_r,
        delta_theta=args.delta_theta,
        optimization_level=args.opt_level,
        input_mode=args.input,
        n_valves=args.valves,
    )
    print_summary(args.theta, args.input, args.valves, results)


if __name__ == "__main__":
    main()
