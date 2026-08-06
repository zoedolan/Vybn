#!/usr/bin/env python3
"""Source–absorber closure interferometer on a two-qubit IBM processor.

This does not test a departure from quantum mechanics. It gives the relational
light conjecture an operational object: the complex interference term between
two coherent source-to-absorber paths that differ only in the order of two
noncommuting transformations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

AX = 0.90 * math.pi
BZ = 0.85 * math.pi
SHOTS_DEFAULT = 2048
VARIANTS = ("forward", "reverse", "commuting", "mismatch", "forward_pi_twirl")
BASES = ("x", "y")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def circuit(variant: str, basis: str, measure: bool = True) -> QuantumCircuit:
    """Prepare |source=0>, coherently route two orders, project on absorber.

    q[0] is the path qubit and q[1] the source/absorber system.  The path-0
    branch receives W0 and path-1 receives W1.  For the main pair,

        W0 = Rz(BZ) Rx(AX),   W1 = Rx(AX) Rz(BZ).

    `reverse` exchanges W0 and W1. `commuting` uses two Rx rotations, so both
    branches are exactly the same unitary. `mismatch` changes only the final
    absorber from |1> to |0>. `forward_pi_twirl` adds a pi relative path phase;
    averaging it with `forward` is an operational dephasing control.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if basis not in BASES:
        raise ValueError(f"unknown path basis: {basis}")

    qc = QuantumCircuit(2, 2, name=f"{variant}_{basis}")
    path, system = 0, 1
    qc.h(path)

    if variant == "commuting":
        branch0 = (("rx", 0.40 * math.pi), ("rx", 0.50 * math.pi))
        branch1 = tuple(reversed(branch0))
    else:
        branch0 = (("rx", AX), ("rz", BZ))
        branch1 = (("rz", BZ), ("rx", AX))
        if variant == "reverse":
            branch0, branch1 = branch1, branch0

    # Apply branch0 when path=0, then branch1 when path=1.
    qc.x(path)
    for gate, angle in branch0:
        getattr(qc, f"c{gate}")(angle, path, system)
    qc.x(path)
    for gate, angle in branch1:
        getattr(qc, f"c{gate}")(angle, path, system)

    if variant == "forward_pi_twirl":
        qc.z(path)

    # Main absorber is |1>; mismatch is |0>.  Map accepted absorber to 0.
    if variant != "mismatch":
        qc.x(system)

    # Read path in X or Y; outcome 0 is the + eigenstate.
    if basis == "x":
        qc.h(path)
    else:
        qc.sdg(path)
        qc.h(path)

    if measure:
        qc.measure(path, 0)
        qc.measure(system, 1)
    return qc


def probabilities_to_counts(probs: dict[str, float], shots: int) -> dict[str, float]:
    return {key: value * shots for key, value in probs.items() if value > 1e-15}


def ideal_counts(shots: int = 1) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        for basis in BASES:
            qc = circuit(variant, basis, measure=False)
            probs = Statevector.from_instruction(qc).probabilities_dict()
            out[f"{variant}_{basis}"] = probabilities_to_counts(probs, shots)
    return out


def normalized_probs(counts: dict[str, float]) -> dict[tuple[int, int], float]:
    """Return {(absorber_bit, path_bit): probability}; bitstring is c1c0."""
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError("empty counts")
    out: dict[tuple[int, int], float] = {}
    for absorber in (0, 1):
        for path in (0, 1):
            out[(absorber, path)] = float(counts.get(f"{absorber}{path}", 0)) / total
    return out


def basis_stats(counts: dict[str, float]) -> dict[str, float]:
    p = normalized_probs(counts)
    joint_difference = p[(0, 0)] - p[(0, 1)]
    success = p[(0, 0)] + p[(0, 1)]
    path_plus = p[(0, 0)] + p[(1, 0)]
    n = float(sum(counts.values()))
    se = math.sqrt(max(success - joint_difference * joint_difference, 0.0) / n)
    return {
        "joint_difference": joint_difference,
        "joint_difference_se": se,
        "absorber_success": success,
        "path_plus_marginal": path_plus,
    }


def analyze(counts_by_name: dict[str, dict[str, float]]) -> dict[str, Any]:
    basis: dict[str, dict[str, dict[str, float]]] = {}
    closure: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        basis[variant] = {
            b: basis_stats(counts_by_name[f"{variant}_{b}"]) for b in BASES
        }
        re = basis[variant]["x"]["joint_difference"]
        im = basis[variant]["y"]["joint_difference"]
        c_abs = math.hypot(re, im)
        closure[variant] = {
            "re": re,
            "im": im,
            "magnitude": c_abs,
            "phase_rad": math.atan2(im, re) if c_abs else 0.0,
            "absorber_success_mean": 0.5 * (
                basis[variant]["x"]["absorber_success"]
                + basis[variant]["y"]["absorber_success"]
            ),
        }

    fwd = closure["forward"]
    rev = closure["reverse"]
    twirl = closure["forward_pi_twirl"]
    mismatch = closure["mismatch"]
    commuting = closure["commuting"]
    twirl_avg_re = 0.5 * (fwd["re"] + twirl["re"])
    twirl_avg_im = 0.5 * (fwd["im"] + twirl["im"])

    no_signal_deltas = {
        b: basis["forward"][b]["path_plus_marginal"]
        - basis["mismatch"][b]["path_plus_marginal"]
        for b in BASES
    }
    controls = {
        "reversal_phase_sum_rad": fwd["phase_rad"] + rev["phase_rad"],
        "reversal_magnitude_delta": fwd["magnitude"] - rev["magnitude"],
        "commuting_phase_rad": commuting["phase_rad"],
        "mismatch_to_match_magnitude_ratio": (
            mismatch["magnitude"] / fwd["magnitude"] if fwd["magnitude"] else None
        ),
        "dephased_average_magnitude": math.hypot(twirl_avg_re, twirl_avg_im),
        "receiver_choice_path_marginal_deltas": no_signal_deltas,
    }
    ratio = controls["mismatch_to_match_magnitude_ratio"]
    checks = {
        "reversal_sign": fwd["phase_rad"] < -0.5 and rev["phase_rad"] > 0.5,
        "reversal_conjugacy": (
            abs(controls["reversal_phase_sum_rad"]) < 0.15
            and abs(controls["reversal_magnitude_delta"]) < 0.10
        ),
        "commuting_flat": abs(controls["commuting_phase_rad"]) < 0.10,
        "mismatch_suppressed": ratio is not None and ratio < 0.10,
        "dephasing_erases": (
            controls["dephased_average_magnitude"] < 0.10 * fwd["magnitude"]
        ),
        "receiver_no_signal": max(abs(v) for v in no_signal_deltas.values()) < 0.05,
    }
    if all(checks.values()):
        verdict = "pass"
    elif all(
        checks[name]
        for name in (
            "reversal_sign",
            "mismatch_suppressed",
            "dephasing_erases",
            "receiver_no_signal",
        )
    ):
        verdict = "partial"
    else:
        verdict = "fail"
    return {
        "basis": basis,
        "closure": closure,
        "controls": controls,
        "checks": checks,
        "verdict": verdict,
    }


def run_ibm(shots: int) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    circuits = [circuit(v, b) for v in VARIANTS for b in BASES]
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1, seed_transpiler=20260806
    )
    isa = pm.run(circuits)
    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa, shots=shots)
    result = job.result()
    names = [f"{v}_{b}" for v in VARIANTS for b in BASES]
    counts = {
        name: pub_result.data.c.get_counts()
        for name, pub_result in zip(names, result)
    }
    metadata = {
        "backend": backend.name,
        "job_id": job.job_id(),
        "shots": shots,
        "optimization_level": 1,
        "seed_transpiler": 20260806,
        "transpiled": [
            {"name": name, "depth": qc.depth(), "two_qubit_ops": qc.num_nonlocal_gates()}
            for name, qc in zip(names, isa)
        ],
    }
    return counts, metadata


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ibm", action="store_true", help="run on IBM hardware")
    parser.add_argument("--shots", type=int, default=SHOTS_DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ideal = ideal_counts(shots=1_000_000)
    payload: dict[str, Any] = {
        "schema": "vybn.source_absorber_closure.v1",
        "timestamp": utc_now(),
        "source_sha256": source_sha256(),
        "claim_limit": (
            "Tests the operational closure/holonomy mathematics under standard quantum "
            "mechanics; it does not distinguish the relational ontology from QED."
        ),
        "parameters": {"AX_rad": AX, "BZ_rad": BZ},
        "ideal": analyze(ideal),
    }
    if args.ibm:
        counts, metadata = run_ibm(args.shots)
        payload["hardware"] = {"metadata": metadata, "counts": counts, "analysis": analyze(counts)}

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
