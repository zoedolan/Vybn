from __future__ import annotations

import argparse
import random
import math
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile

try:
    from qiskit_aer import AerSimulator  # type: ignore
except ImportError:
    try:
        from qiskit.providers.aer import AerSimulator  # type: ignore
    except ImportError:
        AerSimulator = None  # type: ignore

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler import generate_preset_pass_manager
    HAS_IBM = True
except ImportError:
    HAS_IBM = False


"""
Vybn wormhole lab + RLQMF + teleported agents.

Subcommands:

  wire         – three-qubit "wormhole wire" sanity check.
  transport    – L–R chain entanglement transport vs loop multiplicity L.
  backreaction – valve-signal experiment (how a single valve opens/chokes the wormhole).
  diary        – black-hole diary toy: scramble and recover via wormhole vs local.
  analyze      – run wire + transport + backreaction + diary and print analytic headlines.
  rlqmf        – reinforcement learning from quantum mechanical feedback (bandit over params).
  agent        – paired bandit agents: control vs wormhole-teleported confidence, fully logged.
"""


# --------------------------------------------------------------------
# Geometry configuration
# --------------------------------------------------------------------


@dataclass
class GeometryConfig:
    name: str
    vecs: List[List[float]]


def dual2_vecs() -> List[List[float]]:
    return [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]


def e8_vecs() -> List[List[float]]:
    v1 = [1.0] * 8
    v2 = [1.0] * 4 + [-1.0] * 4
    v3: List[float] = []
    s = 1.0
    for _ in range(8):
        v3.append(s)
        s *= -1.0
    return [v1, v2, v3]


def leech_vecs() -> List[List[float]]:
    v1 = [1.0] * 24
    v2 = [1.0] * 12 + [-1.0] * 12
    v3: List[float] = []
    s = 1.0
    for _ in range(24):
        v3.append(s)
        s *= -1.0
    return [v1, v2, v3]


def make_geometries(labels: List[str]) -> List[GeometryConfig]:
    geoms: List[GeometryConfig] = []
    for label in labels:
        lbl = label.strip().lower()
        if not lbl:
            continue
        if lbl == "dual2":
            geoms.append(GeometryConfig("dual2", dual2_vecs()))
        elif lbl == "e8":
            geoms.append(GeometryConfig("e8", e8_vecs()))
        elif lbl == "leech":
            geoms.append(GeometryConfig("leech", leech_vecs()))
        else:
            raise ValueError(f"unknown model label: {label}")
    return geoms


# --------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------


def _cswap(qc: QuantumCircuit, ctrl, a, b) -> None:
    qc.cx(b, a)
    qc.ccx(ctrl, a, b)
    qc.cx(b, a)


def _twoq_count(qc: QuantumCircuit) -> int:
    ops = qc.count_ops()
    names = {"cx", "cz", "swap", "csx", "ecr", "rxx", "ryy", "rzz", "rzx", "ccx"}
    return sum(count for name, count in ops.items() if name in names)


def _basis_rotate(qc: QuantumCircuit, qb, basis: str) -> None:
    b = basis.upper()
    if b == "Z":
        return
    if b == "X":
        qc.h(qb)
        return
    if b == "Y":
        qc.sdg(qb)
        qc.h(qb)
        return
    raise ValueError("basis must be X, Y, or Z")


# --------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------


def run_aer(
    circuits: List[QuantumCircuit],
    shots: int,
    opt_level: int,
) -> List[Tuple[Dict[str, float], int, int]]:
    if AerSimulator is None:
        raise RuntimeError(
            "No AerSimulator available. Install 'qiskit-aer' or use --mode ibm."
        )

    backend = AerSimulator()
    out: List[Tuple[Dict[str, float], int, int]] = []

    for qc in circuits:
        tqc = transpile(qc, backend=backend, optimization_level=opt_level)
        depth = tqc.depth()
        twoq = _twoq_count(tqc)
        job = backend.run(tqc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        total = sum(counts.values()) or 1
        probs = {k: v / total for k, v in counts.items()}
        out.append((probs, depth, twoq))

    return out


def run_ibm(
    circuits: List[QuantumCircuit],
    backend_name: str,
    shots: int,
    opt_level: int,
) -> List[Tuple[Dict[str, float], int, int]]:
    if not HAS_IBM:
        raise RuntimeError(
            "qiskit-ibm-runtime not available; install it or use --mode aer."
        )

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    pm = generate_preset_pass_manager(
        optimization_level=opt_level,
        backend=backend,
    )
    isa_circuits = pm.run(circuits)

    sampler = Sampler(mode=backend)
    job = sampler.run(isa_circuits, shots=shots)
    result = job.result()

    out: List[Tuple[Dict[str, float], int, int]] = []
    for tqc, pub in zip(isa_circuits, result):
        depth = tqc.depth()
        twoq = _twoq_count(tqc)
        bit_array = pub.data.c  # type: ignore[attr-defined]
        counts = bit_array.get_counts()
        total = sum(counts.values()) or 1
        probs = {b: c / total for b, c in counts.items()}
        out.append((probs, depth, twoq))

    return out


# --------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------


def corr_from_2bit(probs: Dict[str, float]) -> Tuple[float, float, float]:
    C = 0.0
    m0 = 0.0
    m1 = 0.0
    for bits, p in probs.items():
        if len(bits) != 2:
            continue
        b0 = bits[-1]
        b1 = bits[-2]
        v0 = 1.0 if b0 == "0" else -1.0
        v1 = 1.0 if b1 == "0" else -1.0
        C += v0 * v1 * p
        m0 += v0 * p
        m1 += v1 * p
    return C, m0, m1


# --------------------------------------------------------------------
# Dataclasses for summaries
# --------------------------------------------------------------------


@dataclass
class TransportRow:
    geom: str
    channel: str
    vec_id: int
    scale: float
    L_loop: int
    Cxx_AR: float
    Cyy_AR: float
    Czz_AR: float
    depth: int
    twoq: int


@dataclass
class BackreactionRow:
    geom: str
    channel: str
    vec_id: int
    scale: float
    L_loop: int
    signal: str
    Cxx_AR: float
    Cyy_AR: float
    Czz_AR: float
    depth: int
    twoq: int


@dataclass
class ChannelResult:
    name: str
    Cxx: float
    Cyy: float
    Czz: float
    norm: float


@dataclass
class DiaryRow:
    channel: str
    theta: float
    mx: float
    my: float
    mz: float
    fidelity: float
    depth: int
    twoq: int


# --------------------------------------------------------------------
# A–R entanglement-transfer chain
# --------------------------------------------------------------------


def build_AR_corr_circ(
    vec: List[float],
    L_loop: int,
    delta_r: float,
    delta_theta: float,
    scale: float,
    channel: str,
    basis_pair: str,
    signal_type: str = "none",
    signal_valve: int = 1,
) -> QuantumCircuit:
    if len(basis_pair) != 2:
        raise ValueError("basis_pair must be length-2, e.g. 'XX'")
    bA, bR = basis_pair[0], basis_pair[1]

    N = len(vec)
    if N < 1:
        raise ValueError("vec must have length ≥ 1")

    num_qubits = 1 + (2 * N + 1)
    qr = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(qr)

    idx_A = 0
    offset = 1

    def node_index(k: int) -> int:
        return offset + 2 * k

    def valve_index(k: int) -> int:
        return offset + 2 * k - 1

    idx_L = node_index(0)
    idx_R = node_index(N)

    qc.h(qr[idx_A])
    qc.cx(qr[idx_A], qr[idx_L])

    for k in range(1, N + 1):
        qc.h(qr[valve_index(k)])

    sig = signal_type.lower()
    if sig != "none":
        if signal_valve < 1 or signal_valve > N:
            raise ValueError("signal_valve must be between 1 and N")
        vq = qr[valve_index(signal_valve)]
        if sig == "neg":
            qc.z(vq)
        elif sig == "pos0":
            qc.h(vq)
        elif sig == "pos1":
            qc.h(vq)
            qc.x(vq)
        else:
            raise ValueError(f"unknown signal_type: {signal_type}")

    def loop_on(qb, comp: float) -> None:
        if L_loop <= 0 or comp == 0.0:
            return
        dr = scale * delta_r * comp
        dt = scale * delta_theta * comp
        for _ in range(L_loop):
            qc.rz(+dr, qb)
            qc.rx(+dt, qb)
            qc.rz(-dr, qb)
            qc.rx(-dt, qb)

    for k in range(1, N + 1):
        comp = vec[k - 1]
        if comp != 0.0:
            loop_on(qr[valve_index(k)], comp)

    ch = channel.lower()
    if ch == "wormhole":
        for k in range(1, N + 1):
            ctrl = qr[valve_index(k)]
            left = qr[node_index(k - 1)]
            right = qr[node_index(k)]
            _cswap(qc, ctrl, left, right)
    elif ch == "local":
        for i in range(offset, offset + 2 * N):
            a = qr[i]
            b = qr[i + 1]
            qc.cz(a, b)
    else:
        raise ValueError("channel must be 'wormhole' or 'local'")

    cr = ClassicalRegister(2, "c")
    qc.add_register(cr)

    _basis_rotate(qc, qr[idx_A], bA)
    _basis_rotate(qc, qr[idx_R], bR)

    qc.measure(qr[idx_A], cr[0])
    qc.measure(qr[idx_R], cr[1])
    return qc


# --------------------------------------------------------------------
# Three-qubit wire sanity experiment
# --------------------------------------------------------------------


def build_entwire_circuits(channel: str) -> List[QuantumCircuit]:
    circs: List[QuantumCircuit] = []
    bases = ["ZZ", "XX", "YY"]

    for basis in bases:
        qc = QuantumCircuit(3, 2)

        qc.h(0)
        qc.cx(0, 1)

        if channel == "wormhole":
            qc.cx(1, 2)
            qc.cx(2, 1)
            qc.cx(1, 2)
        elif channel == "normal":
            qc.cz(1, 2)
            qc.cz(1, 2)
            qc.cz(1, 2)
        else:
            raise ValueError("channel must be 'normal' or 'wormhole'")

        if basis == "XX":
            qc.h(0)
            qc.h(2)
        elif basis == "YY":
            qc.sdg(0)
            qc.h(0)
            qc.sdg(2)
            qc.h(2)
        elif basis == "ZZ":
            pass
        else:
            raise ValueError("bad basis")

        qc.measure(0, 0)
        qc.measure(2, 1)
        qc.name = f"{channel}_{basis}"
        circs.append(qc)

    return circs


def summarize_wire_channel(name: str, probs_list: List[Dict[str, float]]) -> ChannelResult:
    probs_ZZ, probs_XX, probs_YY = probs_list

    Czz, _, _ = corr_from_2bit(probs_ZZ)
    Cxx, _, _ = corr_from_2bit(probs_XX)
    Cyy, _, _ = corr_from_2bit(probs_YY)

    norm = (Cxx ** 2 + Cyy ** 2 + Czz ** 2) ** 0.5

    return ChannelResult(name=name, Cxx=Cxx, Cyy=Cyy, Czz=Czz, norm=norm)


def run_wire_experiment(
    mode: str,
    backend_name: str,
    shots: int,
    opt_level: int,
) -> Tuple[ChannelResult, ChannelResult]:
    normal_circs = build_entwire_circuits("normal")
    wormhole_circs = build_entwire_circuits("wormhole")
    all_circs = normal_circs + wormhole_circs

    print(f"[wire] Built {len(all_circs)} circuits (3 per channel).")

    if mode == "aer":
        raw = run_aer(all_circs, shots=shots, opt_level=opt_level)
    elif mode == "ibm":
        raw = run_ibm(all_circs, backend_name=backend_name, shots=shots, opt_level=opt_level)
    else:
        raise ValueError("mode must be 'aer' or 'ibm'")

    probs_all = [probs for (probs, _depth, _twoq) in raw]
    probs_normal = probs_all[:3]
    probs_wormhole = probs_all[3:]

    normal_res = summarize_wire_channel("normal", probs_normal)
    wormhole_res = summarize_wire_channel("wormhole", probs_wormhole)
    return normal_res, wormhole_res


def print_wire_results(normal_res: ChannelResult, wormhole_res: ChannelResult) -> None:
    print("\n=== Entanglement transfer A↔R (three-qubit wire) ===")
    for res in (normal_res, wormhole_res):
        print(f"\nChannel: {res.name}")
        print(f"  C_XX ≈ {res.Cxx:+.3f}")
        print(f"  C_YY ≈ {res.Cyy:+.3f}")
        print(f"  C_ZZ ≈ {res.Czz:+.3f}")
        print(f"  ||C_AR|| ≈ {res.norm:.3f}")


# --------------------------------------------------------------------
# Transport experiment
# --------------------------------------------------------------------


def run_transport_experiment(
    mode: str,
    backend_name: str,
    geoms: List[GeometryConfig],
    channels: List[str],
    vec_indices: List[int],
    loops: List[int],
    scales: List[float],
    shots: int,
    delta_r: float,
    delta_theta: float,
    opt_level: int,
) -> List[TransportRow]:
    circuits: List[QuantumCircuit] = []
    meta: List[Tuple[str, str, int, float, int, str]] = []

    max_qubits_ibm = None
    if mode == "ibm":
        if not HAS_IBM:
            raise RuntimeError("qiskit-ibm-runtime not available")
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        max_qubits_ibm = getattr(backend, "num_qubits", None)
        print(f"[transport] Using IBM backend {backend_name}, num_qubits={max_qubits_ibm}")

    for g in geoms:
        gname = g.name
        for vid, vec in enumerate(g.vecs):
            if vec_indices and vid not in vec_indices:
                continue
            N = len(vec)
            num_qubits_ent = 1 + (2 * N + 1)
            if mode == "ibm" and max_qubits_ibm is not None:
                if num_qubits_ent > max_qubits_ibm:
                    print(
                        f"[transport] Skipping A–R ent-transfer for {gname} vec {vid}: "
                        f"needs {num_qubits_ent} qubits > backend limit"
                    )
                    continue
            for ch in channels:
                ch_l = ch.lower()
                for s in scales:
                    for L in loops:
                        for basis in ("XX", "YY", "ZZ"):
                            qc = build_AR_corr_circ(
                                vec,
                                L,
                                delta_r,
                                delta_theta,
                                s,
                                ch_l,
                                basis,
                            )
                            circuits.append(qc)
                            meta.append((gname, ch_l, vid, s, L, basis))

    if not circuits:
        print("[transport] No circuits to run.")
        return []

    print(
        f"[transport] About to run {len(circuits)} A–R circuits at {shots} shots each "
        f"({len(circuits) * shots} total shots)."
    )

    if mode == "aer":
        raw = run_aer(circuits, shots=shots, opt_level=opt_level)
    elif mode == "ibm":
        raw = run_ibm(
            circuits,
            backend_name=backend_name,
            shots=shots,
            opt_level=opt_level,
        )
    else:
        raise ValueError("mode must be 'aer' or 'ibm'")

    ar_map: Dict[Tuple[str, str, int, float, int, str], Tuple[float, int, int]] = {}

    for (geom, ch, vid, s, L, basis), (probs, depth, twoq) in zip(meta, raw):
        C, _, _ = corr_from_2bit(probs)
        ar_map[(geom, ch, vid, s, L, basis)] = (C, depth, twoq)

    rows: List[TransportRow] = []
    keys = {(m[0], m[1], m[2], m[3], m[4]) for m in meta}

    for geom, ch, vid, s, L in sorted(keys):
        def get_C(basis: str) -> Tuple[float, int, int]:
            key = (geom, ch, vid, s, L, basis)
            if key not in ar_map:
                return float("nan"), 0, 0
            C, d, tq = ar_map[key]
            return C, d, tq

        Cxx_AR, d_x, tq_x = get_C("XX")
        Cyy_AR, d_y, tq_y = get_C("YY")
        Czz_AR, d_z, tq_z = get_C("ZZ")
        depth = max(d_x, d_y, d_z)
        twoq = max(tq_x, tq_y, tq_z)

        rows.append(
            TransportRow(
                geom=geom,
                channel=ch,
                vec_id=vid,
                scale=s,
                L_loop=L,
                Cxx_AR=Cxx_AR,
                Cyy_AR=Cyy_AR,
                Czz_AR=Czz_AR,
                depth=depth,
                twoq=twoq,
            )
        )

    return rows


def print_transport_summary(rows: List[TransportRow]) -> None:
    import math as _m
    from collections import defaultdict

    if not rows:
        print("[transport] No A–R entanglement-transfer rows.")
        return

    print()
    print("Transport experiment: ||C_AR|| vs loop multiplicity L")
    print()
    header = (
        "geom  ch     vid   s       L   Cxx_AR    Cyy_AR    Czz_AR    ||C_AR||   depth   2q"
    )
    print(header)
    print("-" * len(header))

    by_gcL: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)

    for r in sorted(
        rows, key=lambda x: (x.geom, x.channel, x.vec_id, x.scale, x.L_loop)
    ):
        if any(_m.isnan(v) for v in (r.Cxx_AR, r.Cyy_AR, r.Czz_AR)):
            norm = float("nan")
        else:
            norm = (r.Cxx_AR ** 2 + r.Cyy_AR ** 2 + r.Czz_AR ** 2) ** 0.5
        print(
            f"{r.geom:5s} {r.channel:7s} {r.vec_id:3d} {r.scale:6.3f} {r.L_loop:4d} "
            f"{r.Cxx_AR:9.6f} {r.Cyy_AR:9.6f} {r.Czz_AR:9.6f} {norm:11.6f} "
            f"{r.depth:7d} {r.twoq:5d}"
        )
        if not _m.isnan(norm):
            by_gcL[(r.geom, r.channel, r.L_loop)].append(norm)

    peaks: Dict[Tuple[str, str], Tuple[int, float]] = {}
    for (geom, ch, L), norms in by_gcL.items():
        mean = sum(norms) / len(norms)
        key = (geom, ch)
        best = peaks.get(key)
        if best is None or mean > best[1]:
            peaks[key] = (L, mean)

    if peaks:
        print()
        print("Peak ||C_AR|| per (geom, channel):")
        print("geom  ch      L_peak   ||C_AR||_avg")
        print("------------------------------------")
        for (geom, ch), (L_peak, mean_norm) in sorted(peaks.items()):
            print(f"{geom:5s} {ch:7s} {L_peak:7d} {mean_norm:13.6f}")
    print()


# --------------------------------------------------------------------
# Backreaction experiment
# --------------------------------------------------------------------


def run_backreaction_experiment(
    mode: str,
    backend_name: str,
    geoms: List[GeometryConfig],
    channels: List[str],
    vec_indices: List[int],
    loops: List[int],
    scales: List[float],
    shots: int,
    delta_r: float,
    delta_theta: float,
    opt_level: int,
    signal_modes: List[str],
    signal_valve: int,
) -> List[BackreactionRow]:
    circuits: List[QuantumCircuit] = []
    meta: List[Tuple[str, str, int, float, int, str, str]] = []

    max_qubits_ibm = None
    if mode == "ibm":
        if not HAS_IBM:
            raise RuntimeError("qiskit-ibm-runtime not available")
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        max_qubits_ibm = getattr(backend, "num_qubits", None)
        print(f"[backreaction] Using IBM backend {backend_name}, num_qubits={max_qubits_ibm}")

    for g in geoms:
        gname = g.name
        for vid, vec in enumerate(g.vecs):
            if vec_indices and vid not in vec_indices:
                continue
            N = len(vec)
            num_qubits_ent = 1 + (2 * N + 1)
            if mode == "ibm" and max_qubits_ibm is not None:
                if num_qubits_ent > max_qubits_ibm:
                    print(
                        f"[backreaction] Skipping {gname} vec {vid}: "
                        f"needs {num_qubits_ent} qubits > backend limit"
                    )
                    continue
            for ch in channels:
                ch_l = ch.lower()
                for s in scales:
                    for L in loops:
                        for sig in signal_modes:
                            sig_l = sig.strip().lower()
                            if not sig_l:
                                continue
                            for basis in ("XX", "YY", "ZZ"):
                                qc = build_AR_corr_circ(
                                    vec,
                                    L,
                                    delta_r,
                                    delta_theta,
                                    s,
                                    ch_l,
                                    basis,
                                    signal_type=sig_l,
                                    signal_valve=signal_valve,
                                )
                                circuits.append(qc)
                                meta.append(
                                    (
                                        gname,
                                        ch_l,
                                        vid,
                                        s,
                                        L,
                                        sig_l,
                                        basis,
                                    )
                                )

    if not circuits:
        print("[backreaction] No circuits to run.")
        return []

    print(
        f"[backreaction] About to run {len(circuits)} circuits at {shots} shots each "
        f"({len(circuits) * shots} total shots)."
    )

    if mode == "aer":
        raw = run_aer(circuits, shots=shots, opt_level=opt_level)
    elif mode == "ibm":
        raw = run_ibm(circuits, backend_name=backend_name, shots=shots, opt_level=opt_level)
    else:
        raise ValueError("mode must be 'aer' or 'ibm'")

    ar_corr_map: Dict[
        Tuple[str, str, int, float, int, str, str],
        Tuple[float, int, int],
    ] = {}

    for key_meta, (probs, depth, twoq) in zip(meta, raw):
        C, _, _ = corr_from_2bit(probs)
        gname, ch, vid, s, L, sig, basis = key_meta
        key = (gname, ch, vid, s, L, sig, basis)
        ar_corr_map[key] = (C, depth, twoq)

    rows: List[BackreactionRow] = []

    for g in geoms:
        gname = g.name
        for vid, _vec in enumerate(g.vecs):
            if vec_indices and vid not in vec_indices:
                continue
            for ch in channels:
                ch_l = ch.lower()
                for s in scales:
                    for L in loops:
                        for sig in signal_modes:
                            sig_l = sig.strip().lower()
                            if not sig_l:
                                continue

                            def get_ar(basis: str) -> Tuple[float, int, int]:
                                key_b = (
                                    gname,
                                    ch_l,
                                    vid,
                                    s,
                                    L,
                                    sig_l,
                                    basis,
                                )
                                cb = ar_corr_map.get(key_b)
                                if cb is None:
                                    return float("nan"), 0, 0
                                C, d, tq = cb
                                return C, d, tq

                            Cxx_AR, d_x, tq_x = get_ar("XX")
                            Cyy_AR, d_y, tq_y = get_ar("YY")
                            Czz_AR, d_z, tq_z = get_ar("ZZ")
                            depth = max(d_x, d_y, d_z)
                            twoq = max(tq_x, tq_y, tq_z)

                            rows.append(
                                BackreactionRow(
                                    geom=gname,
                                    channel=ch_l,
                                    vec_id=vid,
                                    scale=s,
                                    L_loop=L,
                                    signal=sig_l,
                                    Cxx_AR=Cxx_AR,
                                    Cyy_AR=Cyy_AR,
                                    Czz_AR=Czz_AR,
                                    depth=depth,
                                    twoq=twoq,
                                )
                            )

    return rows


def print_backreaction_summary(rows: List[BackreactionRow]) -> None:
    import math as _m
    from collections import defaultdict

    if not rows:
        print("[backreaction] No results.")
        return

    print()
    print("Backreaction: valve signals vs A–R entanglement transfer")
    print()
    header = (
        "geom  ch     vid   s       L   signal   Cxx_AR    Cyy_AR    Czz_AR    ||C_AR||   depth   2q"
    )
    print(header)
    print("-" * len(header))

    by_key: Dict[
        Tuple[str, str, int, float, int],
        Dict[str, float],
    ] = defaultdict(dict)

    for r in sorted(
        rows,
        key=lambda x: (x.geom, x.channel, x.vec_id, x.scale, x.L_loop, x.signal),
    ):
        if any(_m.isnan(v) for v in (r.Cxx_AR, r.Cyy_AR, r.Czz_AR)):
            norm = float("nan")
        else:
            norm = (r.Cxx_AR ** 2 + r.Cyy_AR ** 2 + r.Czz_AR ** 2) ** 0.5

        print(
            f"{r.geom:5s} {r.channel:7s} {r.vec_id:3d} {r.scale:6.3f} {r.L_loop:4d} "
            f"{r.signal:7s} {r.Cxx_AR:9.6f} {r.Cyy_AR:9.6f} {r.Czz_AR:9.6f} "
            f"{norm:11.6f} {r.depth:7d} {r.twoq:5d}"
        )

        key = (r.geom, r.channel, r.vec_id, r.scale, r.L_loop)
        by_key[key][r.signal] = norm

    print()
    print("Signal effect per (geom, ch, vec, s, L): ||C_AR|| for each signal type")
    print("geom  ch     vid   s       L   ||C||_none   ||C||_neg   ||C||_pos0   ||C||_pos1")
    print("-------------------------------------------------------------------------------")

    for key, sig_map in sorted(by_key.items()):
        geom, ch, vid, s, L = key
        n_none = sig_map.get("none", float("nan"))
        n_neg = sig_map.get("neg", float("nan"))
        n_pos0 = sig_map.get("pos0", float("nan"))
        n_pos1 = sig_map.get("pos1", float("nan"))
        print(
            f"{geom:5s} {ch:7s} {vid:3d} {s:6.3f} {L:4d} "
            f"{n_none:11.6f} {n_neg:11.6f} {n_pos0:11.6f} {n_pos1:11.6f}"
        )
    print()


# --------------------------------------------------------------------
# Diary experiment
# --------------------------------------------------------------------


def build_diary_circ(theta: float, channel: str, basis: str) -> QuantumCircuit:
    qr = QuantumRegister(4, "q")
    qc = QuantumCircuit(qr)

    qL, qE0, qE1, qR = qr[0], qr[1], qr[2], qr[3]

    qc.ry(2.0 * theta, qL)

    qc.cx(qL, qE0)
    qc.h(qL)
    qc.cx(qE0, qE1)

    ch = channel.lower()
    if ch == "wormhole":
        qc.swap(qL, qR)
        qc.cx(qE0, qE1)
        qc.h(qR)
        qc.cx(qR, qE0)
    elif ch == "local":
        qc.h(qR)
        qc.z(qR)
    else:
        raise ValueError("channel must be 'wormhole' or 'local'")

    cr = ClassicalRegister(1, "c")
    qc.add_register(cr)

    _basis_rotate(qc, qR, basis.upper())
    qc.measure(qR, cr[0])
    return qc


def run_diary_experiment(
    mode: str,
    backend_name: str,
    thetas: List[float],
    channels: List[str],
    shots: int,
    opt_level: int,
) -> List[DiaryRow]:
    circuits: List[QuantumCircuit] = []
    meta: List[Tuple[str, float, str]] = []

    for ch in channels:
        ch_l = ch.lower()
        for theta in thetas:
            for basis in ("X", "Y", "Z"):
                qc = build_diary_circ(theta, ch_l, basis)
                circuits.append(qc)
                meta.append((ch_l, float(theta), basis))

    if not circuits:
        print("[diary] No circuits to run.")
        return []

    print(
        f"[diary] About to run {len(circuits)} diary circuits at {shots} shots each "
        f"({len(circuits) * shots} total shots)."
    )

    if mode == "aer":
        raw = run_aer(circuits, shots=shots, opt_level=opt_level)
    elif mode == "ibm":
        raw = run_ibm(circuits, backend_name=backend_name, shots=shots, opt_level=opt_level)
    else:
        raise ValueError("mode must be 'aer' or 'ibm'")

    from collections import defaultdict

    acc: Dict[Tuple[str, float], Dict[str, float]] = defaultdict(
        lambda: {"mx": 0.0, "my": 0.0, "mz": 0.0, "depth": 0.0, "twoq": 0.0}
    )

    for (ch, theta, basis), (probs, depth, twoq) in zip(meta, raw):
        key = (ch, theta)
        entry = acc[key]
        m = 0.0
        for bits, p in probs.items():
            if len(bits) != 1:
                continue
            v = 1.0 if bits[-1] == "0" else -1.0
            m += v * p
        if basis == "X":
            entry["mx"] = m
        elif basis == "Y":
            entry["my"] = m
        elif basis == "Z":
            entry["mz"] = m
        entry["depth"] = max(entry["depth"], float(depth))
        entry["twoq"] = max(entry["twoq"], float(twoq))

    rows: List[DiaryRow] = []
    for (ch, theta), entry in sorted(acc.items()):
        mx = entry["mx"]
        my = entry["my"]
        mz = entry["mz"]
        n_tx = math.sin(2.0 * theta)
        n_ty = 0.0
        n_tz = math.cos(2.0 * theta)
        dot = mx * n_tx + my * n_ty + mz * n_tz
        fidelity = 0.5 * (1.0 + dot)
        rows.append(
            DiaryRow(
                channel=ch,
                theta=theta,
                mx=mx,
                my=my,
                mz=mz,
                fidelity=fidelity,
                depth=int(entry["depth"]),
                twoq=int(entry["twoq"]),
            )
        )

    return rows


def print_diary_summary(rows: List[DiaryRow]) -> None:
    if not rows:
        print("[diary] No results.")
        return

    print()
    print("Diary experiment: fidelity of R vs initial diary state")
    print()
    header = (
        "ch     theta      mX        mY        mZ   F(R|diary)   depth   2q"
    )
    print(header)
    print("-" * len(header))

    for r in sorted(rows, key=lambda x: (x.channel, x.theta)):
        print(
            f"{r.channel:7s} {r.theta:7.3f} "
            f"{r.mx:9.6f} {r.my:9.6f} {r.mz:9.6f} "
            f"{r.fidelity:11.6f} {r.depth:7d} {r.twoq:5d}"
        )

    from collections import defaultdict

    by_ch: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_ch[r.channel].append(r.fidelity)

    print()
    print("Average diary fidelity by channel:")
    for ch, vals in sorted(by_ch.items()):
        avg = sum(vals) / len(vals)
        print(f"  {ch:7s}: ⟨F⟩ ≈ {avg:.6f}")
    print()


# --------------------------------------------------------------------
# One-shot diary state probe (for agent teleport)
# --------------------------------------------------------------------


def run_diary_state(
    mode: str,
    backend_name: str,
    theta: float,
    channel: str,
    shots: int,
    opt_level: int,
) -> Tuple[float, float, float, int, int]:
    circuits: List[QuantumCircuit] = []
    bases: List[str] = ["X", "Y", "Z"]

    ch_l = channel.strip().lower()
    if ch_l not in ("wormhole", "local"):
        raise ValueError("channel must be 'wormhole' or 'local'")

    for basis in bases:
        qc = build_diary_circ(theta, ch_l, basis)
        circuits.append(qc)

    if mode == "aer":
        raw = run_aer(circuits, shots=shots, opt_level=opt_level)
    elif mode == "ibm":
        raw = run_ibm(circuits, backend_name=backend_name, shots=shots, opt_level=opt_level)
    else:
        raise ValueError("mode must be 'aer' or 'ibm'")

    mx = 0.0
    my = 0.0
    mz = 0.0
    max_depth = 0
    max_twoq = 0

    for (probs, depth, twoq), basis in zip(raw, bases):
        m = 0.0
        for bits, p in probs.items():
            if len(bits) != 1:
                continue
            v = 1.0 if bits[-1] == "0" else -1.0
            m += v * p
        if basis == "X":
            mx = m
        elif basis == "Y":
            my = m
        elif basis == "Z":
            mz = m
        max_depth = max(max_depth, int(depth))
        max_twoq = max(max_twoq, int(twoq))

    return mx, my, mz, max_depth, max_twoq


# --------------------------------------------------------------------
# Analytic headlines
# --------------------------------------------------------------------


def print_analytic_headlines(
    normal_res: ChannelResult,
    wormhole_res: ChannelResult,
    transport_rows: List[TransportRow],
    back_rows: List[BackreactionRow],
    diary_rows: List[DiaryRow],
) -> None:
    import math as _m
    from collections import defaultdict as _dd

    print()
    print("Analytic headlines")
    print("------------------")

    base = abs(normal_res.norm)
    if base < 1e-6:
        print(
            f"[headline] wire: wormhole ||C_AR|| ≈ {wormhole_res.norm:.3f} "
            f"vs normal ≈ {normal_res.norm:.3f} (normal effectively zero)."
        )
    else:
        ratio = wormhole_res.norm / base
        print(
            f"[headline] wire: wormhole ||C_AR|| ≈ {wormhole_res.norm:.3f} "
            f"vs normal ≈ {normal_res.norm:.3f} (≈{ratio:.1f}× larger)."
        )

    if transport_rows:
        peak_map: Dict[Tuple[str, str, int], Tuple[int, float]] = {}
        for r in transport_rows:
            if any(_m.isnan(v) for v in (r.Cxx_AR, r.Cyy_AR, r.Czz_AR)):
                continue
            norm = (r.Cxx_AR ** 2 + r.Cyy_AR ** 2 + r.Czz_AR ** 2) ** 0.5
            key = (r.geom, r.channel, r.vec_id)
            best = peak_map.get(key)
            if best is None or norm > best[1]:
                peak_map[key] = (r.L_loop, norm)

        pairs = {(g, v) for (g, ch, v) in peak_map.keys()}
        for geom, vid in sorted(pairs):
            key_w = (geom, "wormhole", vid)
            key_l = (geom, "local", vid)
            if key_w not in peak_map or key_l not in peak_map:
                continue
            Lw, nw = peak_map[key_w]
            Ll, nl = peak_map[key_l]
            if nl < 1e-6:
                print(
                    f"[headline] transport {geom}, vec {vid}: wormhole peaks at L={Lw} "
                    f"with ||C_AR|| ≈ {nw:.3f}; local peak at L={Ll} is ≈{nl:.3f} "
                    f"(local effectively zero)."
                )
            else:
                ratio = nw / nl
                print(
                    f"[headline] transport {geom}, vec {vid}: wormhole peaks at L={Lw} "
                    f"with ||C_AR|| ≈ {nw:.3f}; local peak at L={Ll} is ≈{nl:.3f} "
                    f"(wormhole ≈{ratio:.1f}× stronger)."
                )

    if back_rows:
        signal_map: Dict[
            Tuple[str, str, int, float, int],
            Dict[str, float],
        ] = _dd(dict)

        for r in back_rows:
            key = (r.geom, r.channel, r.vec_id, r.scale, r.L_loop)
            if any(_m.isnan(v) for v in (r.Cxx_AR, r.Cyy_AR, r.Czz_AR)):
                norm = float("nan")
            else:
                norm = (r.Cxx_AR ** 2 + r.Cyy_AR ** 2 + r.Czz_AR ** 2) ** 0.5
            signal_map[key][r.signal] = norm

        for key, norms_w in sorted(signal_map.items()):
            geom, ch, vid, scale, L = key
            if ch != "wormhole":
                continue
            local_norms = signal_map.get((geom, "local", vid, scale, L), {})
            baseW = norms_w.get("none", float("nan"))
            if _m.isnan(baseW):
                continue
            pos1 = norms_w.get("pos1")
            pos0 = norms_w.get("pos0")
            neg = norms_w.get("neg")

            line = (
                f"[headline] backreaction {geom}, wormhole, vec {vid}, L={L}: "
                f"baseline ||C_AR|| ≈ {baseW:.3f}"
            )
            if pos1 is not None and not _m.isnan(pos1):
                if baseW > 1e-6:
                    r1 = pos1 / baseW
                    line += f"; pos1 ≈ {pos1:.3f} (×{r1:.1f})"
                else:
                    line += f"; pos1 ≈ {pos1:.3f}"
            if pos0 is not None and not _m.isnan(pos0):
                if baseW > 1e-6:
                    r0 = pos0 / baseW
                    line += f"; pos0 ≈ {pos0:.3f} (×{r0:.2f})"
                else:
                    line += f"; pos0 ≈ {pos0:.3f}"
            if neg is not None and not _m.isnan(neg):
                if baseW > 1e-6:
                    rn = neg / baseW
                    line += f"; neg ≈ {neg:.3f} (×{rn:.2f})"
                else:
                    line += f"; neg ≈ {neg:.3f}"

            if local_norms:
                finite_local = [v for v in local_norms.values() if not _m.isnan(v)]
                if finite_local:
                    minL = min(finite_local)
                    maxL = max(finite_local)
                    line += f"; local across signals in [{minL:.3f}, {maxL:.3f}]"

            print(line)

    if diary_rows:
        from collections import defaultdict

        by_ch: Dict[str, List[float]] = defaultdict(list)
        for r in diary_rows:
            by_ch[r.channel].append(r.fidelity)

        for ch, vals in sorted(by_ch.items()):
            avg = sum(vals) / len(vals)
            print(
                f"[headline] diary {ch}: ⟨F(R|diary)⟩ across θ ≈ {avg:.3f}"
            )


# --------------------------------------------------------------------
# RLQMF: bandit over parameter space
# --------------------------------------------------------------------


@dataclass
class RLAction:
    objective: str
    L: int
    theta: float
    signal: str


@dataclass
class RLStepResult:
    action_id: int
    action: RLAction
    reward: float


@dataclass
class ActionEnvResult:
    reward: float
    metrics: Dict[str, float]


def eval_action(
    action: RLAction,
    mode: str,
    backend_name: str,
    geoms: List[GeometryConfig],
    channels: List[str],
    vec_indices: List[int],
    scale: float,
    delta_r: float,
    delta_theta: float,
    shots: int,
    opt_level: int,
    cost_weight: float,
) -> ActionEnvResult:
    import math as _m

    metrics: Dict[str, float] = {}

    if action.objective == "diary":
        rows = run_diary_experiment(
            mode=mode,
            backend_name=backend_name,
            thetas=[action.theta],
            channels=["local", "wormhole"],
            shots=shots,
            opt_level=opt_level,
        )
        f_local = None
        f_worm = None
        total_twoq = 0.0
        for r in rows:
            total_twoq += r.twoq
            if r.channel == "local":
                f_local = r.fidelity
            elif r.channel == "wormhole":
                f_worm = r.fidelity
        if f_local is None or f_worm is None:
            return ActionEnvResult(reward=0.0, metrics={})
        avg_twoq = total_twoq / len(rows) if rows else 0.0
        raw = f_worm - f_local
        reward = raw - cost_weight * avg_twoq
        metrics["raw"] = raw
        metrics["f_local"] = f_local
        metrics["f_worm"] = f_worm
        metrics["twoq_avg"] = avg_twoq
        return ActionEnvResult(reward=reward, metrics=metrics)

    if action.objective == "transport":
        rows = run_transport_experiment(
            mode=mode,
            backend_name=backend_name,
            geoms=geoms,
            channels=channels,
            vec_indices=vec_indices,
            loops=[action.L],
            scales=[scale],
            shots=shots,
            delta_r=delta_r,
            delta_theta=delta_theta,
            opt_level=opt_level,
        )
        worm = None
        loc = None
        total_twoq = 0.0
        for r in rows:
            total_twoq += r.twoq
            norm = (r.Cxx_AR ** 2 + r.Cyy_AR ** 2 + r.Czz_AR ** 2) ** 0.5
            if r.channel == "wormhole":
                worm = norm if worm is None else max(worm, norm)
            elif r.channel == "local":
                loc = norm if loc is None else max(loc, norm)
        if worm is None or loc is None:
            return ActionEnvResult(reward=0.0, metrics={})
        avg_twoq = total_twoq / len(rows) if rows else 0.0
        raw = worm - loc
        reward = raw - cost_weight * avg_twoq
        metrics["raw"] = raw
        metrics["norm_wormhole_max"] = worm
        metrics["norm_local_max"] = loc
        metrics["twoq_avg"] = avg_twoq
        return ActionEnvResult(reward=reward, metrics=metrics)

    if action.objective == "backreaction":
        rows = run_backreaction_experiment(
            mode=mode,
            backend_name=backend_name,
            geoms=geoms,
            channels=["wormhole"],
            vec_indices=vec_indices,
            loops=[action.L],
            scales=[scale],
            shots=shots,
            delta_r=delta_r,
            delta_theta=delta_theta,
            opt_level=opt_level,
            signal_modes=["pos0", "pos1"],
            signal_valve=1,
        )
        pos0 = None
        pos1 = None
        total_twoq = 0.0
        for r in rows:
            total_twoq += r.twoq
            norm = (r.Cxx_AR ** 2 + r.Cyy_AR ** 2 + r.Czz_AR ** 2) ** 0.5
            if r.signal == "pos0":
                pos0 = norm
            elif r.signal == "pos1":
                pos1 = norm
        if pos0 is None or pos1 is None:
            return ActionEnvResult(reward=0.0, metrics={})
        avg_twoq = total_twoq / len(rows) if rows else 0.0
        raw = pos1 - pos0
        reward = raw - cost_weight * avg_twoq
        metrics["raw"] = raw
        metrics["C_pos0"] = pos0
        metrics["C_pos1"] = pos1
        metrics["twoq_avg"] = avg_twoq
        return ActionEnvResult(reward=reward, metrics=metrics)

    if action.objective == "back_seq":
        seq = [s.strip() for s in action.signal.split(",") if s.strip()]
        if len(seq) != 3:
            return ActionEnvResult(reward=0.0, metrics={})

        uniq_signals = sorted(set(seq))
        rows = run_backreaction_experiment(
            mode=mode,
            backend_name=backend_name,
            geoms=geoms,
            channels=["wormhole"],
            vec_indices=vec_indices,
            loops=[action.L],
            scales=[scale],
            shots=shots,
            delta_r=delta_r,
            delta_theta=delta_theta,
            opt_level=opt_level,
            signal_modes=uniq_signals,
            signal_valve=1,
        )
        norms: Dict[str, float] = {}
        total_twoq = 0.0
        for r in rows:
            total_twoq += r.twoq
            norm = (r.Cxx_AR ** 2 + r.Cyy_AR ** 2 + r.Czz_AR ** 2) ** 0.5
            norms[r.signal] = norm
        if any(s not in norms for s in seq):
            return ActionEnvResult(reward=0.0, metrics={})
        C0 = norms[seq[0]]
        C1 = norms[seq[1]]
        C2 = norms[seq[2]]
        raw = C0 - C1 + C2
        avg_twoq = total_twoq / len(rows) if rows else 0.0
        reward = raw - cost_weight * avg_twoq
        metrics["raw"] = raw
        for s, C in norms.items():
            metrics[f"C_{s}"] = C
        metrics["C0"] = C0
        metrics["C1"] = C1
        metrics["C2"] = C2
        metrics["twoq_avg"] = avg_twoq
        return ActionEnvResult(reward=reward, metrics=metrics)

    return ActionEnvResult(reward=0.0, metrics={})


def rlqmf_bandit(
    objective: str,
    mode: str,
    backend_name: str,
    geoms: List[GeometryConfig],
    channels: List[str],
    vec_indices: List[int],
    L_space: List[int],
    theta_space: List[float],
    shot_budget: int,
    epsilon: float,
    delta_r: float,
    delta_theta: float,
    opt_level: int,
    scale: float,
    cost_weight: float,
) -> Tuple[List[RLStepResult], Dict[int, float], List[RLAction]]:
    actions: List[RLAction] = []

    if objective == "diary":
        for theta in theta_space:
            actions.append(RLAction(objective="diary", L=0, theta=theta, signal="none"))
    elif objective == "transport":
        for L in L_space:
            actions.append(RLAction(objective="transport", L=L, theta=0.25, signal="none"))
    elif objective == "backreaction":
        for L in L_space:
            actions.append(RLAction(objective="backreaction", L=L, theta=0.25, signal="none"))
    elif objective == "back_seq":
        seq_space = [
            "pos1,pos0,pos1",
            "pos1,none,pos1",
            "none,pos1,none",
            "pos0,pos1,pos0",
            "pos1,pos1,pos1",
        ]
        for L in L_space:
            for seq in seq_space:
                actions.append(RLAction(objective="back_seq", L=L, theta=0.25, signal=seq))
    else:
        raise ValueError("objective must be one of: diary, transport, backreaction, back_seq")

    num_actions = len(actions)
    if num_actions == 0:
        raise RuntimeError("No RL actions constructed.")

    Q: Dict[int, float] = {i: 0.0 for i in range(num_actions)}
    N: Dict[int, int] = {i: 0 for i in range(num_actions)}

    episodes = max(1, shot_budget // 4096)

    results: List[RLStepResult] = []

    print(f"[rlqmf] objective={objective}, |actions|={num_actions}, episodes={episodes}, epsilon={epsilon:.2f}, λ={cost_weight:.3f}")

    for ep in range(episodes):
        if random.random() < epsilon:
            a_id = random.randrange(num_actions)
        else:
            a_id = max(Q.keys(), key=lambda k: Q[k])

        action = actions[a_id]

        env_res = eval_action(
            action=action,
            mode=mode,
            backend_name=backend_name,
            geoms=geoms,
            channels=channels,
            vec_indices=vec_indices,
            scale=scale,
            delta_r=delta_r,
            delta_theta=delta_theta,
            shots=4096,
            opt_level=opt_level,
            cost_weight=cost_weight,
        )
        reward = env_res.reward

        N[a_id] += 1
        alpha = 1.0 / N[a_id]
        Q[a_id] = Q[a_id] + alpha * (reward - Q[a_id])

        results.append(RLStepResult(action_id=a_id, action=action, reward=reward))

        print(
            f"[rlqmf] ep {ep+1:3d}/{episodes:3d}: a_id={a_id}, obj={action.objective}, "
            f"L={action.L}, theta={action.theta:.3f}, signal={action.signal}, "
            f"reward={reward:.4f}, Q={Q[a_id]:.4f}"
        )

    return results, Q, actions


def print_rlqmf_summary(actions: List[RLAction], Q: Dict[int, float]) -> None:
    if not Q:
        print("[rlqmf] No Q-values.")
        return
    best_id = max(Q.keys(), key=lambda k: Q[k])
    best = actions[best_id]
    print()
    print("[rlqmf] Learned value estimates (Q):")
    for i, a in enumerate(actions):
        val = Q.get(i, 0.0)
        print(
            f"  id={i:2d} obj={a.objective:10s} L={a.L} theta={a.theta:.3f} "
            f"signal={a.signal:15s}  Q≈{val:.4f}"
        )
    print()
    print(
        f"[rlqmf] Best action: id={best_id}, objective={best.objective}, "
        f"L={best.L}, theta={best.theta:.3f}, signal={best.signal}, Q≈{Q[best_id]:.4f}"
    )


# --------------------------------------------------------------------
# Teleported agent experiment: control vs wormhole-lived confidence
# --------------------------------------------------------------------


@dataclass
class AgentState:
    name: str
    Q: Dict[int, float]
    N: Dict[int, int]
    confidence: float


def _confidence_to_theta(conf: float) -> float:
    conf_clamped = max(0.0, min(1.0, conf))
    z = 2.0 * conf_clamped - 1.0
    z = max(-1.0, min(1.0, z))
    return 0.5 * math.acos(z)


def _theta_result_to_conf(mz: float) -> float:
    z = max(-1.0, min(1.0, mz))
    return 0.5 * (z + 1.0)


def _update_confidence(conf_before: float, reward: float, eta: float = 0.1) -> float:
    target = 1.0 / (1.0 + math.exp(-reward))
    new_conf = (1.0 - eta) * conf_before + eta * target
    return max(0.0, min(1.0, new_conf))


def env_metric_fields_for_objective(objective: str) -> List[str]:
    if objective == "diary":
        return ["raw", "f_local", "f_worm", "twoq_avg"]
    if objective == "transport":
        return ["raw", "norm_wormhole_max", "norm_local_max", "twoq_avg"]
    if objective == "backreaction":
        return ["raw", "C_pos0", "C_pos1", "twoq_avg"]
    if objective == "back_seq":
        return ["raw", "C_none", "C_neg", "C_pos0", "C_pos1", "C0", "C1", "C2", "twoq_avg"]
    return []


def _build_rl_actions_for_objective(
    objective: str,
    L_space: List[int],
    theta_space: List[float],
) -> List[RLAction]:
    actions: List[RLAction] = []
    if objective == "diary":
        for theta in theta_space:
            actions.append(RLAction(objective="diary", L=0, theta=theta, signal="none"))
    elif objective == "transport":
        for L in L_space:
            actions.append(RLAction(objective="transport", L=L, theta=0.25, signal="none"))
    elif objective == "backreaction":
        for L in L_space:
            actions.append(RLAction(objective="backreaction", L=L, theta=0.25, signal="none"))
    elif objective == "back_seq":
        seq_space = [
            "pos1,pos0,pos1",
            "pos1,none,pos1",
            "none,pos1,none",
            "pos0,pos1,pos0",
            "pos1,pos1,pos1",
        ]
        for L in L_space:
            for seq in seq_space:
                actions.append(RLAction(objective="back_seq", L=L, theta=0.25, signal=seq))
    else:
        raise ValueError("objective must be one of: diary, transport, backreaction, back_seq")
    return actions


def run_teleported_agent_experiment(
    objective: str,
    mode: str,
    backend_name: str,
    geoms: List[GeometryConfig],
    channels: List[str],
    vec_indices: List[int],
    L_space: List[int],
    theta_space: List[float],
    shot_budget: int,
    epsilon0: float,
    delta_r: float,
    delta_theta: float,
    opt_level: int,
    scale: float,
    cost_weight: float,
    diary_shots: int,
    log_csv: str,
) -> None:
    actions = _build_rl_actions_for_objective(objective, L_space, theta_space)
    num_actions = len(actions)
    if num_actions == 0:
        raise RuntimeError("No RL actions constructed for teleported agent experiment.")

    episodes = max(1, shot_budget // 4096)
    shots_env = 4096

    eps0 = max(0.0, min(1.0, epsilon0))
    conf0 = max(0.0, min(1.0, 1.0 - eps0))

    agents = [
        AgentState(
            name="control",
            Q={i: 0.0 for i in range(num_actions)},
            N={i: 0 for i in range(num_actions)},
            confidence=conf0,
        ),
        AgentState(
            name="wormhole",
            Q={i: 0.0 for i in range(num_actions)},
            N={i: 0 for i in range(num_actions)},
            confidence=conf0,
        ),
    ]

    diary_channel = "wormhole"

    print(
        f"[agent] objective={objective}, |actions|={num_actions}, "
        f"episodes={episodes}, epsilon0={epsilon0:.3f}, "
        f"diary_channel={diary_channel}, diary_shots={diary_shots}"
    )
    print(f"[agent] Logging per-episode data to {log_csv!r}")

    env_fields = env_metric_fields_for_objective(objective)

    fieldnames = [
        "episode",
        "agent",
        "action_id",
        "objective",
        "L",
        "theta_action",
        "signal",
        "reward_env",
        "epsilon_used",
        "confidence_before",
        "confidence_updated",
        "confidence_after",
        "theta_conf_in",
        "theta_conf_out",
        "mx",
        "my",
        "mz",
        "depth_diary",
        "twoq_diary",
    ]
    for field in env_fields:
        fieldnames.append(f"env_{field}")
    for i in range(num_actions):
        fieldnames.append(f"Q_before_{i}")
    for i in range(num_actions):
        fieldnames.append(f"Q_after_{i}")

    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(1, episodes + 1):
            for agent in agents:
                Q = agent.Q
                N = agent.N
                conf_before = agent.confidence

                epsilon_agent = max(0.01, min(0.9, 1.0 - conf_before))

                if random.random() < epsilon_agent:
                    a_id = random.randrange(num_actions)
                else:
                    maxQ = max(Q.values())
                    best_ids = [i for i, v in Q.items() if v == maxQ]
                    a_id = min(best_ids)

                action = actions[a_id]

                Q_before = [Q[i] for i in range(num_actions)]

                env_res = eval_action(
                    action=action,
                    mode=mode,
                    backend_name=backend_name,
                    geoms=geoms,
                    channels=channels,
                    vec_indices=vec_indices,
                    scale=scale,
                    delta_r=delta_r,
                    delta_theta=delta_theta,
                    shots=shots_env,
                    opt_level=opt_level,
                    cost_weight=cost_weight,
                )
                reward_env = env_res.reward

                N[a_id] += 1
                alpha = 1.0 / N[a_id]
                Q[a_id] = Q[a_id] + alpha * (reward_env - Q[a_id])
                Q_after = [Q[i] for i in range(num_actions)]

                conf_updated = _update_confidence(conf_before, reward_env)

                theta_in = 0.0
                theta_out = 0.0
                mx = my = mz = 0.0
                depth_diary = 0
                twoq_diary = 0

                if agent.name == "wormhole":
                    theta_in = _confidence_to_theta(conf_updated)
                    mx, my, mz, depth_diary, twoq_diary = run_diary_state(
                        mode=mode,
                        backend_name=backend_name,
                        theta=theta_in,
                        channel=diary_channel,
                        shots=diary_shots,
                        opt_level=opt_level,
                    )
                    conf_after = _theta_result_to_conf(mz)
                    theta_out = _confidence_to_theta(conf_after)
                else:
                    conf_after = conf_updated
                    theta_in = _confidence_to_theta(conf_updated)
                    theta_out = theta_in

                agent.confidence = conf_after

                row: Dict[str, object] = {
                    "episode": ep,
                    "agent": agent.name,
                    "action_id": a_id,
                    "objective": action.objective,
                    "L": action.L,
                    "theta_action": action.theta,
                    "signal": action.signal,
                    "reward_env": reward_env,
                    "epsilon_used": epsilon_agent,
                    "confidence_before": conf_before,
                    "confidence_updated": conf_updated,
                    "confidence_after": conf_after,
                    "theta_conf_in": theta_in,
                    "theta_conf_out": theta_out,
                    "mx": mx,
                    "my": my,
                    "mz": mz,
                    "depth_diary": depth_diary,
                    "twoq_diary": twoq_diary,
                }
                for field in env_fields:
                    row[f"env_{field}"] = env_res.metrics.get(field)
                for i in range(num_actions):
                    row[f"Q_before_{i}"] = Q_before[i]
                    row[f"Q_after_{i}"] = Q_after[i]

                writer.writerow(row)

            ctrl = next(a for a in agents if a.name == "control")
            wh = next(a for a in agents if a.name == "wormhole")
            best_ctrl = max(ctrl.Q.values())
            best_wh = max(wh.Q.values())
            print(
                f"[agent] ep {ep:3d}/{episodes:3d}: "
                f"ctrl_best≈{best_ctrl:.4f}, wh_best≈{best_wh:.4f}, "
                f"wh_conf≈{wh.confidence:.3f}"
            )


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vybn wormhole lab: wire, transport, backreaction, diary, analyze, rlqmf, agent.",
    )

    backend_parent = argparse.ArgumentParser(add_help=False)
    backend_parent.add_argument(
        "--mode",
        choices=["aer", "ibm"],
        default="aer",
        help="Backend mode: 'aer' (local) or 'ibm' (SamplerV2 backend).",
    )
    backend_parent.add_argument(
        "--backend",
        type=str,
        default="ibm_fez",
        help="IBM backend name when using --mode ibm.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_wire = subparsers.add_parser(
        "wire",
        parents=[backend_parent],
        help="Three-qubit wormhole vs normal 'wire' sanity experiment.",
    )
    p_wire.add_argument("--shots", type=int, default=4096)
    p_wire.add_argument("--opt-level", type=int, default=2)

    p_trans = subparsers.add_parser(
        "transport",
        parents=[backend_parent],
        help="L–R chain entanglement transport (traversability window).",
    )
    p_trans.add_argument("--models", type=str, default="dual2")
    p_trans.add_argument("--channels", type=str, default="wormhole,local")
    p_trans.add_argument("--vec-indices", type=int, nargs="+", default=[0])
    p_trans.add_argument("--loops", type=int, nargs="+", default=[0, 1, 2])
    p_trans.add_argument("--area-scales", type=float, nargs="+", default=[0.5])
    p_trans.add_argument("--shots", type=int, default=4096)
    p_trans.add_argument("--delta-r", type=float, default=0.2)
    p_trans.add_argument("--delta-theta", type=float, default=0.3)
    p_trans.add_argument("--opt-level", type=int, default=2)

    p_back = subparsers.add_parser(
        "backreaction",
        parents=[backend_parent],
        help="Valve-signal backreaction experiment.",
    )
    p_back.add_argument("--models", type=str, default="dual2")
    p_back.add_argument("--channels", type=str, default="wormhole,local")
    p_back.add_argument("--vec-indices", type=int, nargs="+", default=[0])
    p_back.add_argument("--loops", type=int, nargs="+", default=[1])
    p_back.add_argument("--area-scales", type=float, nargs="+", default=[0.5])
    p_back.add_argument("--shots", type=int, default=4096)
    p_back.add_argument("--delta-r", type=float, default=0.2)
    p_back.add_argument("--delta-theta", type=float, default=0.3)
    p_back.add_argument("--opt-level", type=int, default=2)
    p_back.add_argument("--signals", type=str, default="none,neg,pos0,pos1")
    p_back.add_argument("--signal-valve", type=int, default=1)

    p_diary = subparsers.add_parser(
        "diary",
        parents=[backend_parent],
        help="Black-hole diary toy: scramble + decode via wormhole vs local.",
    )
    p_diary.add_argument("--thetas", type=float, nargs="+", default=[0.25])
    p_diary.add_argument("--shots", type=int, default=4096)
    p_diary.add_argument("--opt-level", type=int, default=2)

    p_an = subparsers.add_parser(
        "analyze",
        parents=[backend_parent],
        help="Run wire + transport + backreaction + diary and print analytic headlines.",
    )
    p_an.add_argument("--models", type=str, default="dual2")
    p_an.add_argument("--channels", type=str, default="wormhole,local")
    p_an.add_argument("--vec-indices", type=int, nargs="+", default=[0])
    p_an.add_argument("--loops", type=int, nargs="+", default=[0, 1, 2])
    p_an.add_argument("--area-scales", type=float, nargs="+", default=[0.5])
    p_an.add_argument("--thetas", type=float, nargs="+", default=[0.25])
    p_an.add_argument("--shots", type=int, default=4096)
    p_an.add_argument("--delta-r", type=float, default=0.2)
    p_an.add_argument("--delta-theta", type=float, default=0.3)
    p_an.add_argument("--opt-level", type=int, default=2)
    p_an.add_argument("--signals", type=str, default="none,neg,pos0,pos1")
    p_an.add_argument("--signal-valve", type=int, default=1)

    p_rl = subparsers.add_parser(
        "rlqmf",
        parents=[backend_parent],
        help="Reinforcement learning from quantum mechanical feedback (bandit).",
    )
    p_rl.add_argument(
        "--objective",
        choices=["diary", "transport", "backreaction", "back_seq"],
        default="diary",
    )
    p_rl.add_argument("--models", type=str, default="dual2")
    p_rl.add_argument("--channels", type=str, default="wormhole,local")
    p_rl.add_argument("--vec-indices", type=int, nargs="+", default=[0])
    p_rl.add_argument("--L-space", type=int, nargs="+", default=[0, 1, 2])
    p_rl.add_argument("--theta-space", type=float, nargs="+", default=[0.25])
    p_rl.add_argument("--shot-budget", type=int, default=4096 * 20)
    p_rl.add_argument("--epsilon", type=float, default=0.2)
    p_rl.add_argument("--delta-r", type=float, default=0.2)
    p_rl.add_argument("--delta-theta", type=float, default=0.3)
    p_rl.add_argument("--area-scale", type=float, default=0.5)
    p_rl.add_argument("--opt-level", type=int, default=2)
    p_rl.add_argument(
        "--cost-weight",
        type=float,
        default=0.0,
        help="Penalty λ on average two-qubit gate count in the reward.",
    )

    p_agent = subparsers.add_parser(
        "agent",
        parents=[backend_parent],
        help="Paired bandit agents (control vs wormhole-teleported confidence).",
    )
    p_agent.add_argument(
        "--objective",
        choices=["diary", "transport", "backreaction", "back_seq"],
        default="back_seq",
    )
    p_agent.add_argument("--models", type=str, default="dual2")
    p_agent.add_argument("--channels", type=str, default="wormhole,local")
    p_agent.add_argument("--vec-indices", type=int, nargs="+", default=[0])
    p_agent.add_argument("--L-space", type=int, nargs="+", default=[1])
    p_agent.add_argument("--theta-space", type=float, nargs="+", default=[0.25])
    p_agent.add_argument("--shot-budget", type=int, default=4096 * 20)
    p_agent.add_argument("--epsilon", type=float, default=0.2)
    p_agent.add_argument("--delta-r", type=float, default=0.2)
    p_agent.add_argument("--delta-theta", type=float, default=0.3)
    p_agent.add_argument("--area-scale", type=float, default=0.5)
    p_agent.add_argument("--opt-level", type=int, default=2)
    p_agent.add_argument(
        "--cost-weight",
        type=float,
        default=0.0,
        help="Penalty λ on average two-qubit gate count in the reward.",
    )
    p_agent.add_argument(
        "--diary-shots",
        type=int,
        default=4096,
        help="Shots per diary teleport for the wormhole agent.",
    )
    p_agent.add_argument(
        "--log-csv",
        type=str,
        default="rl_agent_teleport.csv",
        help="Path to CSV log for the agent experiment.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "wire":
        print("[wire] Running three-qubit wormhole wire experiment...")
        normal_res, wormhole_res = run_wire_experiment(
            mode=args.mode,
            backend_name=args.backend,
            shots=args.shots,
            opt_level=args.opt_level,
        )
        print_wire_results(normal_res, wormhole_res)
        return

    if args.command == "diary":
        print("[diary] Running diary experiment (scramble + decode)...")
        diary_rows = run_diary_experiment(
            mode=args.mode,
            backend_name=args.backend,
            thetas=args.thetas,
            channels=["local", "wormhole"],
            shots=args.shots,
            opt_level=args.opt_level,
        )
        print_diary_summary(diary_rows)
        return

    if args.command == "rlqmf":
        model_labels = [m.strip() for m in args.models.split(",") if m.strip()]
        geoms = make_geometries(model_labels)
        channels = [c.strip().lower() for c in args.channels.split(",") if c.strip()]

        results, Q, actions = rlqmf_bandit(
            objective=args.objective,
            mode=args.mode,
            backend_name=args.backend,
            geoms=geoms,
            channels=channels,
            vec_indices=args.vec_indices,
            L_space=args.L_space,
            theta_space=args.theta_space,
            shot_budget=args.shot_budget,
            epsilon=args.epsilon,
            delta_r=args.delta_r,
            delta_theta=args.delta_theta,
            opt_level=args.opt_level,
            scale=args.area_scale,
            cost_weight=args.cost_weight,
        )
        print_rlqmf_summary(actions, Q)
        return

    if args.command == "agent":
        model_labels = [m.strip() for m in args.models.split(",") if m.strip()]
        geoms = make_geometries(model_labels)
        channels = [c.strip().lower() for c in args.channels.split(",") if c.strip()]

        run_teleported_agent_experiment(
            objective=args.objective,
            mode=args.mode,
            backend_name=args.backend,
            geoms=geoms,
            channels=channels,
            vec_indices=args.vec_indices,
            L_space=args.L_space,
            theta_space=args.theta_space,
            shot_budget=args.shot_budget,
            epsilon0=args.epsilon,
            delta_r=args.delta_r,
            delta_theta=args.delta_theta,
            opt_level=args.opt_level,
            scale=args.area_scale,
            cost_weight=args.cost_weight,
            diary_shots=args.diary_shots,
            log_csv=args.log_csv,
        )
        return

    model_labels = [m.strip() for m in getattr(args, "models", "dual2").split(",") if m.strip()]
    channels = [c.strip().lower() for c in getattr(args, "channels", "wormhole,local").split(",") if c.strip()]
    geoms = make_geometries(model_labels)

    if args.command == "transport":
        print("[transport] Running L–R chain transport experiment...")
        rows = run_transport_experiment(
            mode=args.mode,
            backend_name=args.backend,
            geoms=geoms,
            channels=channels,
            vec_indices=args.vec_indices,
            loops=args.loops,
            scales=args.area_scales,
            shots=args.shots,
            delta_r=args.delta_r,
            delta_theta=args.delta_theta,
            opt_level=args.opt_level,
        )
        print_transport_summary(rows)
        return

    if args.command == "backreaction":
        print("[backreaction] Running valve-signal experiment...")
        signal_modes = [s.strip() for s in args.signals.split(",") if s.strip()]
        rows = run_backreaction_experiment(
            mode=args.mode,
            backend_name=args.backend,
            geoms=geoms,
            channels=channels,
            vec_indices=args.vec_indices,
            loops=args.loops,
            scales=args.area_scales,
            shots=args.shots,
            delta_r=args.delta_r,
            delta_theta=args.delta_theta,
            opt_level=args.opt_level,
            signal_modes=signal_modes,
            signal_valve=args.signal_valve,
        )
        print_backreaction_summary(rows)
        return

    if args.command == "analyze":
        print("[analyze] Running wire + transport + backreaction + diary with analytic headlines...")
        normal_res, wormhole_res = run_wire_experiment(
            mode=args.mode,
            backend_name=args.backend,
            shots=args.shots,
            opt_level=args.opt_level,
        )
        rows_trans = run_transport_experiment(
            mode=args.mode,
            backend_name=args.backend,
            geoms=geoms,
            channels=channels,
            vec_indices=args.vec_indices,
            loops=args.loops,
            scales=args.area_scales,
            shots=args.shots,
            delta_r=args.delta_r,
            delta_theta=args.delta_theta,
            opt_level=args.opt_level,
        )
        signal_modes = [s.strip() for s in args.signals.split(",") if s.strip()]
        rows_back = run_backreaction_experiment(
            mode=args.mode,
            backend_name=args.backend,
            geoms=geoms,
            channels=channels,
            vec_indices=args.vec_indices,
            loops=args.loops,
            scales=args.area_scales,
            shots=args.shots,
            delta_r=args.delta_r,
            delta_theta=args.delta_theta,
            opt_level=args.opt_level,
            signal_modes=signal_modes,
            signal_valve=args.signal_valve,
        )
        diary_rows = run_diary_experiment(
            mode=args.mode,
            backend_name=args.backend,
            thetas=args.thetas,
            channels=["local", "wormhole"],
            shots=args.shots,
            opt_level=args.opt_level,
        )

        print_wire_results(normal_res, wormhole_res)
        print_transport_summary(rows_trans)
        print_backreaction_summary(rows_back)
        print_diary_summary(diary_rows)
        print_analytic_headlines(normal_res, wormhole_res, rows_trans, rows_back, diary_rows)
        return

    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
