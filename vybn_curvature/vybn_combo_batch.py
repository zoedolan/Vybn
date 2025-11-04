"""
vybn_combo_batch.py

A small scaffold to multiplex multiple curvature-related experiments into one batch
so a single quantum job yields several datasets. It is intentionally minimal.
Wire this into your existing 'nailbiter.py' flow or import pieces from here.

Principle: reuse one loop kernel (cw vs ccw over a small enclosed area) and
vary only (i) what "a step" means and (ii) what we read out from the same bitstrings.
This keeps the job compact while touching curvature, locality, reversibility,
and FR-link comparisons.

Nothing runs on import. All heavy dependencies live behind try/except so you
can vendor this without breaking non-QPU environments.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional, Any
import math, json, random

# Optional Qiskit imports guarded for offline environments.
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.circuit import Parameter
except Exception:  # pragma: no cover
    QuantumCircuit = object
    QuantumRegister = object
    ClassicalRegister = object
    Parameter = object
    def transpile(*a, **k):  # type: ignore
        raise RuntimeError("Qiskit not available in this environment")

# Try to reuse Zoe's local helpers when present.
try:
    import nailbiter  # Zoe's uploaded script (single-qubit commutator engine)
    NB_AVAILABLE = True
except Exception:
    NB_AVAILABLE = False

# -------------------- Data schema --------------------

@dataclass
class PairMeta:
    tag: str                     # logical name for this pair (e.g., 'comm_xy_long_r_a0.08')
    plane: str                   # 'xz','yz','xy' or 'qca'
    micro_shape: str             # 'square','long_theta','long_r', etc.
    area: float                  # signed loop area in the control plane
    theta: float                 # resolved parameter 1
    phi: float                   # resolved parameter 2
    shots: int                   # shot budget for this pair
    tau_loop_s: float            # scheduled loop duration (seconds) if known, else NaN
    step_count: int = 1          # for automaton/QCA experiments
    n_qubits: int = 1            # width when applicable

@dataclass
class ResultRow:
    tag: str
    plane: str
    micro_shape: str
    area: float
    theta: float
    phi: float
    shots_cw: int
    shots_ccw: int
    p1_cw: float                 # Pr(measure 1 on reference qubit) or any chosen observable
    p1_ccw: float
    delta: float                 # orientation-odd component = p1_cw - p1_ccw
    slope_per_area: float        # delta / area for small loops
    tau_loop_s: float            # scheduled duration if available
    kappa_eff_per_area_Hz: float # slope_per_area / tau_loop_s if tau is known
    extra: Dict[str, Any]        # place for MI, correlations, entanglement proxies, etc.

# -------------------- Helpers --------------------

def binom_stderr(p: float, n: int) -> float:
    if n <= 0:
        return float('nan')
    p = max(0.0, min(1.0, p))
    return math.sqrt(max(0.0, p*(1.0-p)) / float(n))

def plateau_score(slopes: List[float], k_small: int = 6) -> Tuple[float, bool]:
    """
    Coefficient of variation and same-sign check over the first k_small slopes.
    Returns (cv, same_sign_boolean).
    """
    if not slopes:
        return float('nan'), False
    xs = slopes[:max(1, min(k_small, len(slopes)))]
    m = sum(xs)/len(xs)
    if len(xs) < 2 or abs(m) < 1e-12:
        return float('inf'), all(x >= 0 for x in xs) or all(x <= 0 for x in xs)
    var = sum((x-m)*(x-m) for x in xs) / len(xs)
    cv = abs(math.sqrt(max(0.0, var)) / m)
    return cv, (all(x >= 0 for x in xs) or all(x <= 0 for x in xs))

def shape_params(area: float, shape: str, aspect: float) -> Tuple[float, float]:
    """
    Map a signed loop area to concrete angles (theta, phi) given a micro-shape choice.
    Mirrors the logic in nailbiter.py so datasets line up.
    """
    s = (shape or 'square').lower()
    a = float(abs(area))
    if s == 'square':
        t = math.sqrt(a); return (math.copysign(t, area), t)
    if s == 'long_theta':
        t = max(1e-9, aspect)*math.sqrt(a); return (math.copysign(t, area), a/max(t,1e-18))
    if s == 'long_r':
        p = max(1e-9, aspect)*math.sqrt(a); return (a/max(p,1e-18), math.copysign(p, area))
    raise ValueError(f'unknown micro-shape: {shape}')

def adaptive_shots(area: float, amin: float, m: int, base: int, mn: int, mx: int) -> int:
    """
    Concentrate budget near the origin, gently boost with sqrt(m).
    """
    if area <= 0 or amin <= 0:
        return int(max(mn, min(mx, base)))
    exponent = 1.5
    scale = (amin / area) ** exponent
    scale *= max(1.0, math.sqrt(max(1, m)))
    shots = int(round(base * scale))
    return int(max(mn, min(mx, shots)))

# -------------------- Loop builders --------------------

def build_commutator_templates(plane: str = 'xz', m_loops: int = 1):
    """
    Prefer Zoe's implementation when available for perfect alignment with prior data.
    Fallback to a tiny in-file variant if nailbiter isn't importable.
    Returns (cw_tpl, ccw_tpl, THETA, PHI).
    """
    if NB_AVAILABLE and hasattr(nailbiter, 'build_commutator_templates'):
        return nailbiter.build_commutator_templates(plane=plane, m_loops=m_loops)

    # Fallback: simple one-qubit version using Qiskit.
    if QuantumCircuit is object:
        raise RuntimeError('Qiskit not available and nailbiter not importable')
    g1, g2 = None, None
    pl = (plane or 'xz').lower()
    if pl == 'xz': g1, g2 = 'rz', 'rx'
    elif pl == 'yz': g1, g2 = 'rz', 'ry'
    elif pl == 'xy': g1, g2 = 'rx', 'ry'
    else: raise ValueError('plane must be xz, yz, or xy')
    THETA = Parameter('theta')
    PHI   = Parameter('phi')

    def apply(qc, name, angle, q=0):
        if name == 'rx': qc.rx(angle, q)
        elif name == 'ry': qc.ry(angle, q)
        elif name == 'rz': qc.rz(angle, q)
        else: raise ValueError(name)

    def comm_layer(qc, forward=True):
        a, b = (g1, g2) if forward else (g2, g1)
        apply(qc, a, THETA); apply(qc, b, PHI); apply(qc, a, -THETA); apply(qc, b, -PHI)

    cw  = QuantumCircuit(1, 1)
    ccw = QuantumCircuit(1, 1)
    for _ in range(max(1, int(m_loops))):
        comm_layer(cw, forward=True)
        comm_layer(ccw, forward=False)
    cw.measure(0, 0); ccw.measure(0, 0)
    return cw, ccw, THETA, PHI

def build_qca_step(n_qubits: int, theta: float, phi: float) -> 'QuantumCircuit':
    """
    One step of a tiny 1D quantum cellular automaton:
    - Odd bonds: RZ(theta/2) on all, then CX(i->i+1) for i even
    - Even bonds: RX(phi/2) on all, then CX(i->i+1) for i odd
    The goal is locality + brickwork, not a particular model.
    """
    if QuantumCircuit is object:
        raise RuntimeError('Qiskit is required to build QCA steps')
    n = max(2, int(n_qubits))
    qc = QuantumCircuit(n, n)
    for q in range(n):
        qc.rz(theta/2.0, q)
    for i in range(0, n-1, 2):
        qc.cx(i, i+1)
    for q in range(n):
        qc.rx(phi/2.0, q)
    for i in range(1, n-1, 2):
        qc.cx(i, i+1)
    return qc

def make_pair_from_step(step: 'QuantumCircuit', m_loops: int = 1) -> Tuple['QuantumCircuit','QuantumCircuit']:
    """
    Wrap a QCA 'step' in a minimal group-commutator around its two control angles.
    We reuse the one-qubit loop idea: cw applies (theta,phi, -theta, -phi), ccw swaps the order.
    """
    if QuantumCircuit is object:
        raise RuntimeError('Qiskit is required to wrap QCA steps')
    n = step.num_qubits
    THETA = Parameter('theta'); PHI = Parameter('phi')
    def inst(theta, phi):
        # Re-parametrize the step by scaling its RX/RZ angles.
        qc = QuantumCircuit(n, n)
        for q in range(n): qc.rz(theta/2.0, q)
        for i in range(0, n-1, 2): qc.cx(i, i+1)
        for q in range(n): qc.rx(phi/2.0, q)
        for i in range(1, n-1, 2): qc.cx(i, i+1)
        qc.barrier()
        return qc
    def commute_block(qc, theta, phi, forward=True):
        if forward:
            qc.compose(inst(THETA, PHI), inplace=True)
            qc.compose(inst(-THETA, -PHI), inplace=True)
        else:
            qc.compose(inst(PHI, THETA), inplace=True)   # swap, then inverse order
            qc.compose(inst(-PHI, -THETA), inplace=True)
    cw  = QuantumCircuit(n, n)
    ccw = QuantumCircuit(n, n)
    for _ in range(max(1, int(m_loops))):
        commute_block(cw, THETA, PHI, forward=True)
        commute_block(ccw, THETA, PHI, forward=False)
    cw.measure(range(n), range(n))
    ccw.measure(range(n), range(n))
    return cw, ccw

# -------------------- Reduction --------------------

def reduce_counts_to_p1(counts: Dict[str,int], ref_bit: int = 0) -> float:
    """
    Compute Pr(bit=1) on a chosen reference bit from a full bitstring histogram.
    """
    if not counts:
        return float('nan')
    total = sum(counts.values())
    if total <= 0:
        return float('nan')
    p1 = 0
    for bitstring, c in counts.items():
        # qiskit returns little-endian bitstrings; safest is to index from 0.
        b = bitstring[ref_bit] if ref_bit < len(bitstring) else bitstring[-1]
        if b == '1': p1 += c
    return p1 / float(total)

def pair_row_from_counts(meta: PairMeta,
                         cw_counts: Dict[str,int],
                         ccw_counts: Dict[str,int],
                         ref_bit: int = 0) -> ResultRow:
    p1_cw = reduce_counts_to_p1(cw_counts, ref_bit=ref_bit)
    p1_ccw = reduce_counts_to_p1(ccw_counts, ref_bit=ref_bit)
    delta = (p1_cw - p1_ccw) if (math.isfinite(p1_cw) and math.isfinite(p1_ccw)) else float('nan')
    slope = (delta / meta.area) if (meta.area != 0) else float('nan')
    kappa = (slope / meta.tau_loop_s) if (meta.tau_loop_s and math.isfinite(meta.tau_loop_s) and meta.tau_loop_s > 0) else float('nan')
    return ResultRow(
        tag=meta.tag, plane=meta.plane, micro_shape=meta.micro_shape, area=meta.area,
        theta=meta.theta, phi=meta.phi, shots_cw=sum(cw_counts.values()), shots_ccw=sum(ccw_counts.values()),
        p1_cw=p1_cw, p1_ccw=p1_ccw, delta=delta, slope_per_area=slope,
        tau_loop_s=meta.tau_loop_s, kappa_eff_per_area_Hz=kappa, extra={},
    )

# -------------------- Planning --------------------

def plan_commutator_pairs(plane: str,
                          micro_shapes: List[str],
                          n_points: int,
                          max_angle: float,
                          m_loops: int,
                          aspect: float = 2.0,
                          base_shots: int = 400,
                          min_shots: int = 200,
                          max_shots: int = 2000) -> List[PairMeta]:
    """
    Build a small, origin-focused schedule of areas and turn them into metas.
    """
    areas = [ (max_angle**2) * (i+1)/float(n_points) for i in range(n_points) ]
    amin = areas[0] if areas else 0.0
    metas : List[PairMeta] = []
    for shape in micro_shapes:
        for a in areas:
            t, p = shape_params(a, shape, aspect)
            shots = adaptive_shots(a, amin, m_loops, base_shots, min_shots, max_shots)
            metas.append(PairMeta(
                tag=f'comm_{plane}_{shape}_a{a:.4f}',
                plane=plane, micro_shape=shape, area=a, theta=t, phi=p,
                shots=shots, tau_loop_s=float('nan'), step_count=1, n_qubits=1
            ))
    return metas

def plan_qca_pairs(n_qubits: int,
                   micro_shapes: List[str],
                   n_points: int,
                   max_angle: float,
                   m_loops: int,
                   aspect: float = 2.0,
                   base_shots: int = 400,
                   min_shots: int = 200,
                   max_shots: int = 2000,
                   step_count: int = 3) -> List[PairMeta]:
    areas = [ (max_angle**2) * (i+1)/float(n_points) for i in range(n_points) ]
    amin = areas[0] if areas else 0.0
    metas : List[PairMeta] = []
    for shape in micro_shapes:
        for a in areas:
            t, p = shape_params(a, shape, aspect)
            shots = adaptive_shots(a, amin, m_loops, base_shots, min_shots, max_shots)
            metas.append(PairMeta(
                tag=f'qca_{n_qubits}q_{shape}_a{a:.4f}',
                plane='qca', micro_shape=shape, area=a, theta=t, phi=p,
                shots=shots, tau_loop_s=float('nan'), step_count=step_count, n_qubits=n_qubits
            ))
    return metas

# -------------------- Serialization --------------------

def rows_to_jsonl(rows: List[ResultRow]) -> str:
    return "\n".join(json.dumps(asdict(r), ensure_ascii=False) for r in rows)

def save_rows(path: str, rows: List[ResultRow]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(rows_to_jsonl(rows))

# -------------------- Correlation utilities --------------------

def _bitprob(counts: Dict[str,int], idx: int, bitval: str) -> float:
    if not counts: return float('nan')
    tot = sum(counts.values())
    if tot <= 0: return float('nan')
    s = 0
    for bstr, c in counts.items():
        if idx < len(bstr):
            if bstr[idx] == bitval: s += c
        else:
            if bstr[-1] == bitval: s += c
    return s/float(tot)

def mutual_information_2bit(counts: Dict[str,int], i: int, j: int) -> float:
    """
    Empirical MI for two measured bits from a bitstring histogram.
    """
    if not counts: return float('nan')
    tot = sum(counts.values())
    if tot <= 0: return float('nan')
    # Joint
    joint = {('0','0'):0,('0','1'):0,('1','0'):0,('1','1'):0}
    for bstr, c in counts.items():
        bi = bstr[i] if i < len(bstr) else bstr[-1]
        bj = bstr[j] if j < len(bstr) else bstr[-1]
        joint[(bi,bj)] += c
    import math as _m
    ps = {k: v/float(tot) for k, v in joint.items()}
    pi = {'0': ps[('0','0')] + ps[('0','1')],
          '1': ps[('1','0')] + ps[('1','1')]}
    pj = {'0': ps[('0','0')] + ps[('1','0')],
          '1': ps[('0','1')] + ps[('1','1')]}
    mi = 0.0
    for a in ('0','1'):
        for b in ('0','1'):
            p_ab = ps[(a,b)]
            if p_ab > 0 and pi[a] > 0 and pj[b] > 0:
                mi += p_ab * _m.log(p_ab/(pi[a]*pj[b]), 2)
    return mi

def zz_correlation(counts: Dict[str,int], i: int, j: int) -> float:
    """
    E[Z_i Z_j] from bitstrings coded as '0'->+1, '1'->-1.
    """
    if not counts: return float('nan')
    tot = sum(counts.values())
    if tot <= 0: return float('nan')
    val = 0.0
    for bstr, c in counts.items():
        bi = bstr[i] if i < len(bstr) else bstr[-1]
        bj = bstr[j] if j < len(bstr) else bstr[-1]
        zi = 1.0 if bi == '0' else -1.0
        zj = 1.0 if bj == '0' else -1.0
        val += zi*zj * c
    return val/float(tot)

# -------------------- Link-enabled QCA steps --------------------

def build_qca_step_static_links(n_qubits: int, theta: float, phi: float, links: Optional[List[int]] = None) -> 'QuantumCircuit':
    """
    Static "cut-and-glue" via a link mask. If links[k]==1, bond (k,k+1) is active; else it's cut.
    This is compile-time gating (no link register). It's shallow and hardware-friendly.
    """
    if QuantumCircuit is object:
        raise RuntimeError('Qiskit is required to build QCA steps')
    n = max(2, int(n_qubits))
    qc = QuantumCircuit(n, n)
    # Single-qubit "on-site" pieces
    for q in range(n): qc.rz(theta/2.0, q)
    # Odd bonds
    L = links if links is not None else [1]*(n-1)
    for i in range(0, n-1, 2):
        if L[i] == 1:
            qc.cx(i, i+1)
    for q in range(n): qc.rx(phi/2.0, q)
    # Even bonds
    for i in range(1, n-1, 2):
        if L[i] == 1:
            qc.cx(i, i+1)
    return qc

def build_qca_step_with_link_register(n_qubits: int, theta: float, phi: float) -> 'QuantumCircuit':
    """
    Unitary, translation-invariant "cut-and-glue" using a link register:
    for each bond (i,i+1) we add a link qubit L_i and apply CSWAP(L_i; i, i+1).
    This models glue when L_i=1 and cut when L_i=0, without mid-circuit measurement.
    Depth grows, so keep n small (<=5) for NISQ sanity.
    """
    if QuantumCircuit is object:
        raise RuntimeError('Qiskit is required to build QCA steps')
    n = max(2, int(n_qubits))
    l = n-1  # one link per bond
    total = n + l
    qc = QuantumCircuit(total, total)
    # On-site pieces on matter
    for q in range(n): qc.rz(theta/2.0, q)
    # Odd bonds CSWAP controlled by link qubits L_i (indexed after matter)
    for i in range(0, n-1, 2):
        link = n + i
        qc.cswap(link, i, i+1)
    for q in range(n): qc.rx(phi/2.0, q)
    # Even bonds
    for i in range(1, n-1, 2):
        link = n + i
        qc.cswap(link, i, i+1)
    # By default we measure only matter; callers can extend to link bits.
    qc.measure(range(n), range(n))
    return qc

# Improve repo-compat import behavior for Zoe's script
if not NB_AVAILABLE:
    try:
        import importlib
        nailbiter = importlib.import_module("vybn_curvature.nailbiter")  # repo path variant
        NB_AVAILABLE = True
    except Exception:
        NB_AVAILABLE = False
