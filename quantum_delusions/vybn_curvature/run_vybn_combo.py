"""
run_vybn_combo.py

Builds a compact batch:
- one-qubit commutator pairs (curvature control)
- a tiny 1D QCA step (static links + optional link-register)

Outputs:
  *.plan.jsonl        (PairMeta plan rows)
  *.manifest.json     (names + per-pair mapping and planned shots)
  *.qpy               (optional; if Qiskit present)

No submission here. Your runner handles backend + counts.
"""
import json, math, argparse, sys
from pathlib import Path
from dataclasses import asdict

# Qiskit optional
QISKIT = False
try:
    from qiskit import QuantumCircuit, transpile
    try:
        from qiskit import qpy
    except Exception:
        # Qiskit 1.x: qpy under qiskit.qpy
        from qiskit.qpy import dump as _qpy_dump, load as _qpy_load
        class _QPY:
            dump = staticmethod(_qpy_dump)
            load = staticmethod(_qpy_load)
        qpy = _QPY()
    QISKIT = True
except Exception:
    QISKIT = False

# Load scaffold sitting next to this file
import importlib.util, sys as _sys
spec = importlib.util.spec_from_file_location("vybn_combo_batch", "vybn_combo_batch.py")
if spec is None or spec.loader is None:
    print("Cannot find vybn_combo_batch.py next to this script.", file=sys.stderr)
    sys.exit(2)
mod = importlib.util.module_from_spec(spec)
_sys.modules["vybn_combo_batch"] = mod
spec.loader.exec_module(mod)  # type: ignore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=3)
    ap.add_argument("--plane", type=str, default="xz")
    ap.add_argument("--max-angle", type=float, default=0.18)
    ap.add_argument("--points", type=int, default=6)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--out", type=str, default="vybn_combo")
    args = ap.parse_args()

    # Plan commutator pairs
    comm_metas = mod.plan_commutator_pairs(
        plane=args.plane,
        micro_shapes=["square","long_theta"],
        n_points=args.points,
        max_angle=args.max_angle,
        m_loops=args.m,
        aspect=2.0,
        base_shots=384, min_shots=192, max_shots=4096
    )

    # Plan QCA pairs
    qca_metas = mod.plan_qca_pairs(
        n_qubits=args.nq,
        micro_shapes=["square"],
        n_points=max(3, args.points//2),
        max_angle=0.12,
        m_loops=args.m,
        step_count=2
    )

    circuits = []
    names = []
    manifest = []

    if QISKIT:
        cw, ccw, THETA, PHI = mod.build_commutator_templates(plane=args.plane, m_loops=args.m)
        for meta in comm_metas:
            b1 = cw.assign_parameters({THETA: meta.theta, PHI: meta.phi})
            b2 = ccw.assign_parameters({THETA: meta.theta, PHI: meta.phi})
            tag1 = f"{meta.tag}_cw"; tag2 = f"{meta.tag}_ccw"
            b1.name = tag1; b2.name = tag2
            circuits.extend([b1, b2])
            names.extend([tag1, tag2])
            manifest.append({"tag": meta.tag, "pair": [tag1, tag2], "shots": meta.shots, "plane": meta.plane})

        # QCA static links
        links = [1]*(args.nq-1)
        if len(links) >= 2: links[1] = 0
        for meta in qca_metas:
            step = mod.build_qca_step_static_links(args.nq, meta.theta, meta.phi, links=links)
            cw_pair, ccw_pair = mod.make_pair_from_step(step, m_loops=args.m)
            tag1 = f"{meta.tag}_staticlinks_cw"; tag2 = f"{meta.tag}_staticlinks_ccw"
            cw_pair.name = tag1; ccw_pair.name = tag2
            circuits.extend([cw_pair, ccw_pair])
            names.extend([tag1, tag2])
            manifest.append({"tag": meta.tag, "pair": [tag1, tag2], "shots": meta.shots, "plane": meta.plane, "links": links})

        # Optional link-register (small nq only)
        if args.nq <= 4:
            for meta in qca_metas[:2]:
                step = mod.build_qca_step_with_link_register(args.nq, meta.theta, meta.phi)
                # Here we just duplicate the step for cw/ccw tags for the same loop angles
                cw_step = step.copy(); ccw_step = step.copy()
                cw_step.name = f"{meta.tag}_linkreg_cw"
                ccw_step.name = f"{meta.tag}_linkreg_ccw"
                circuits.extend([cw_step, ccw_step])
                names.extend([cw_step.name, ccw_step.name])
                manifest.append({"tag": meta.tag, "pair": [cw_step.name, ccw_step.name], "shots": meta.shots, "plane": "qca_linkreg"})

        try:
            circuits = transpile(circuits, optimization_level=1)
        except Exception:
            pass

        with open(f"{args.out}.qpy", "wb") as f:
            qpy.dump(circuits, f)
        print(f"Wrote {args.out}.qpy with {len(circuits)} circuits")

    # Always write plan + manifest
    with open(f"{args.out}.plan.jsonl", "w", encoding="utf-8") as f:
        for m in (comm_metas + qca_metas):
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
    with open(f"{args.out}.manifest.json", "w", encoding="utf-8") as f:
        json.dump({"pairs": manifest, "names": names}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out}.plan.jsonl and {args.out}.manifest.json")
    if not QISKIT:
        print("Qiskit not found; skipped building QPY. That's fine—your runner can compile from plan+manifest.")

if __name__ == "__main__":
    main()
