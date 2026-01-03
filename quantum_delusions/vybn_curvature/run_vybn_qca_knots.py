#!/usr/bin/env python3
# submit_vybn_combo.py
# IBM Quantum Platform (free plan) friendly: V2 Sampler in job mode, no sessions.
# - Loads QPY circuits + manifest pairs + plan (free-form schema; extra keys are fine)
# - Optional tag filters (include/exclude prefixes, regex)
# - Per-pair parameter bind (theta/phi) if circuit is parameterized
# - Per-pair transpile to backend target (kills legacy U/U3 etc.)
# - Submits pairs [cw, ccw], writes/updates a counts JSON after each pair (resume-safe)
# - Skips pairs already present in the counts file

import argparse, json, re, sys, time
from pathlib import Path
from types import SimpleNamespace

# QPY + transpile for Qiskit 2.x (with fallback for older import path shims)
try:
    from qiskit import qpy, transpile
except Exception:
    from qiskit import transpile
    from qiskit.qpy import load as _load, dump as _dump
    class _QPY:  # minimal shim
        load = staticmethod(_load)
        dump = staticmethod(_dump)
    qpy = _QPY()

def load_plan(path: str):
    """Return dict[tag] -> SimpleNamespace(meta). Accept any fields (mask, step_count, …)."""
    metas = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tag = d.get("tag")
            if not tag:
                continue
            metas[tag] = SimpleNamespace(**d)
    return metas

def load_manifest(path: str):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    names = obj.get("names", [])
    pairs = obj.get("pairs", [])
    return names, pairs

def load_qpy(path: str):
    with open(path, "rb") as f:
        circs = list(qpy.load(f))
    return {c.name: c for c in circs}

def bind_theta_phi_if_needed(circ, theta, phi):
    """Assign parameters named like 'theta'/'phi' (case-insensitive). No-op if none present."""
    try:
        params = getattr(circ, "parameters", None)
        if not params:
            return circ
        mapping = {}
        for p in params:
            name = getattr(p, "name", "")
            lname = name.lower()
            if lname == "theta":
                mapping[p] = float(theta)
            elif lname == "phi":
                mapping[p] = float(phi)
        if mapping:
            return circ.assign_parameters(mapping)
        return circ
    except Exception:
        # If anything odd happens, just pass the original circuit through.
        return circ

def select_pairs(pairs, include_prefixes, exclude_prefixes, tag_regex, start_index, max_pairs):
    def ok(tag: str):
        if include_prefixes and not any(tag.startswith(p) for p in include_prefixes):
            return False
        if exclude_prefixes and any(tag.startswith(p) for p in exclude_prefixes):
            return False
        if tag_regex and not re.search(tag_regex, tag):
            return False
        return True
    filtered = [p for p in pairs if ok(p.get("tag",""))]
    if start_index and start_index > 0:
        filtered = filtered[start_index:]
    if max_pairs and max_pairs > 0:
        filtered = filtered[:max_pairs]
    return filtered

def bitpad(key: str, n: int) -> str:
    """Left-pad bitstring key to width n if it looks like a plain binary string; else return as-is."""
    if isinstance(key, str) and all(ch in "01" for ch in key):
        return key.zfill(n)
    return str(key)

def extract_counts_from_pub(pub, nbits: int):
    """
    Try a few result layouts to get integer counts.
    Works with qiskit-ibm-runtime 0.43.x SamplerV2 results.
    """
    # 1) The convenient aggregator (present in recent Qiskit)
    try:
        jd = pub.join_data()  # may expose get_counts()
        if hasattr(jd, "get_counts"):
            cnts = jd.get_counts()
            # keys may be ints or strings; normalize + pad
            return {bitpad(k, nbits): int(v) for k, v in dict(cnts).items()}
    except Exception:
        pass

    # 2) Reach into pub.data.meas if it provides get_counts()
    try:
        meas = getattr(pub.data, "meas", None)
        if meas and hasattr(meas, "get_counts"):
            cnts = meas.get_counts()
            return {bitpad(k, nbits): int(v) for k, v in dict(cnts).items()}
    except Exception:
        pass

    # 3) Quasi-dist fallback → approximate counts using shots metadata if available
    try:
        shots = getattr(pub, "shots", None)
        qd = getattr(pub.data, "quasi_dist", None)
        if qd and shots:
            # quasi keys may be ints (bit integers)
            out = {}
            for k, p in dict(qd).items():
                # convert bit-int to str if needed
                if isinstance(k, int):
                    key = format(k, f"0{nbits}b")
                else:
                    key = bitpad(k, nbits)
                out[key] = int(round(float(p) * int(shots)))
            if out:
                return out
    except Exception:
        pass

    return None

def main():
    ap = argparse.ArgumentParser(
        description="Submit cw/ccw pairs (job-mode V2) with per-pair bind+transpile; resume-safe counts writer."
    )
    ap.add_argument("--qpy", required=True, help="Input circuits (.qpy)")
    ap.add_argument("--manifest", required=True, help="Manifest .json listing names/pairs")
    ap.add_argument("--plan", required=True, help="Plan .jsonl mapping tag → metadata (theta, phi, shots, etc.)")
    ap.add_argument("--backend", default="ibm_fez", help="Backend name (e.g., ibm_fez)")
    ap.add_argument("--channel", default="ibm_quantum_platform", help="Account channel")
    ap.add_argument("--instance", default=None, help="Instance (e.g., open-instance or crn string)")
    ap.add_argument("--token", default=None, help="Optional API token (usually not needed if already saved)")
    ap.add_argument("--out", default="vybn_combo.counts.json", help="Counts output path (resume-safe)")
    ap.add_argument("--include-prefix", default=None, help="Comma-separated tag prefixes to include (filter)")
    ap.add_argument("--exclude-prefix", default=None, help="Comma-separated tag prefixes to exclude")
    ap.add_argument("--tag-regex", default=None, help="Only include tags that match this regex")
    ap.add_argument("--start-index", type=int, default=0, help="Skip the first N selected pairs")
    ap.add_argument("--max-pairs", type=int, default=None, help="Cap the number of pairs submitted")
    ap.add_argument("--opt-level", type=int, default=1, choices=[0,1,2,3], help="Transpile optimization level")
    ap.add_argument("--seed-transpile", type=int, default=None, help="Set a transpiler seed for reproducibility")
    ap.add_argument("--flush-every", type=int, default=1, help="Write the counts file after this many pairs")
    ap.add_argument("--dry-run", action="store_true", help="Plan & transpile only; do not submit jobs")
    args = ap.parse_args()

    # ---- Load inputs
    by_name = load_qpy(args.qpy)
    _, all_pairs = load_manifest(args.manifest)
    meta_by_tag = load_plan(args.plan)

    # ---- Select which pairs to run
    inc = [s.strip() for s in args.include_prefix.split(",")] if args.include_prefix else None
    exc = [s.strip() for s in args.exclude_prefix.split(",")] if args.exclude_prefix else None
    selected_pairs = select_pairs(all_pairs, inc, exc, args.tag_regex, args.start_index, args.max_pairs)
    total = len(selected_pairs)
    if total == 0:
        print("Nothing to submit after filtering. Check your include/exclude/regex options.")
        return

    # ---- Connect to IBM Quantum Platform (V2 primitives)
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    svc_kwargs = {"channel": args.channel}
    if args.instance: svc_kwargs["instance"] = args.instance
    if args.token:    svc_kwargs["token"] = args.token
    service = QiskitRuntimeService(**svc_kwargs)
    backend = service.backend(args.backend)

    # ---- Resume-safe: load or initialize counts
    out_path = Path(args.out)
    if out_path.exists():
        try:
            out_counts = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            out_counts = {}
    else:
        out_counts = {}

    # ---- V2 sampler in job mode (no Session)
    sampler = Sampler(backend)

    # ---- Process pairs
    done_since_flush = 0
    for i, pair in enumerate(selected_pairs, 1):
        tag = pair.get("tag", "")
        cw_name, ccw_name = pair["pair"]
        shots = int(pair.get("shots", 1024))

        # Skip already present
        if cw_name in out_counts and ccw_name in out_counts:
            print(f"[{i}/{total}] {tag}  (already collected)")
            continue

        # Source circuits
        cw_orig = by_name.get(cw_name)
        ccw_orig = by_name.get(ccw_name)
        if cw_orig is None or ccw_orig is None:
            print(f"[{i}/{total}] {tag}  [warn] missing circuits; skipping", file=sys.stderr)
            continue

        # Bind θ, φ if present
        meta = meta_by_tag.get(tag)
        theta = getattr(meta, "theta", None)
        phi   = getattr(meta, "phi",   None)

        cw_bound  = bind_theta_phi_if_needed(cw_orig,  theta, phi)
        ccw_bound = bind_theta_phi_if_needed(ccw_orig, theta, phi)

        # Transpile per pair to backend target (kills U/U3 etc., maps to device)
        try:
            tpair = transpile([cw_bound, ccw_bound],
                              backend=backend,
                              optimization_level=args.opt_level,
                              seed_transpiler=args.seed_transpile)
            t_cw, t_ccw = tpair[0], tpair[1]
        except Exception as e:
            print(f"[{i}/{total}] {tag}  [error] transpile failed: {e}", file=sys.stderr)
            continue

        nbits_cw  = getattr(t_cw,  "num_clbits", 0) or 1
        nbits_ccw = getattr(t_ccw, "num_clbits", 0) or 1

        if args.dry_run:
            print(f"[{i}/{total}] {tag}  (dry-run) shots={shots}  cw={cw_name} ccw={ccw_name}")
            continue

        # Submit & extract counts
        try:
            job = sampler.run([t_cw, t_ccw], shots=shots)
            res = job.result()
            pubs = [res[0], res[1]]
        except Exception as e:
            print(f"[{i}/{total}] {tag}  [error] job failed: {e}", file=sys.stderr)
            continue

        try:
            c_cw  = extract_counts_from_pub(pubs[0], nbits_cw)
            c_ccw = extract_counts_from_pub(pubs[1], nbits_ccw)
            if not c_cw or not c_ccw:
                print(f"[{i}/{total}] {tag}  [warn] no counts extracted; skipping write", file=sys.stderr)
                continue
            out_counts[cw_name]  = c_cw
            out_counts[ccw_name] = c_ccw
            print(f"[{i}/{total}] {tag}  shots={shots}")
        except Exception as e:
            print(f"[{i}/{total}] {tag}  [error] postprocess failed: {e}", file=sys.stderr)
            continue

        # Incremental flush
        done_since_flush += 1
        if done_since_flush >= max(1, int(args.flush_every)):
            out_path.write_text(json.dumps(out_counts, ensure_ascii=False, indent=2), encoding="utf-8")
            done_since_flush = 0

    # Final write
    out_path.write_text(json.dumps(out_counts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} with {len(out_counts)} histograms")

if __name__ == "__main__":
    main()
