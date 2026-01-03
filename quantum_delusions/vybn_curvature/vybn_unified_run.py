#!/usr/bin/env python3
"""
vybn_unified_run.py  —  minimal, no-drama master script

Goal: run ONE decisive pass end‑to‑end with the smallest surface area.
Assumes these files live in the SAME directory as this script (renamed without (1)s):
  - run_vybn_combo.py
  - submit_vybn_curvature.py
  - reduce_vybn_combo.py
  - post_reducer_qca.py
  - vybn_combo_batch.py   (required by run_vybn_combo.py)

Pipeline:
  1) Plan    -> <out>.plan.jsonl, <out>.manifest.json, <out>.qpy
  2) Submit  -> <out>.counts.json (resume‑safe)
  3) Reduce  -> <out>.results.jsonl  +  qca_post_reduce.csv/json
  4) Report  -> print a single slope/area summary from small‑area points

Design choices to keep complexity down:
  - Very few flags. Backend/instance/out are the only required ones.
  - Sensible defaults (xz plane, 6 points, m=4, max‑angle=0.18, opt‑level=1).
  - Guarded fast‑fail if a required file is missing.
  - Resume‑safe submission (delegated to submit_vybn_curvature.py).
"""
import argparse, json, subprocess, sys, statistics, time
from pathlib import Path

HERE = Path(__file__).resolve().parent

def check_files():
    must = [
        "run_vybn_combo.py",
        "submit_vybn_curvature.py",
        "reduce_vybn_combo.py",
        "post_reducer_qca.py",
        "vybn_combo_batch.py",
    ]
    missing = [m for m in must if not (HERE / m).exists()]
    if missing:
        raise SystemExit(f"Missing required file(s) next to this script: {missing}")

def run(cmd):
    t0 = time.time()
    print("[run]", " ".join(str(x) for x in cmd), flush=True)
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    print(f"[done] ({rc}) in {dt:.1f}s", flush=True)
    if rc != 0:
        raise SystemExit(rc)

def small_area_slope(results_path: Path, top_frac: float = 0.33):
    """
    Read <out>.results.jsonl and estimate slope(Δ vs signed_area) for the smallest |area| points.
    This attempts to be robust to field naming: looks for keys containing 'area' and 'delta'.
    """
    rows = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # heuristic field extraction
            a = None; d = None
            for k, v in obj.items():
                kl = k.lower()
                if a is None and "area" in kl and isinstance(v, (int, float)):
                    a = float(v)
                if d is None and "delta" in kl and isinstance(v, (int, float)):
                    d = float(v)
            if a is not None and d is not None:
                rows.append((a, d))

    if not rows:
        return {"n": 0, "slope": None, "corr": None}

    # pick the smallest |area| fraction
    rows.sort(key=lambda t: abs(t[0]))
    k = max(2, int(len(rows) * top_frac))
    sub = rows[:k]
    xs = [x for x, _ in sub]
    ys = [y for _, y in sub]

    # simple least‑squares
    x_bar = statistics.fmean(xs)
    y_bar = statistics.fmean(ys)
    sxx = sum((x - x_bar) ** 2 for x in xs)
    sxy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx != 0 else 0.0
    # correlation (for rough health check)
    syy = sum((y - y_bar) ** 2 for y in ys)
    corr = sxy / (sxx ** 0.5 * syy ** 0.5) if sxx > 0 and syy > 0 else 0.0
    return {"n": len(sub), "slope": slope, "corr": corr}

def main():
    check_files()

    ap = argparse.ArgumentParser(description="Run a minimal curvature+QCA batch end‑to‑end.")
    ap.add_argument("--backend", required=True, help="e.g., ibm_fez")
    ap.add_argument("--instance", required=True, help="e.g., open-instance")
    ap.add_argument("--out-prefix", required=True, help="e.g., vybn_combo_unified")

    # optional knobs (kept tiny)
    ap.add_argument("--plane", default="xz")
    ap.add_argument("--points", type=int, default=6)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--max-angle", type=float, default=0.18)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--seed-transpile", type=int, default=None)
    ap.add_argument("--flush-every", type=int, default=1)
    args = ap.parse_args()

    out = args.out_prefix
    plan = HERE / f"{out}.plan.jsonl"
    manifest = HERE / f"{out}.manifest.json"
    qpy = HERE / f"{out}.qpy"
    counts = HERE / f"{out}.counts.json"
    results = HERE / f"{out}.results.jsonl"
    qca_csv = HERE / "qca_post_reduce.csv"
    qca_json = HERE / "qca_post_reduce.json"

    # 1) plan
    run([sys.executable, str(HERE/"run_vybn_combo.py"),
         "--nq", "3",
         "--plane", args.plane,
         "--max-angle", str(args.max_angle),
         "--points", str(args.points),
         "--m", str(args.m),
         "--out", out])

    if not qpy.exists():
        raise SystemExit(f"Planner did not write {qpy}. Ensure qiskit is installed and vybn_combo_batch.py is present.")

    # 2) submit (resume‑safe)
    run([sys.executable, str(HERE/"submit_vybn_curvature.py"),
         "--qpy", str(qpy),
         "--manifest", str(manifest),
         "--plan", str(plan),
         "--backend", args.backend,
         "--channel", "ibm_quantum_platform",
         "--instance", args.instance,
         "--out", str(counts),
         "--opt-level", str(args.opt_level),
         "--flush-every", str(args.flush_every)] + (
            ["--seed-transpile", str(args.seed_transpile)] if args.seed_transpile is not None else []
         ))

    # 3) reduce
    run([sys.executable, str(HERE/"reduce_vybn_combo.py"),
         "--plan", str(plan),
         "--manifest", str(manifest),
         "--counts", str(counts),
         "--out", str(results)])

    run([sys.executable, str(HERE/"post_reducer_qca.py"),
         "--counts", str(counts),
         "--out", str(qca_csv),
         "--json-out", str(qca_json)])

    # 4) quick small‑area report
    rep = small_area_slope(results)
    print("\n=== Small‑area summary ===")
    print(json.dumps(rep, indent=2))
    verdict = "GO" if rep["slope"] not in (None, 0.0) and abs(rep["corr"]) >= 0.2 else "RECHECK"
    print(f"Verdict: {verdict} (slope near zero or weak correlation → recheck geometry / shots)")

if __name__ == "__main__":
    main()
