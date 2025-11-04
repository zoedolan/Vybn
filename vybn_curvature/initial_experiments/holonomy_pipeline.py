#!/usr/bin/env python3
# holonomy_pipeline.py
# Runs holonomy_trans_fixedpoint.py across a grid and summarizes Δ/area plateau,
# micro-shape invariance, and commuting control.
#
# Usage examples at bottom.

import csv, json, math, os, subprocess, sys
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev

PY = sys.executable or "python"
HERE = Path(__file__).resolve().parent
EXE = HERE / "holonomy_trans_fixedpoint.py"  # assumes same folder

def run_one(out, model="vit", areas="1e-10,3e-10,1e-9,3e-9",
            micro_shapes="balanced",
            replicates=8, rng="numpy", seed=42,
            tape_scale=32, T=0.125, fixed_tape=True, fixed_batches=True,
            preburn=50, commute=False):
    cmd = [
        PY, str(EXE),
        "--model", model,
        "--areas", areas,
        "--micro-shapes", micro_shapes,
        "--replicates", str(replicates),
        "--rng", rng, "--seed", str(seed),
        "--tape-scale", str(tape_scale),
        "--T", str(T),
        "--preburn", str(preburn),
        "--out", out
    ]
    if fixed_tape: cmd.append("--fixed-tape")
    if fixed_batches: cmd.append("--fixed-batches")
    if commute: cmd.append("--commute")
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def _read_csv(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def _to_float(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except:
        return default

def load_results(folder):
    """Return dict keyed by (out_name) with rows from its CSV."""
    out = {}
    for csv_path in Path(folder).glob("*.csv"):
        if csv_path.name.endswith("_summary.csv"):  # per-run summaries
            continue
        try:
            rows = _read_csv(csv_path)
        except Exception:
            continue
        if rows:
            out[csv_path.stem] = rows
    return out

def summarize(rows, key_delta_over_area="A_slope_per_area_procrustes"):
    """
    Group by (micro_shape, T, substeps_A, area) and compute mean/sem of Δ/area.
    Returns nested dict and a flat list for CSV export.
    """
    # Try multiple keys in case you prefer projk
    candidates = [
        "A_slope_per_area_procrustes",
        "A_slope_per_area_projk"
    ]
    if key_delta_over_area not in candidates:
        candidates.insert(0, key_delta_over_area)

    # pick first available metric
    metric_key = None
    for k in candidates:
        if any(k in r for r in rows):
            metric_key = k
            break
    if metric_key is None:
        metric_key = "A_slope_per_area_procrustes"  # fallback

    groups = defaultdict(list)
    for r in rows:
        area = _to_float(r, "area")
        T = _to_float(r, "T")
        subA = _to_float(r, "substeps_A")
        shape = r.get("micro_shape", "balanced")
        val = _to_float(r, metric_key)
        rep = r.get("replicate")
        if not (math.isfinite(area) and math.isfinite(T) and math.isfinite(subA) and math.isfinite(val)):
            continue
        groups[(shape, T, subA, area)].append(val)

    flat = []
    by_shape_T_sub = defaultdict(list)
    for (shape, T, subA, area), vals in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        m = mean(vals)
        sd = pstdev(vals) if len(vals) > 1 else 0.0
        sem = sd / math.sqrt(max(1, len(vals)))
        flat.append({
            "micro_shape": shape,
            "T": T,
            "substeps_A": subA,
            "area": area,
            "n": len(vals),
            "delta_over_area_mean": m,
            "delta_over_area_sem": sem
        })
        by_shape_T_sub[(shape, T, subA)].append((area, m, sem))

    return flat, by_shape_T_sub, metric_key

def plateau_score(area_series):
    """
    Given [(area, mean, sem), ...] for small areas (e.g., 1e-10, 3e-10, 1e-9, 3e-9),
    compute coefficient of variation (CV) of the means and check sign consistency.
    Returns (cv, same_sign:boolean)
    """
    vals = [m for (_, m, _) in area_series]
    signs = [math.copysign(1, v) if v != 0 else 0 for v in vals]
    same_sign = all(s == signs[0] for s in signs)
    mu = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    cv = abs(sd / mu) if mu != 0 else float("inf")
    return cv, same_sign

def micro_shape_spread(series_by_shape):
    """
    Given {shape: [(area, mean, sem), ...]} at fixed (T, substeps),
    compute average relative spread across shapes per area.
    """
    areas = sorted(set(a for s in series_by_shape.values() for (a, _, _) in s))
    spreads = []
    for a in areas:
        vals = [m for s in series_by_shape.values() for (aa, m, _) in s if aa == a]
        if len(vals) < 2:
            continue
        mu = mean(vals)
        if mu == 0: 
            continue
        span = (max(vals) - min(vals)) / abs(mu)
        spreads.append(span)
    return mean(spreads) if spreads else float("nan")

def save_summary_csv(path, rows):
    if not rows: 
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {path}")

def analyze_folder(folder, prefer_metric="A_slope_per_area_procrustes"):
    all_results = load_results(folder)
    if not all_results:
        print("No CSVs found.")
        return

    grand_summary = []
    for out_name, rows in all_results.items():
        flat, by_shape_T_sub, metric = summarize(rows, key_delta_over_area=prefer_metric)

        # Export per-run summary
        save_summary_csv(Path(folder) / f"{out_name}_analysis.csv", flat)

        # Human verdicts per (T, substeps)
        verdicts = []
        # Group by T, substeps across shapes
        by_T_sub_shape = defaultdict(lambda: defaultdict(list))
        for (shape, T, subA), series in by_shape_T_sub.items():
            by_T_sub_shape[(T, subA)][shape] = series

        for (T, subA), shape_map in sorted(by_T_sub_shape.items()):
            # plateau per shape
            plateau_ok = True
            same_sign_ok = True
            cvs = []
            for shape, series in shape_map.items():
                cv, same_sign = plateau_score(series)
                cvs.append(cv)
                plateau_ok &= (cv <= 0.2)
                same_sign_ok &= same_sign

            # shape invariance at fixed (T, substeps)
            spread = micro_shape_spread(shape_map)
            shape_ok = (spread <= 0.15)

            verdicts.append({
                "out": out_name, "metric": metric,
                "T": T, "substeps_A": subA,
                "plateau_cv_avg": mean(cvs) if cvs else float("nan"),
                "plateau_pass": plateau_ok and same_sign_ok,
                "shape_spread": spread,
                "shape_pass": shape_ok
            })

        grand_summary.extend(verdicts)
        save_summary_csv(Path(folder) / f"{out_name}_verdicts.csv", verdicts)

    # Grand table
    save_summary_csv(Path(folder) / "ALL_VERDICTS.csv", grand_summary)

if __name__ == "__main__":
    # Example: uncomment to run an entire grid automatically.
    # 1) Plateau T-sweep (balanced)
    # run_one(out="vybn_plateau_T025",  T=0.25,  micro_shapes="balanced")
    # run_one(out="vybn_plateau_T0125", T=0.125, micro_shapes="balanced")
    # run_one(out="vybn_plateau_T00625",T=0.0625,micro_shapes="balanced")
    #
    # 2) Micro-shape invariance at T=0.125
    # run_one(out="vybn_shapes_T0125", T=0.125, micro_shapes="balanced,long_theta,long_r")
    #
    # 3) Commuting control at T=0.125
    # run_one(out="vybn_commute_T0125", T=0.125, micro_shapes="balanced", commute=True)
    #
    # After runs finish, analyze the current directory (or point to a results folder):
    analyze_folder(folder=".")
