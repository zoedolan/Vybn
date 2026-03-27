#!/usr/bin/env python3
"""
experiment_analysis.py

Load saved results from experiment_weight_topology.py and produce:
  - Console summary with statistical tests
  - results_plot.json (bar chart data) for external visualisation
  - per-condition persistence diagram data

Usage:
  python experiment_analysis.py
  python experiment_analysis.py --results_dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "experiment_results"


def load_results(results_dir: Path) -> list:
    results = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            pass
    return results


def analyse(results: list, output_dir: Path):
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    cond_order = ["random", "coherent", "diverse", "order", "synthetic"]
    present = [c for c in cond_order if c in by_cond]

    print("\n" + "=" * 70)
    print("WEIGHT-SPACE TOPOLOGY EXPERIMENT — ANALYSIS")
    print("=" * 70)
    print(f"  Conditions: {present}")
    print(f"  Total runs: {len(results)}")
    print()

    cond_stats = {}
    for cond in present:
        runs = by_cond[cond]
        tp = [r["topology"]["total_persistence_h1"] for r in runs]
        b1 = [r["topology"]["betti_1"] for r in runs]
        ent = [r["topology"]["persistence_entropy_h1"] for r in runs]
        losses = [r["final_loss"] for r in runs]
        cond_stats[cond] = {
            "n": len(runs),
            "tp_h1": tp,
            "b1": b1,
            "entropy": ent,
            "loss": losses,
            "tp_mean": float(np.mean(tp)),
            "tp_std": float(np.std(tp)),
            "b1_mean": float(np.mean(b1)),
            "b1_std": float(np.std(b1)),
            "ent_mean": float(np.mean(ent)),
            "loss_mean": float(np.mean(losses)),
        }
        print(
            f"  {cond:12s}  n={len(runs):2d}  "
            f"tp_h1={np.mean(tp):.4f}±{np.std(tp):.4f}  "
            f"b1={np.mean(b1):.2f}±{np.std(b1):.2f}  "
            f"entropy={np.mean(ent):.4f}  "
            f"loss={np.mean(losses):.4f}"
        )

    # ── Statistical tests ──
    print("\n" + "-" * 70)
    print("STATISTICAL TESTS")
    print("-" * 70)

    real = [c for c in ["random", "coherent", "diverse"] if c in cond_stats]
    if len(real) >= 2:
        try:
            from scipy.stats import kruskal, mannwhitneyu
            groups = [cond_stats[c]["tp_h1"] for c in real]
            if all(len(g) > 1 for g in groups):
                H, p = kruskal(*groups)
                print(f"  Kruskal-Wallis (random/coherent/diverse): H={H:.4f}  p={p:.6f}")
                sig = p < 0.05
                print(f"  Result: {'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'} (alpha=0.05)")

                # Pairwise Mann-Whitney
                print()
                for i, ca in enumerate(real):
                    for cb in real[i+1:]:
                        if len(cond_stats[ca]["tp_h1"]) > 1 and len(cond_stats[cb]["tp_h1"]) > 1:
                            u, pu = mannwhitneyu(
                                cond_stats[ca]["tp_h1"],
                                cond_stats[cb]["tp_h1"],
                                alternative="two-sided"
                            )
                            print(f"  Mann-Whitney {ca} vs {cb}: U={u:.1f}  p={pu:.4f}"
                                  f"  {'*' if pu < 0.05 else ''}")
        except ImportError:
            print("  scipy not available — using variance ratio")
            all_v = [v for c in real for v in cond_stats[c]["tp_h1"]]
            gm = np.mean(all_v)
            bv = np.mean([(np.mean(cond_stats[c]["tp_h1"]) - gm) ** 2 for c in real])
            wv = np.mean([np.var(cond_stats[c]["tp_h1"]) for c in real])
            print(f"  Between/within variance ratio: {bv/(wv+1e-12):.4f}")

    # Order effect
    if "order" in cond_stats:
        ov = cond_stats["order"]["tp_h1"]
        print(f"\n  Order-permutation variance: {float(np.var(ov)):.6f}")
        print(f"  (>0 means reading order changes weight-space topology)")

    # Synthetic control
    if "synthetic" in cond_stats and "random" in cond_stats:
        real_tp = cond_stats["random"]["tp_mean"]
        syn_tp = cond_stats["synthetic"]["tp_mean"]
        diff = real_tp - syn_tp
        print(f"\n  Real vs synthetic tp_h1 difference: {diff:+.4f}")
        if diff > 0.01:
            print("  Real text produces RICHER topology than random sequences.")
            print("  This suggests the topology signal reflects linguistic structure, not just counting.")
        elif abs(diff) <= 0.01:
            print("  Real and synthetic text produce SIMILAR topology.")
            print("  The topology signal may be a counting/dimensionality artifact.")
        else:
            print("  Synthetic text produces RICHER topology than real text (unexpected).")

    # ── Correlation: topology vs loss ──
    print("\n" + "-" * 70)
    print("TOPOLOGY vs LOSS CORRELATION")
    print("-" * 70)
    all_tp = [r["topology"]["total_persistence_h1"] for r in results]
    all_loss = [r["final_loss"] for r in results]
    if len(all_tp) > 3:
        corr = float(np.corrcoef(all_tp, all_loss)[0, 1])
        print(f"  Pearson r(tp_h1, final_loss) = {corr:.4f}")
        if abs(corr) > 0.3:
            sign = "positively" if corr > 0 else "negatively"
            print(f"  Richer topology is {sign} correlated with final loss.")
        else:
            print("  Weak correlation — topology and loss are largely independent.")

    # ── Save plot data ──
    plot_data = {
        "conditions": present,
        "tp_h1_means": [cond_stats[c]["tp_mean"] for c in present],
        "tp_h1_stds": [cond_stats[c]["tp_std"] for c in present],
        "b1_means": [cond_stats[c]["b1_mean"] for c in present],
        "b1_stds": [cond_stats[c]["b1_std"] for c in present],
        "cond_stats": cond_stats,
    }
    plot_path = output_dir / "results_plot_data.json"
    with open(plot_path, "w") as f:
        json.dump(plot_data, f, indent=2, default=str)
    print(f"\n  Plot data saved to: {plot_path}")

    # ── Final verdict ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    try:
        from scipy.stats import kruskal
        groups = [cond_stats[c]["tp_h1"] for c in real if len(cond_stats[c]["tp_h1"]) > 1]
        if len(groups) >= 2:
            _, p = kruskal(*groups)
            if p < 0.05:
                print("YES — text selection measurably affects weight-space topology (p < 0.05).")
                print("The topology of weight space carries information about *what* was learned,")
                print("not just *how much*. The nw fitness component is detecting real structure.")
            else:
                print("NO (p >= 0.05) — text selection does not significantly affect weight-space topology.")
                print("The nw fitness component may reward a counting artifact (more texts → higher Betti).")
                print("Recommendation: redesign nw to normalise by n_snapshots, or remove it.")
        else:
            print("(Insufficient groups for Kruskal-Wallis)")
    except ImportError:
        print("(scipy not available — inspect statistics above)")


def main():
    parser = argparse.ArgumentParser(description="Analyse weight-topology experiment results")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        print("Run experiment_weight_topology.py first.")
        return
    analyse(results, args.results_dir)


if __name__ == "__main__":
    main()
