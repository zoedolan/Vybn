#!/usr/bin/env python3
"""
experiment_analysis.py

Unified analysis for both topology experiment approaches:
  - PCA-first persistence (experiment_weight_topology.py)
  - Activation-space persistence (experiment_activation_topology.py)

Loads results from either or both experiment directories, produces:
  - Console summary with statistical tests
  - results_plot_data.json for external visualisation
  - Comparative report when both experiments have data

Usage:
  python experiment_analysis.py                          # analyse all available results
  python experiment_analysis.py --experiment pca         # PCA-first only
  python experiment_analysis.py --experiment activation  # activation-space only
  python experiment_analysis.py --results_dir /path      # custom directory
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PCA_RESULTS_DIR = SCRIPT_DIR / "experiment_results" / "pca_topology"
ACTIVATION_RESULTS_DIR = SCRIPT_DIR / "experiment_results" / "activation_topology"
# Legacy: flat experiment_results/ from the old raw weight-space experiment
LEGACY_RESULTS_DIR = SCRIPT_DIR / "experiment_results"


def load_results(results_dir: Path) -> list:
    results = []
    if not results_dir.exists():
        return results
    for f in sorted(results_dir.glob("*.json")):
        if f.name in ("summary.json", "results_plot_data.json"):
            continue
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            pass
    return results


def analyse_single(results: list, experiment_name: str, output_dir: Path) -> dict:
    """Analyse results from one experiment type. Returns cond_stats dict."""
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    cond_order = ["random", "coherent", "diverse", "order", "synthetic"]
    present = [c for c in cond_order if c in by_cond]

    print(f"\n{'=' * 70}")
    print(f"{experiment_name.upper()} EXPERIMENT — ANALYSIS")
    print(f"{'=' * 70}")
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
        extra = {}
        if "variance_explained" in runs[0].get("topology", {}):
            var_expl = [r["topology"]["variance_explained"] for r in runs]
            extra["var_explained_mean"] = float(np.mean(var_expl))
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
            **extra,
        }
        var_str = f"  var_expl={extra['var_explained_mean']:.2f}" if "var_explained_mean" in extra else ""
        print(
            f"  {cond:12s}  n={len(runs):2d}  "
            f"tp_h1={np.mean(tp):.4f}±{np.std(tp):.4f}  "
            f"b1={np.mean(b1):.2f}±{np.std(b1):.2f}  "
            f"entropy={np.mean(ent):.4f}  "
            f"loss={np.mean(losses):.4f}{var_str}"
        )

    # ── Statistical tests ──
    print(f"\n{'-' * 70}")
    print("STATISTICAL TESTS")
    print(f"{'-' * 70}")

    real = [c for c in ["random", "coherent", "diverse"] if c in cond_stats]
    kw_p = None
    if len(real) >= 2:
        try:
            from scipy.stats import kruskal, mannwhitneyu
            groups = [cond_stats[c]["tp_h1"] for c in real]
            if all(len(g) > 1 for g in groups):
                H, p = kruskal(*groups)
                kw_p = p
                print(f"  Kruskal-Wallis (random/coherent/diverse): H={H:.4f}  p={p:.6f}")
                print(f"  Result: {'SIGNIFICANT' if p < 0.05 else 'NOT SIGNIFICANT'} (alpha=0.05)")
                print()
                for i, ca in enumerate(real):
                    for cb in real[i + 1:]:
                        if len(cond_stats[ca]["tp_h1"]) > 1 and len(cond_stats[cb]["tp_h1"]) > 1:
                            u, pu = mannwhitneyu(
                                cond_stats[ca]["tp_h1"],
                                cond_stats[cb]["tp_h1"],
                                alternative="two-sided",
                            )
                            print(f"  Mann-Whitney {ca} vs {cb}: U={u:.1f}  p={pu:.4f}"
                                  f"  {'*' if pu < 0.05 else ''}")
        except ImportError:
            all_v = [v for c in real for v in cond_stats[c]["tp_h1"]]
            gm = np.mean(all_v)
            bv = np.mean([(np.mean(cond_stats[c]["tp_h1"]) - gm) ** 2 for c in real])
            wv = np.mean([np.var(cond_stats[c]["tp_h1"]) for c in real])
            print(f"  Between/within variance ratio: {bv / (wv + 1e-12):.4f}")

    if "order" in cond_stats:
        ov = cond_stats["order"]["tp_h1"]
        print(f"\n  Order-permutation variance: {float(np.var(ov)):.6f}")

    if "synthetic" in cond_stats and "random" in cond_stats:
        real_tp = cond_stats["random"]["tp_mean"]
        syn_tp = cond_stats["synthetic"]["tp_mean"]
        diff = real_tp - syn_tp
        print(f"\n  Real vs synthetic tp_h1 difference: {diff:+.4f}")
        if diff > 0.01:
            print("  Real text produces RICHER topology than random sequences.")
        elif abs(diff) <= 0.01:
            print("  Real and synthetic text produce SIMILAR topology.")
        else:
            print("  Synthetic text produces RICHER topology (unexpected).")

    # ── Correlation: topology vs loss ──
    print(f"\n{'-' * 70}")
    print("TOPOLOGY vs LOSS CORRELATION")
    print(f"{'-' * 70}")
    all_tp = [r["topology"]["total_persistence_h1"] for r in results]
    all_loss = [r["final_loss"] for r in results]
    if len(all_tp) > 3:
        corr = float(np.corrcoef(all_tp, all_loss)[0, 1])
        print(f"  Pearson r(tp_h1, final_loss) = {corr:.4f}")

    # ── Save plot data ──
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data = {
        "experiment": experiment_name,
        "conditions": present,
        "tp_h1_means": [cond_stats[c]["tp_mean"] for c in present],
        "tp_h1_stds": [cond_stats[c]["tp_std"] for c in present],
        "b1_means": [cond_stats[c]["b1_mean"] for c in present],
        "b1_stds": [cond_stats[c]["b1_std"] for c in present],
    }
    plot_path = output_dir / "results_plot_data.json"
    with open(plot_path, "w") as f:
        json.dump(plot_data, f, indent=2, default=str)
    print(f"\n  Plot data saved to: {plot_path}")

    # ── Verdict ──
    print(f"\n{'=' * 70}")
    print(f"VERDICT ({experiment_name})")
    print(f"{'=' * 70}")
    if kw_p is not None:
        if kw_p < 0.05:
            print(f"YES — text selection significantly affects {experiment_name} topology (p < 0.05).")
        else:
            print(f"NO (p >= 0.05) — text selection does not significantly affect {experiment_name} topology.")
    else:
        print("(Insufficient data or scipy unavailable for verdict)")

    return cond_stats


def compare_experiments(pca_stats: dict, act_stats: dict):
    """Print side-by-side comparison when both experiments have results."""
    print(f"\n{'=' * 70}")
    print("CROSS-EXPERIMENT COMPARISON")
    print(f"{'=' * 70}")

    conds = sorted(set(pca_stats.keys()) & set(act_stats.keys()))
    if not conds:
        print("  No overlapping conditions to compare.")
        return

    print(f"  {'Condition':12s}  {'PCA tp_h1':>12s}  {'Act tp_h1':>12s}  {'PCA b1':>8s}  {'Act b1':>8s}")
    print(f"  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 8}  {'-' * 8}")
    for c in conds:
        p, a = pca_stats[c], act_stats[c]
        print(
            f"  {c:12s}  {p['tp_mean']:8.4f}±{p['tp_std']:.3f}  "
            f"{a['tp_mean']:8.4f}±{a['tp_std']:.3f}  "
            f"{p['b1_mean']:6.2f}±{p['b1_std']:.1f}  "
            f"{a['b1_mean']:6.2f}±{a['b1_std']:.1f}"
        )

    # Which approach shows more inter-condition variance?
    for name, stats in [("PCA-first", pca_stats), ("Activation", act_stats)]:
        real = [c for c in ["random", "coherent", "diverse"] if c in stats]
        if len(real) >= 2:
            means = [stats[c]["tp_mean"] for c in real]
            spread = max(means) - min(means)
            print(f"\n  {name} inter-condition tp_h1 spread: {spread:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyse topology experiment results")
    parser.add_argument("--experiment", choices=["pca", "activation", "both"], default="both",
                        help="Which experiment to analyse (default: both)")
    parser.add_argument("--results_dir", type=Path, default=None,
                        help="Override results directory (for single-experiment mode)")
    args = parser.parse_args()

    pca_stats, act_stats = None, None

    if args.experiment in ("pca", "both"):
        d = args.results_dir if args.results_dir and args.experiment == "pca" else PCA_RESULTS_DIR
        results = load_results(d)
        if results:
            pca_stats = analyse_single(results, "PCA-first persistence", d)
        elif args.experiment == "pca":
            print(f"No PCA results found in {d}")

    if args.experiment in ("activation", "both"):
        d = args.results_dir if args.results_dir and args.experiment == "activation" else ACTIVATION_RESULTS_DIR
        results = load_results(d)
        if results:
            act_stats = analyse_single(results, "Activation-space persistence", d)
        elif args.experiment == "activation":
            print(f"No activation results found in {d}")

    if pca_stats and act_stats:
        compare_experiments(pca_stats, act_stats)

    if not pca_stats and not act_stats:
        print("No results found. Run one of:")
        print("  python experiment_weight_topology.py   # PCA-first approach")
        print("  python experiment_activation_topology.py  # activation-space approach")


if __name__ == "__main__":
    main()
