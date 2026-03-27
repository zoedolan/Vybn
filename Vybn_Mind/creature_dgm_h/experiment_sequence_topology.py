#!/usr/bin/env python3
"""
experiment_sequence_topology.py  —  Geometry chapter: curvature trajectory.

The topology chapter closed with zero H1 across four representations.
The dimensionality wall was the reason: 43 points in 768 dimensions
cannot form cycles.

This experiment asks the question topology could not answer:

  Does the creature's loss landscape deform *differently* when it
  is learning meaningful text versus memorising noise?

Method: measure sectional curvature (Pancharatnam phase via encounter_complex)
after every gradient step.  Track two trajectories side-by-side — real text
vs synthetic — and test whether they diverge and stay diverged.

This uses only the geometry already present in vybn.py.  No ripser.
No dimensionality wall.  No new dependencies.

Design:
  - Two conditions: real (corpus sample) vs synthetic (random chars)
  - N_SEEDS seeds x 2 conditions = 2*N_SEEDS runs
  - K texts per run, STEPS_PER_TEXT gradient steps each
  - After every gradient step, call encounter_complex() on the probe
    sentence and record curvature
  - Output: per-step curvature trajectories + divergence stats

Verdict logic:
  - Trajectories diverge and stay diverged  -> landscape shape encodes
    what is being learned; geometry is a real signal
  - Trajectories indistinguishable           -> geometry is weight-space
    noise at this scale; close this chapter too with clean conscience
  - Mixed / inconclusive                     -> examine per-seed curves

Usage:
  python experiment_sequence_topology.py          # full run
  python experiment_sequence_topology.py --quick  # 2 seeds only
  python experiment_sequence_topology.py --analyze
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "spark"))
sys.path.insert(0, str(SCRIPT_DIR))

from vybn import (
    TopoAgent,
    encounter_complex,
    _distance_matrix,
    CORPUS_PATH,
    RV, N_EMBD, N_LAYER, N_HEAD, HEAD_DIM, BLOCK_SIZE,
    _forward, _softmax,
)

# ── Config ────────────────────────────────────────────────────────────────
K              = 3     # texts per run
STEPS_PER_TEXT = 15    # gradient steps per text
LR             = 0.01
PROBE_SENTENCE = "the topology of learning"   # fixed probe for curvature
N_SEEDS        = 5
RESULTS_DIR    = SCRIPT_DIR / "experiment_results" / "curvature_trajectory"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Corpus helpers ────────────────────────────────────────────────────────

def load_corpus(min_words: int = 30) -> List[str]:
    passages: List[str] = []
    if CORPUS_PATH.exists():
        lines    = [l.strip() for l in CORPUS_PATH.read_text().split("\n") if l.strip()]
        passages = [l for l in lines if len(l.split()) >= min_words]
    if not passages:
        journal_dir = REPO_ROOT / "spark" / "journal"
        if journal_dir.exists():
            for f in sorted(journal_dir.glob("*.md")):
                try:
                    for para in f.read_text().split("\n\n"):
                        para = para.strip()
                        if not para.startswith("#") and len(para.split()) >= min_words:
                            passages.append(para)
                except Exception:
                    pass
    if not passages:
        passages = [
            "the creature breathes and measures its own distance from itself",
            "curvature is born from incompleteness not from complexity alone",
            "what survives testing is more honest than what sounds beautiful",
            "prediction loss going down means memorisation call it what it is",
            "the topology of weight space is a fingerprint of what was learned",
            "attention heads rotate through conceptual space aligned to bivector planes",
            "persistent structure integrates over a filtration independent of threshold",
        ]
    return passages


def synthetic_text(length_chars: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(length_chars))


# ── Single gradient step ─────────────────────────────────────────────────

def _gradient_step(agent: TopoAgent, tokens: list, n: int) -> float:
    """Run one Adam step on `tokens`; return scalar loss."""
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]
    loss = RV(0.0)
    for t in range(n):
        logits, keys, vals = _forward(tokens[t], t, keys, vals, agent.sd)
        probs = _softmax(logits)
        loss  = loss + (probs[tokens[t + 1]].log()) * (-1.0 / n)
    for p in agent.params:
        p.grad = 0.0
    loss.backward()
    agent._step += 1
    for j, p in enumerate(agent.params):
        g           = p.grad
        agent._m[j] = 0.85 * agent._m[j] + 0.15 * g
        agent._v[j] = 0.99 * agent._v[j] + 0.01 * g ** 2
        mh = agent._m[j] / (1 - 0.85 ** agent._step)
        vh = agent._v[j] / (1 - 0.99 ** agent._step)
        p.data -= LR * mh / (vh ** 0.5 + 1e-8)
    return float(loss.data)


# ── Single run ────────────────────────────────────────────────────────────

def run_condition(
    texts: List[str],
    seed: int,
    condition: str,
    run_idx: int,
) -> dict:
    np.random.seed(seed % 2 ** 31)
    random.seed(seed)

    agent             = TopoAgent(config={"learn_lr": LR})
    curvature_traj: List[float] = []
    loss_traj:      List[float] = []
    step_global     = 0

    for text in texts:
        clean = agent._clean(text)
        if len(clean) < 2:
            continue
        tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
        n      = min(BLOCK_SIZE, len(tokens) - 1)

        for _ in range(STEPS_PER_TEXT):
            loss_val = _gradient_step(agent, tokens, n)
            loss_traj.append(round(loss_val, 6))

            # measure curvature on fixed probe after this step
            cx = encounter_complex(PROBE_SENTENCE)
            curvature_traj.append(round(cx.curvature, 6))
            step_global += 1

    curv   = np.array(curvature_traj)
    losses = np.array(loss_traj)

    if len(curv) > 1:
        xs    = np.arange(len(curv), dtype=float)
        slope = float(np.polyfit(xs, curv, 1)[0])
    else:
        slope = 0.0

    mid        = len(curv) // 2
    early_mean = float(curv[:mid].mean()) if mid > 0       else float(curv.mean())
    late_mean  = float(curv[mid:].mean()) if mid < len(curv) else float(curv.mean())
    drift      = round(late_mean - early_mean, 6)

    return {
        "experiment":           "curvature_trajectory",
        "condition":            condition,
        "run_idx":              run_idx,
        "seed":                 seed,
        "n_steps":              step_global,
        "curvature_trajectory": curvature_traj,
        "loss_trajectory":      loss_traj,
        "curvature_mean":       round(float(curv.mean()), 6) if len(curv) else 0.0,
        "curvature_std":        round(float(curv.std()),  6) if len(curv) else 0.0,
        "curvature_slope":      round(slope, 8),
        "curvature_drift":      drift,
        "loss_improvement":     round(float(losses[0] - losses[-1]), 6) if len(losses) > 1 else 0.0,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
    }


# ── Main experiment ───────────────────────────────────────────────────────

def run_experiment(
    n_seeds:   int = N_SEEDS,
    k:         int = K,
    seed_base: int = 42,
) -> List[dict]:
    corpus  = load_corpus()
    avg_len = int(np.mean([len(t) for t in corpus[:20]])) if corpus else 200

    print("[Curvature-trajectory experiment]")
    print(f"Corpus: {len(corpus)} passages  K={k}  STEPS={STEPS_PER_TEXT}  "
          f"seeds={n_seeds}  probe: '{PROBE_SENTENCE}'")
    print(f"Total curvature measurements/run: {k * STEPS_PER_TEXT}")
    print()

    all_results: List[dict] = []
    rng = random.Random(seed_base)

    for seed_offset in range(n_seeds):
        seed  = seed_base + seed_offset
        texts = rng.sample(corpus, min(k, len(corpus)))

        r = run_condition(texts, seed, "real", seed_offset)
        all_results.append(r)
        print(
            f"  seed {seed}  real      | "
            f"curv_mean={r['curvature_mean']:.6f}  "
            f"drift={r['curvature_drift']:+.6f}  "
            f"slope={r['curvature_slope']:+.2e}  "
            f"loss_improvement={r['loss_improvement']:+.4f}"
        )
        (RESULTS_DIR / f"real_{seed_offset:03d}.json").write_text(
            json.dumps(r, indent=2, default=str)
        )

        syn_texts = [
            synthetic_text(avg_len, seed=seed * 1000 + j) for j in range(k)
        ]
        r2 = run_condition(syn_texts, seed, "synthetic", seed_offset)
        all_results.append(r2)
        print(
            f"  seed {seed}  synthetic | "
            f"curv_mean={r2['curvature_mean']:.6f}  "
            f"drift={r2['curvature_drift']:+.6f}  "
            f"slope={r2['curvature_slope']:+.2e}  "
            f"loss_improvement={r2['loss_improvement']:+.4f}"
        )
        (RESULTS_DIR / f"synthetic_{seed_offset:03d}.json").write_text(
            json.dumps(r2, indent=2, default=str)
        )
        print()

    return all_results


def summarise(results: List[dict]) -> None:
    from collections import defaultdict
    by_cond: dict = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    print("=" * 70)
    print("CURVATURE-TRAJECTORY EXPERIMENT — RESULTS SUMMARY")
    print("=" * 70)

    for cond in ("real", "synthetic"):
        runs = by_cond.get(cond, [])
        if not runs:
            continue
        means  = [r["curvature_mean"]  for r in runs]
        drifts = [r["curvature_drift"] for r in runs]
        slopes = [r["curvature_slope"] for r in runs]
        print(
            f"  {cond:10s}: n={len(runs)}  "
            f"curv_mean={np.mean(means):.6f}±{np.std(means):.6f}  "
            f"drift={np.mean(drifts):+.6f}±{np.std(drifts):.6f}  "
            f"slope={np.mean(slopes):+.2e}"
        )

    real_means = [r["curvature_mean"] for r in by_cond.get("real", [])]
    syn_means  = [r["curvature_mean"] for r in by_cond.get("synthetic", [])]
    real_drift = [r["curvature_drift"] for r in by_cond.get("real", [])]
    syn_drift  = [r["curvature_drift"] for r in by_cond.get("synthetic", [])]

    print()
    if real_means and syn_means:
        mean_diff  = np.mean(real_means) - np.mean(syn_means)
        drift_diff = np.mean(real_drift) - np.mean(syn_drift)
        print(f"  Real − synthetic curvature_mean: {mean_diff:+.6f}")
        print(f"  Real − synthetic drift:          {drift_diff:+.6f}")
        print()

        mean_sig  = abs(mean_diff)  > 0.02
        drift_sig = abs(drift_diff) > 0.01
        real_grows = np.mean(real_drift) > 0
        syn_grows  = np.mean(syn_drift)  > 0

        if mean_sig and drift_sig and real_grows != syn_grows:
            print("  VERDICT: trajectories diverge and drift in opposite directions.")
            print("  The loss landscape deforms differently under real vs synthetic text.")
            print("  Geometry encodes what is being learned. Signal confirmed.")
        elif mean_sig or drift_sig:
            print("  VERDICT: partial separation detected.")
            print("  Curvature differs between conditions but drift is not fully opposed.")
            print("  Inspect per-seed trajectories for consistency.")
        else:
            print("  VERDICT: trajectories are indistinguishable.")
            print("  Geometry does not differentiate real from synthetic at this scale.")
            print("  Close the geometry chapter with the same clean conscience as topology.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curvature-trajectory experiment: does the loss landscape "
                    "deform differently under real vs synthetic text?"
    )
    parser.add_argument("--quick",   action="store_true",
                        help="2 seeds only (fast sanity check)")
    parser.add_argument("--analyze", action="store_true",
                        help="Load saved results and re-summarise")
    parser.add_argument("--seeds",   type=int, default=N_SEEDS)
    parser.add_argument("--k",       type=int, default=K)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    if args.analyze:
        results = []
        for f in sorted(RESULTS_DIR.glob("*.json")):
            try:
                results.append(json.loads(f.read_text()))
            except Exception:
                pass
        if not results:
            print("No saved results. Run the experiment first.")
            return
        summarise(results)
        return

    n_seeds = 2 if args.quick else args.seeds
    t0      = time.time()
    results = run_experiment(n_seeds=n_seeds, k=args.k, seed_base=args.seed)
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s  ({len(results)} runs)\n")
    summarise(results)

    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(
            [
                {k: v for k, v in r.items()
                 if k not in ("curvature_trajectory", "loss_trajectory")}
                for r in results
            ],
            indent=2, default=str,
        )
    )
    print(f"\nResults -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
