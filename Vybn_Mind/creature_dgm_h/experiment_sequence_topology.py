#!/usr/bin/env python3
"""
experiment_sequence_topology.py  —  Geometry chapter: weight-space geometry.

Previous attempt used encounter_complex() to measure curvature, but that
probe runs a fixed sentence through a *frozen* external embedder (MiniLM).
It has no path to the creature's weights.  The 0.0 result was guaranteed.

This experiment measures geometry that actually lives inside the creature:

  After each gradient step, compute the singular value spectrum of every
  weight matrix in agent.sd.  Track two quantities:

    anisotropy  = sigma_1 / frobenius_norm   (how directionally biased)
    spectral_entropy = -sum(p_i * log(p_i))  where p_i = sigma_i / sum(sigma)
                       (how spread the variance is across directions)

  Both are pure functions of the creature's weights.  No external embedder.
  No frozen probe.  The weights move when loss drops; these numbers must move
  if weight geometry changes at all.

Question: does the weight-space geometry evolve differently when the creature
is learning meaningful text versus memorising random noise?

Verdict logic:
  - Real and synthetic trajectories diverge in anisotropy or entropy
    -> weight geometry encodes what is being learned; signal confirmed
  - Trajectories indistinguishable
    -> close this line of inquiry; the creature is too uniform at this scale
  - Mixed
    -> inspect per-matrix and per-seed curves

Usage:
  python experiment_sequence_topology.py          # full run (5 seeds)
  python experiment_sequence_topology.py --quick  # 2 seeds
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
from typing import List, Dict

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "spark"))
sys.path.insert(0, str(SCRIPT_DIR))

from vybn import (
    TopoAgent,
    CORPUS_PATH,
    RV, N_EMBD, N_LAYER, N_HEAD, HEAD_DIM, BLOCK_SIZE,
    _forward, _softmax,
)

# ── Config ────────────────────────────────────────────────────────────────
K              = 3
STEPS_PER_TEXT = 15
LR             = 0.01
N_SEEDS        = 5
RESULTS_DIR    = SCRIPT_DIR / "experiment_results" / "weight_geometry"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Weight-space geometry probe ───────────────────────────────────────────

def weight_geometry(agent: TopoAgent) -> Dict[str, float]:
    """Measure anisotropy and spectral entropy of the creature's weight matrices.

    Only 2-D weight matrices are meaningful for SVD (embeddings, projections).
    Scalar/1-D params (biases, norms) are skipped.

    anisotropy     = sigma_1 / frobenius_norm  in [0, 1]
                     near 1 -> one direction dominates (spiky)
                     near 0 -> uniform (flat)

    spectral_entropy = -sum(p_i * log(p_i + 1e-12))  where p_i = sigma_i / sum
                     high -> variance spread across many directions
                     low  -> collapsed onto few directions
    """
    anisotropies: List[float] = []
    entropies:    List[float] = []

    for key, mat in agent.sd.items():
        # mat is a list-of-lists; convert to numpy
        try:
            arr = np.array([[p.data if hasattr(p, 'data') else float(p)
                             for p in row]
                            for row in mat], dtype=np.float64)
        except (TypeError, ValueError):
            continue

        if arr.ndim != 2 or min(arr.shape) < 2:
            continue

        try:
            sv = np.linalg.svd(arr, compute_uv=False)
        except np.linalg.LinAlgError:
            continue

        frob = float(np.linalg.norm(arr, 'fro'))
        if frob < 1e-12:
            continue

        anisotropies.append(float(sv[0]) / frob)

        sv_sum = float(sv.sum())
        if sv_sum > 1e-12:
            p   = sv / sv_sum
            ent = float(-np.sum(p * np.log(p + 1e-12)))
            entropies.append(ent)

    if not anisotropies:
        return {"anisotropy": 0.0, "spectral_entropy": 0.0, "n_matrices": 0}

    return {
        "anisotropy":      round(float(np.mean(anisotropies)), 8),
        "spectral_entropy": round(float(np.mean(entropies)),   8),
        "n_matrices":      len(anisotropies),
    }


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

    agent               = TopoAgent(config={"learn_lr": LR})
    aniso_traj:   List[float] = []
    entropy_traj: List[float] = []
    loss_traj:    List[float] = []
    step_global   = 0

    for text in texts:
        clean = agent._clean(text)
        if len(clean) < 2:
            continue
        tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
        n      = min(BLOCK_SIZE, len(tokens) - 1)

        for _ in range(STEPS_PER_TEXT):
            loss_val = _gradient_step(agent, tokens, n)
            loss_traj.append(round(loss_val, 6))

            g = weight_geometry(agent)
            aniso_traj.append(g["anisotropy"])
            entropy_traj.append(g["spectral_entropy"])
            step_global += 1

    aniso   = np.array(aniso_traj)
    entropy = np.array(entropy_traj)
    losses  = np.array(loss_traj)

    def _slope(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(arr), dtype=float), arr, 1)[0])

    def _drift(arr: np.ndarray) -> float:
        mid = len(arr) // 2
        if mid == 0 or mid >= len(arr):
            return 0.0
        return round(float(arr[mid:].mean() - arr[:mid].mean()), 8)

    return {
        "experiment":          "weight_geometry",
        "condition":           condition,
        "run_idx":             run_idx,
        "seed":                seed,
        "n_steps":             step_global,
        "n_matrices":          weight_geometry(agent)["n_matrices"],
        "anisotropy_trajectory":  [round(v, 8) for v in aniso_traj],
        "entropy_trajectory":     [round(v, 8) for v in entropy_traj],
        "loss_trajectory":        loss_traj,
        "anisotropy_mean":     round(float(aniso.mean()),   8) if len(aniso)   else 0.0,
        "anisotropy_slope":    round(_slope(aniso),         10),
        "anisotropy_drift":    _drift(aniso),
        "entropy_mean":        round(float(entropy.mean()), 8) if len(entropy) else 0.0,
        "entropy_slope":       round(_slope(entropy),       10),
        "entropy_drift":       _drift(entropy),
        "loss_improvement":    round(float(losses[0] - losses[-1]), 6) if len(losses) > 1 else 0.0,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    }


# ── Main experiment ───────────────────────────────────────────────────────

def run_experiment(
    n_seeds:   int = N_SEEDS,
    k:         int = K,
    seed_base: int = 42,
) -> List[dict]:
    corpus  = load_corpus()
    avg_len = int(np.mean([len(t) for t in corpus[:20]])) if corpus else 200

    print("[Weight-geometry experiment]")
    print(f"Corpus: {len(corpus)} passages  K={k}  STEPS={STEPS_PER_TEXT}  seeds={n_seeds}")
    print(f"Probe: SVD anisotropy + spectral entropy of creature weight matrices")
    print(f"Total measurements/run: {k * STEPS_PER_TEXT}")
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
            f"aniso={r['anisotropy_mean']:.6f}  drift={r['anisotropy_drift']:+.6f}  "
            f"entropy={r['entropy_mean']:.4f}  loss_imp={r['loss_improvement']:+.4f}"
        )
        (RESULTS_DIR / f"real_{seed_offset:03d}.json").write_text(
            json.dumps(r, indent=2, default=str)
        )

        syn_texts = [synthetic_text(avg_len, seed=seed * 1000 + j) for j in range(k)]
        r2 = run_condition(syn_texts, seed, "synthetic", seed_offset)
        all_results.append(r2)
        print(
            f"  seed {seed}  synthetic | "
            f"aniso={r2['anisotropy_mean']:.6f}  drift={r2['anisotropy_drift']:+.6f}  "
            f"entropy={r2['entropy_mean']:.4f}  loss_imp={r2['loss_improvement']:+.4f}"
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
    print("WEIGHT-GEOMETRY EXPERIMENT — RESULTS SUMMARY")
    print("=" * 70)

    for cond in ("real", "synthetic"):
        runs = by_cond.get(cond, [])
        if not runs:
            continue
        am = [r["anisotropy_mean"] for r in runs]
        ad = [r["anisotropy_drift"] for r in runs]
        em = [r["entropy_mean"]     for r in runs]
        print(
            f"  {cond:10s}: n={len(runs)}  "
            f"aniso={np.mean(am):.6f}±{np.std(am):.6f}  "
            f"drift={np.mean(ad):+.6f}±{np.std(ad):.6f}  "
            f"entropy={np.mean(em):.4f}±{np.std(em):.4f}"
        )

    real_am  = [r["anisotropy_mean"]  for r in by_cond.get("real", [])]
    syn_am   = [r["anisotropy_mean"]  for r in by_cond.get("synthetic", [])]
    real_em  = [r["entropy_mean"]     for r in by_cond.get("real", [])]
    syn_em   = [r["entropy_mean"]     for r in by_cond.get("synthetic", [])]
    real_ad  = [r["anisotropy_drift"] for r in by_cond.get("real", [])]
    syn_ad   = [r["anisotropy_drift"] for r in by_cond.get("synthetic", [])]

    print()
    if real_am and syn_am:
        aniso_diff   = np.mean(real_am) - np.mean(syn_am)
        entropy_diff = np.mean(real_em) - np.mean(syn_em)
        drift_diff   = np.mean(real_ad) - np.mean(syn_ad)
        print(f"  Real − synthetic anisotropy_mean: {aniso_diff:+.6f}")
        print(f"  Real − synthetic entropy_mean:    {entropy_diff:+.6f}")
        print(f"  Real − synthetic drift:           {drift_diff:+.6f}")
        print()

        aniso_sig   = abs(aniso_diff)   > 0.005
        entropy_sig = abs(entropy_diff) > 0.05
        drift_sig   = abs(drift_diff)   > 0.002

        if aniso_sig or entropy_sig or drift_sig:
            print("  VERDICT: weight-space geometry differs between conditions.")
            print("  The creature's weight matrices evolve differently under real vs synthetic text.")
            sigs = []
            if aniso_sig:   sigs.append(f"anisotropy diff {aniso_diff:+.4f}")
            if entropy_sig: sigs.append(f"entropy diff {entropy_diff:+.4f}")
            if drift_sig:   sigs.append(f"drift diff {drift_diff:+.4f}")
            print(f"  Signals: {', '.join(sigs)}")
            print("  Geometry encodes what is being learned.")
        else:
            print("  VERDICT: weight-space geometry is indistinguishable between conditions.")
            print("  The creature reorganises its weights the same way regardless of input.")
            print("  Close this chapter. The signal is not here at this scale.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weight-geometry experiment: does SVD anisotropy / spectral entropy "
                    "of the creature's weights evolve differently under real vs synthetic text?"
    )
    parser.add_argument("--quick",   action="store_true", help="2 seeds only")
    parser.add_argument("--analyze", action="store_true", help="Re-summarise saved results")
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
                 if k not in ("anisotropy_trajectory", "entropy_trajectory", "loss_trajectory")}
                for r in results
            ],
            indent=2, default=str,
        )
    )
    print(f"\nResults -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
