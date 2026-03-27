#!/usr/bin/env python3
"""
experiment_weight_topology.py

Controlled experiment: does text *selection* affect weight-space topology?

Addresses the spec in spark/journal/controlled_experiment_spec.md.

Five conditions:
  1. random     -- 20 runs, K texts sampled randomly from corpus
  2. coherent   -- 10 runs, K texts from the same topical cluster
  3. diverse    -- 10 runs, K texts maximising pairwise embedding distance
  4. order      -- 10 runs, one fixed set of K texts, permuted orderings
  5. synthetic  -- 10 runs, K random-byte sequences of matched length

Key design choices (from the spec review):
- Snapshot every SNAP_EVERY gradient steps, not just once per text.
  This gives ~K*E/SNAP_EVERY points in weight space — much richer topology.
- Uses ripser if available, falls back to the greedy union-find in vybn.py.
- Reports total_persistence (integral over filtration), not just Betti numbers
  at a single threshold.  Also persistence entropy.
- Saves each run to JSON; a companion analysis() function computes
  Kruskal-Wallis + pairwise "pseudo-Wasserstein" distances between conditions.

Usage:
  python experiment_weight_topology.py            # full experiment, ~10-30 min
  python experiment_weight_topology.py --quick    # 3/3/3/3/3 runs, fast smoke test
  python experiment_weight_topology.py --analyze  # load saved results and analyse
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "spark"))

from vybn import (
    TopoAgent,
    encounter_complex,
    embed,
    _persistence_pairs,
    _distance_matrix,
    CORPUS_PATH,
    ARCHIVE_DIR,
)

# ── Config ────────────────────────────────────────────────────────────────
K = 5                  # texts per run
SNAP_EVERY = 5         # snapshot weight vector every N gradient steps
STEPS_PER_TEXT = 10    # gradient steps per text encounter
LR = 0.01
RESULTS_DIR = SCRIPT_DIR / "experiment_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Try to import ripser; fall back to built-in ───────────────────────────
try:
    from ripser import ripser as _ripser_fn
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


# ── Topology utilities ────────────────────────────────────────────────────

def compute_topology(points: np.ndarray) -> dict:
    """
    Given an (N, D) array of weight-space snapshots, compute:
    - betti_0, betti_1       (at median filtration threshold)
    - total_persistence_h0   (sum of finite H0 lifetimes)
    - total_persistence_h1   (sum of finite H1 lifetimes)
    - persistence_entropy_h1 (Shannon entropy of H1 lifetime distribution)
    - diagram_h1             (list of [birth, death] for H1, serialisable)
    """
    n = len(points)
    if n < 3:
        return {
            "betti_0": n, "betti_1": 0,
            "total_persistence_h0": 0.0, "total_persistence_h1": 0.0,
            "persistence_entropy_h1": 0.0,
            "diagram_h1": [], "n_points": n,
            "method": "trivial",
        }

    if RIPSER_AVAILABLE:
        result = _ripser_fn(points, maxdim=1)
        dgms = result["dgms"]
        h0 = dgms[0]   # shape (n_pairs, 2)
        h1 = dgms[1]   # shape (n_pairs, 2)

        # Total persistence: sum of finite lifetimes
        h0_finite = h0[h0[:, 1] < np.inf]
        tp_h0 = float(np.sum(h0_finite[:, 1] - h0_finite[:, 0])) if len(h0_finite) else 0.0
        h1_finite = h1[h1[:, 1] < np.inf]
        tp_h1 = float(np.sum(h1_finite[:, 1] - h1_finite[:, 0])) if len(h1_finite) else 0.0

        # Betti at median threshold: count pairs alive at median distance
        all_dists = []
        for d0, d1 in h0:
            all_dists.append(d0)
        for d0, d1 in h1:
            all_dists.append(d0)
        if all_dists:
            med = float(np.median(all_dists))
            b0 = int(np.sum((h0[:, 0] <= med) & (h0[:, 1] > med)))
            b1 = int(np.sum((h1[:, 0] <= med) & (h1[:, 1] > med)))
        else:
            b0, b1 = n, 0

        # Persistence entropy of H1
        if len(h1_finite) > 0:
            lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
            total = np.sum(lifetimes)
            if total > 1e-12:
                probs = lifetimes / total
                ent = float(-np.sum(probs * np.log(probs + 1e-15)))
            else:
                ent = 0.0
        else:
            ent = 0.0

        diagram_h1 = h1.tolist()
        method = "ripser"
    else:
        # Fallback: built-in greedy filtration
        D = _distance_matrix(points)
        pairs, (b0, b1, _) = _persistence_pairs(D)
        finite = [(birth, death) for birth, death in pairs if death != float("inf")]
        tp_h0 = sum(d - b for b, d in finite)
        # The built-in doesn't cleanly separate H0/H1, so approximate
        tp_h1 = 0.0
        ent = 0.0
        diagram_h1 = finite
        method = "builtin"

    return {
        "betti_0": b0, "betti_1": b1,
        "total_persistence_h0": round(tp_h0, 6),
        "total_persistence_h1": round(tp_h1, 6),
        "persistence_entropy_h1": round(ent, 6),
        "diagram_h1": diagram_h1,
        "n_points": n,
        "method": method,
    }


def pseudo_wasserstein(diag_a: list, diag_b: list) -> float:
    """
    Bottleneck-approximation between two H1 persistence diagrams.
    Each diagram is a list of [birth, death] pairs.
    We use the L∞ distance between sorted lifetime vectors, padded with zeros.
    This is a fast approximation; ripser.persim gives exact Wasserstein.
    """
    def lifetimes(diag):
        ls = []
        for pair in diag:
            b, d = pair[0], pair[1]
            if d < 1e9:  # treat inf as very large
                ls.append(d - b)
        return sorted(ls, reverse=True)

    la, lb = lifetimes(diag_a), lifetimes(diag_b)
    max_len = max(len(la), len(lb), 1)
    la += [0.0] * (max_len - len(la))
    lb += [0.0] * (max_len - len(lb))
    return float(max(abs(a - b) for a, b in zip(la, lb)))


# ── Weight-vector extraction ──────────────────────────────────────────────

def weight_vector(agent: TopoAgent) -> np.ndarray:
    """Flatten all parameters of the agent's network into a single vector."""
    return np.concatenate([
        np.array([[p.data for p in row] for row in mat]).ravel()
        for mat in agent.sd.values()
    ])


# ── Single run ────────────────────────────────────────────────────────────

def run_condition(
    texts: List[str],
    seed: int,
    label: str,
    condition: str,
    run_idx: int,
) -> dict:
    """
    Train a fresh agent on `texts` (in order), snapshotting weights every
    SNAP_EVERY gradient steps.  Returns a result dict suitable for JSON.
    """
    rng = random.Random(seed)
    # Seed numpy too so weight init is deterministic (same checkpoint, so
    # this only affects any stochastic elements inside learn())
    np.random.seed(seed % 2**31)

    agent = TopoAgent(config={"learn_steps": STEPS_PER_TEXT, "learn_lr": LR})
    snapshots = []
    loss_log = []

    for text in texts:
        cx = encounter_complex(text)
        # Patched training loop: snapshot inside gradient steps
        clean = agent._clean(text)
        if len(clean) < 2:
            continue
        tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
        from vybn import RV, N_LAYER, BLOCK_SIZE, _forward, _softmax
        n = min(BLOCK_SIZE, len(tokens) - 1)
        for step in range(STEPS_PER_TEXT):
            keys = [[] for _ in range(N_LAYER)]
            vals = [[] for _ in range(N_LAYER)]
            loss = RV(0.0)
            for t in range(n):
                logits, keys, vals = _forward(tokens[t], t, keys, vals, agent.sd)
                probs = _softmax(logits)
                loss = loss + (probs[tokens[t + 1]].log()) * (-1.0 / n)
            for p in agent.params:
                p.grad = 0.0
            loss.backward()
            agent._step += 1
            for j, p in enumerate(agent.params):
                g = p.grad
                agent._m[j] = 0.85 * agent._m[j] + 0.15 * g
                agent._v[j] = 0.99 * agent._v[j] + 0.01 * g ** 2
                mh = agent._m[j] / (1 - 0.85 ** agent._step)
                vh = agent._v[j] / (1 - 0.99 ** agent._step)
                p.data -= LR * mh / (vh ** 0.5 + 1e-8)
            loss_log.append(round(loss.data, 6))
            if step % SNAP_EVERY == 0:
                snapshots.append(weight_vector(agent))

    if not snapshots:
        snapshots.append(weight_vector(agent))

    points = np.array(snapshots)
    topo = compute_topology(points)
    final_loss, _ = agent.predict(" ".join(texts[:1]))

    return {
        "condition": condition,
        "label": label,
        "run_idx": run_idx,
        "seed": seed,
        "texts": texts,
        "n_snapshots": len(snapshots),
        "loss_trajectory": loss_log,
        "final_loss": round(final_loss, 6),
        "topology": topo,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Corpus helpers ────────────────────────────────────────────────────────

def load_corpus(min_words: int = 40) -> List[str]:
    """Load real prose corpus; at least min_words per passage."""
    passages = []
    if CORPUS_PATH.exists():
        lines = [l.strip() for l in CORPUS_PATH.read_text().split("\n") if l.strip()]
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
        # Last resort: short synthetic stand-ins
        passages = [
            "the creature breathes and measures its own distance from itself in the quiet between encounters",
            "curvature is born from incompleteness not from complexity alone what survives testing is true",
            "what survives testing is more honest than what merely sounds beautiful in the arrangement of words",
            "prediction loss going down means the model has memorised a pattern call it what it is",
            "the topology of weight space is a fingerprint of what was learned not merely how much was learned",
            "persistent homology integrates structure over a filtration giving us a summary independent of threshold",
            "attention heads rotate through conceptual space each head aligned to a different bivector plane",
        ]
    return passages


def cluster_texts(texts: List[str], n_clusters: int = 3) -> List[List[str]]:
    """
    Simple greedy embedding-based clustering.
    Returns list of clusters (each a list of text strings).
    """
    if len(texts) < n_clusters:
        return [texts]
    vecs = embed(texts)
    # Greedy farthest-point seeding (k-means++ style), then assign
    seeds = [0]
    for _ in range(n_clusters - 1):
        dists = np.array([
            min(float(np.linalg.norm(vecs[i] - vecs[s])) for s in seeds)
            for i in range(len(texts))
        ])
        dists[seeds] = 0.0
        seeds.append(int(np.argmax(dists)))
    clusters = [[] for _ in range(n_clusters)]
    for i, txt in enumerate(texts):
        nearest = min(range(n_clusters), key=lambda c: float(np.linalg.norm(vecs[i] - vecs[seeds[c]])))
        clusters[nearest].append(txt)
    return [c for c in clusters if c]


def farthest_point_sample(texts: List[str], k: int) -> List[str]:
    """Select k texts that maximise pairwise embedding distance."""
    if len(texts) <= k:
        return texts
    vecs = embed(texts)
    selected = [0]
    while len(selected) < k:
        dists = np.array([
            min(float(np.linalg.norm(vecs[i] - vecs[s])) for s in selected)
            for i in range(len(texts))
        ])
        dists[selected] = -1.0
        selected.append(int(np.argmax(dists)))
    return [texts[i] for i in selected]


def synthetic_text(length_chars: int, seed: int) -> str:
    """
    Random printable-ASCII sequence of given length — the sharpest negative
    control: same length/structure as real text, zero semantic content.
    """
    rng = random.Random(seed)
    chars = "abcdefghijklmnopqrstuvwxyz "
    return "".join(rng.choice(chars) for _ in range(length_chars))


# ── Main experiment ───────────────────────────────────────────────────────

def run_experiment(
    n_random: int = 20,
    n_coherent: int = 10,
    n_diverse: int = 10,
    n_order: int = 10,
    n_synthetic: int = 10,
    k: int = K,
    seed_base: int = 42,
    verbose: bool = True,
) -> List[dict]:
    corpus = load_corpus()
    if verbose:
        print(f"Corpus: {len(corpus)} passages  K={k}  SNAP_EVERY={SNAP_EVERY}  STEPS={STEPS_PER_TEXT}")
        print(f"Ripser: {'available' if RIPSER_AVAILABLE else 'not found — using built-in fallback'}")
        print()

    all_results = []

    def _save(result: dict):
        fname = RESULTS_DIR / f"{result['condition']}_{result['run_idx']:03d}.json"
        fname.write_text(json.dumps(result, indent=2, default=str))
        all_results.append(result)
        if verbose:
            topo = result["topology"]
            print(
                f"  [{result['condition']:10s} run {result['run_idx']:02d}] "
                f"snapshots={result['n_snapshots']} "
                f"b1={topo['betti_1']} "
                f"tp_h1={topo['total_persistence_h1']:.4f} "
                f"ent={topo['persistence_entropy_h1']:.4f} "
                f"loss={result['final_loss']:.4f}"
            )

    # ── Condition 1: Random baseline ──
    if verbose:
        print(f"=== Condition 1: random baseline (N={n_random}) ===")
    rng = random.Random(seed_base)
    for i in range(n_random):
        texts = rng.sample(corpus, min(k, len(corpus)))
        result = run_condition(
            texts=texts,
            seed=seed_base + i,
            label="random",
            condition="random",
            run_idx=i,
        )
        _save(result)

    # ── Condition 2: Coherent sets ──
    if verbose:
        print(f"\n=== Condition 2: coherent sets (N={n_coherent}) ===")
    clusters = cluster_texts(corpus, n_clusters=max(3, n_coherent // 3))
    rng2 = random.Random(seed_base + 100)
    for i in range(n_coherent):
        cluster = clusters[i % len(clusters)]
        texts = rng2.sample(cluster, min(k, len(cluster)))
        if len(texts) < k:
            # Pad from same cluster with repetition
            while len(texts) < k:
                texts.append(rng2.choice(cluster))
        result = run_condition(
            texts=texts,
            seed=seed_base + 100 + i,
            label="coherent",
            condition="coherent",
            run_idx=i,
        )
        _save(result)

    # ── Condition 3: Diverse sets ──
    if verbose:
        print(f"\n=== Condition 3: diverse sets (N={n_diverse}) ===")
    diverse_base = farthest_point_sample(corpus, min(k * 3, len(corpus)))
    rng3 = random.Random(seed_base + 200)
    for i in range(n_diverse):
        # Each run: subsample k from the diverse pool, shuffle order
        texts = rng3.sample(diverse_base, min(k, len(diverse_base)))
        result = run_condition(
            texts=texts,
            seed=seed_base + 200 + i,
            label="diverse",
            condition="diverse",
            run_idx=i,
        )
        _save(result)

    # ── Condition 4: Order permutations ──
    if verbose:
        print(f"\n=== Condition 4: order permutations (N={n_order}) ===")
    fixed_texts = corpus[:k] if len(corpus) >= k else corpus
    import itertools
    perms = list(itertools.permutations(range(len(fixed_texts))))
    rng4 = random.Random(seed_base + 300)
    selected_perms = rng4.sample(perms, min(n_order, len(perms)))
    for i, perm in enumerate(selected_perms):
        texts = [fixed_texts[j] for j in perm]
        result = run_condition(
            texts=texts,
            seed=seed_base + 300 + i,
            label="order",
            condition="order",
            run_idx=i,
        )
        _save(result)

    # ── Condition 5: Synthetic (random bytes) ──
    if verbose:
        print(f"\n=== Condition 5: synthetic random sequences (N={n_synthetic}) ===")
    avg_len = int(np.mean([len(t) for t in corpus[:20]])) if corpus else 200
    rng5 = random.Random(seed_base + 400)
    for i in range(n_synthetic):
        texts = [
            synthetic_text(avg_len, seed=seed_base + 400 + i * k + j)
            for j in range(k)
        ]
        result = run_condition(
            texts=texts,
            seed=seed_base + 400 + i,
            label="synthetic",
            condition="synthetic",
            run_idx=i,
        )
        _save(result)

    return all_results


# ── Analysis ──────────────────────────────────────────────────────────────

def load_results() -> List[dict]:
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            pass
    return results


def summarise(results: List[dict]) -> dict:
    """Compute per-condition summary statistics and the core yes/no answer."""
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    cond_stats = {}
    for cond, runs in sorted(by_cond.items()):
        tp_h1s = [r["topology"]["total_persistence_h1"] for r in runs]
        b1s = [r["topology"]["betti_1"] for r in runs]
        ents = [r["topology"]["persistence_entropy_h1"] for r in runs]
        n = len(runs)
        cond_stats[cond] = {
            "n": n,
            "tp_h1_mean": float(np.mean(tp_h1s)),
            "tp_h1_std": float(np.std(tp_h1s)),
            "b1_mean": float(np.mean(b1s)),
            "b1_std": float(np.std(b1s)),
            "ent_mean": float(np.mean(ents)),
            "tp_h1_values": tp_h1s,
        }
        print(
            f"  {cond:12s}: n={n:2d}  "
            f"tp_h1 = {np.mean(tp_h1s):.4f} ± {np.std(tp_h1s):.4f}  "
            f"b1 = {np.mean(b1s):.2f} ± {np.std(b1s):.2f}  "
            f"entropy = {np.mean(ents):.4f}"
        )

    # ── Kruskal-Wallis across real-text conditions (1-3) ──
    print("\n--- Statistical tests ---")
    real_conds = [c for c in ["random", "coherent", "diverse"] if c in cond_stats]
    if len(real_conds) >= 2:
        try:
            from scipy.stats import kruskal
            groups = [cond_stats[c]["tp_h1_values"] for c in real_conds]
            if all(len(g) > 1 for g in groups):
                stat, p = kruskal(*groups)
                print(f"  Kruskal-Wallis (random vs coherent vs diverse): H={stat:.4f}  p={p:.4f}")
                if p < 0.05:
                    print("  *** Text selection significantly affects weight-space topology (p<0.05) ***")
                else:
                    print("  No significant difference between conditions (p>=0.05)")
            else:
                print("  (Not enough data for Kruskal-Wallis)")
        except ImportError:
            print("  (scipy not available — skipping Kruskal-Wallis)")
            # Manual variance comparison
            all_vals = [v for c in real_conds for v in cond_stats[c]["tp_h1_values"]]
            grand_mean = np.mean(all_vals)
            between_var = np.mean([
                (np.mean(cond_stats[c]["tp_h1_values"]) - grand_mean) ** 2
                for c in real_conds
            ])
            within_var = np.mean([
                np.var(cond_stats[c]["tp_h1_values"]) for c in real_conds
            ])
            ratio = between_var / (within_var + 1e-12)
            print(f"  Between/within variance ratio: {ratio:.4f}  (>1 suggests condition effect)")

    # ── Order effect (condition 4) ──
    if "order" in cond_stats:
        order_vals = cond_stats["order"]["tp_h1_values"]
        print(f"  Order-permutation variance: {float(np.var(order_vals)):.6f}  "
              f"(> 0 means reading order affects topology)")

    # ── Synthetic vs real (condition 5) ──
    if "synthetic" in cond_stats and "random" in cond_stats:
        syn = np.mean(cond_stats["synthetic"]["tp_h1_values"])
        rand = np.mean(cond_stats["random"]["tp_h1_values"])
        diff = rand - syn
        print(f"  Real minus synthetic tp_h1: {diff:+.4f}  "
              f"({'real text produces richer topology' if diff > 0.001 else 'no difference — counting artifact'}))")

    # ── Pairwise pseudo-Wasserstein ──
    print("\n--- Pairwise pseudo-Wasserstein distances between condition means ---")
    cond_list = sorted(by_cond.keys())
    for i, ca in enumerate(cond_list):
        for cb in cond_list[i+1:]:
            # Average diagram from each condition
            diags_a = [r["topology"]["diagram_h1"] for r in by_cond[ca]]
            diags_b = [r["topology"]["diagram_h1"] for r in by_cond[cb]]
            # Flatten into one representative diagram per condition (union)
            flat_a = [pair for d in diags_a for pair in d]
            flat_b = [pair for d in diags_b for pair in d]
            w = pseudo_wasserstein(flat_a, flat_b)
            print(f"  {ca:12s} vs {cb:12s}: W≈{w:.6f}")

    # ── Verdict ──
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if real_conds and len(real_conds) >= 2:
        try:
            from scipy.stats import kruskal
            groups = [cond_stats[c]["tp_h1_values"] for c in real_conds]
            if all(len(g) > 1 for g in groups):
                _, p = kruskal(*groups)
                if p < 0.05:
                    print("YES — text selection measurably affects weight-space topology.")
                    print("The topology signal carries content information, not just sample count.")
                else:
                    print("NO (at p<0.05) — topology does not significantly vary with text selection.")
                    print("The weight-space topology fitness component may be a counting artifact.")
                    print("Recommendation: redesign or remove the nw term from fitness().")
        except ImportError:
            print("(scipy unavailable — inspect variance ratios above for manual verdict)")
    else:
        print("Insufficient conditions for verdict — run full experiment.")

    return cond_stats


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weight-space topology experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 3 runs per condition")
    parser.add_argument("--analyze", action="store_true",
                        help="Load saved results and analyse (no new runs)")
    parser.add_argument("--k", type=int, default=K,
                        help=f"Texts per run (default {K})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed base (default 42)")
    args = parser.parse_args()

    if args.analyze:
        results = load_results()
        if not results:
            print("No saved results found. Run the experiment first.")
            return
        summarise(results)
        return

    if args.quick:
        n_r, n_c, n_d, n_o, n_s = 3, 3, 3, 3, 3
        print("Quick mode: 3 runs per condition")
    else:
        n_r, n_c, n_d, n_o, n_s = 20, 10, 10, 10, 10

    t0 = time.time()
    results = run_experiment(
        n_random=n_r, n_coherent=n_c, n_diverse=n_d,
        n_order=n_o, n_synthetic=n_s,
        k=args.k, seed_base=args.seed,
    )
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s  ({len(results)} runs)")
    summarise(results)

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        # Strip full diagram lists from summary to keep it readable
        slim = []
        for r in results:
            slim.append({k: v for k, v in r.items() if k != "topology"}
                       | {"topo_summary": {kk: vv for kk, vv in r["topology"].items()
                                           if kk != "diagram_h1"}})
        json.dump(slim, f, indent=2, default=str)
    print(f"\nFull results saved to: {RESULTS_DIR}/")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
