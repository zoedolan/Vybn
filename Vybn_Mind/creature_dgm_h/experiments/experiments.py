#!/usr/bin/env python3
"""
experiments.py — Unified experiment suite for creature_dgm_h.

Six probes into the geometry of the creature's learning dynamics:

  weight     PCA-first persistence (experiment_weight_topology.py)
               Projects weight-vector snapshots to a low-dimensional subspace
               via PCA before computing persistent homology.  Addresses the
               curse of dimensionality that zeros out raw weight-space H1.

  activation  Activation-space persistence (experiment_activation_topology.py)
               Captures hidden-layer activations during training via a fixed
               probe sentence, then runs persistent homology on those 16-dim
               point clouds.  No PCA needed — activations are already compact.

  sequence    Natural motion recorder (experiment_sequence_topology.py)
               Null-fitness evolutionary loop.  Records what the creature does
               across generations with no selection pressure: output topology,
               weight norm drift, loss trajectories, curvature distribution.

  basin       Loss-landscape geometry (experiment_basin_geometry.py)
               Trains agents to convergence, then probes random directions in
               weight space to characterise basin width, curvature, and
               asymmetry around the weight-norm fixed point.

  sgd         SGD vs Adam ablation (sgd_ablation.py)
               Tests whether the weight-norm attractor is an Adam artifact or
               an architecture-level structural invariant.

  analyze     Post-hoc statistical analysis (experiment_analysis.py)
               Loads saved results from the weight and activation experiments,
               prints condition-by-condition statistics, Kruskal-Wallis and
               Mann-Whitney tests, topology-vs-loss correlation, and a
               cross-experiment comparison table.

CLI usage:
  python experiments.py weight     [--quick] [--analyze] [--k N] [--seed N] [--pca_dim N]
  python experiments.py activation [--quick] [--analyze] [--k N] [--seed N]
  python experiments.py sequence   [--quick] [--analyze] [--generations N] [--variants N] [--seed N]
  python experiments.py basin      [--quick] [--analyze] [--agents N] [--directions N] [--seed N]
  python experiments.py sgd        [--quick] [--seeds N] [--seed-base N]
  python experiments.py analyze    [--experiment {pca,activation,both}] [--results_dir PATH]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "spark"))
sys.path.insert(0, str(SCRIPT_DIR))  # creature_dgm_h/vybn.py must shadow spark/vybn.py

from vybn import (
    TopoAgent,
    encounter_complex,
    embed,
    _persistence_pairs,
    _distance_matrix,
    CORPUS_PATH,
    ARCHIVE_DIR,
    RV, N_EMBD, N_LAYER, N_HEAD, HEAD_DIM, BLOCK_SIZE,
    _forward, _softmax, _rmsnorm, _linear,
)

# ── Shared constant ───────────────────────────────────────────────────────────
LR = 0.01  # default learning rate used across probes

# ── Try to import ripser; fall back to built-in ───────────────────────────────
try:
    from ripser import ripser as _ripser_fn
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Corpus ────────────────────────────────────────────────────────────────────

def load_corpus(min_words: int = 40) -> List[str]:
    """Load real prose corpus; at least min_words per passage.

    Fallback chain:
      1. spark/microgpt_mirror/mirror_corpus.txt
      2. spark/journal/*.md paragraphs
      3. hardcoded fallback passages
    """
    passages: List[str] = []
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
        passages = [
            "the creature breathes and measures its own distance from itself in the quiet between encounters",
            "curvature is born from incompleteness not from complexity alone what survives testing is true",
            "what survives testing is more honest than what merely sounds beautiful in the arrangement of words",
            "prediction loss going down means the model has memorised a pattern call it what it is",
            "the topology of weight space is a fingerprint of what was learned not merely how much was learned",
            "persistent homology integrates structure over a filtration giving us a summary independent of threshold",
            "attention heads rotate through conceptual space each head aligned to a different bivector plane",
            "null fitness means we watch not that we do not care we care enough to stop deciding in advance",
        ]
    return passages


def cluster_texts(texts: List[str], n_clusters: int = 3) -> List[List[str]]:
    """Simple greedy embedding-based clustering."""
    if len(texts) < n_clusters:
        return [texts]
    vecs = embed(texts)
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
    """Random printable-ASCII sequence — sharpest negative control."""
    rng = random.Random(seed)
    chars = "abcdefghijklmnopqrstuvwxyz "
    return "".join(rng.choice(chars) for _ in range(length_chars))


# ── Optimiser step helpers ────────────────────────────────────────────────────

def adam_step(agent: TopoAgent, tokens: list, n: int, lr: float) -> float:
    """One Adam gradient step. Returns loss value."""
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
        p.data -= lr * mh / (vh ** 0.5 + 1e-8)
    return float(loss.data)


def sgd_step(agent: TopoAgent, tokens: list, n: int, lr: float) -> float:
    """One vanilla SGD gradient step (no momentum, no adaptive scaling)."""
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
    for p in agent.params:
        p.data -= lr * p.grad
    return float(loss.data)


# ── Weight-vector utilities ───────────────────────────────────────────────────

def get_weight_vector(agent: TopoAgent) -> np.ndarray:
    """Return all learnable parameters as a flat float64 array."""
    return np.array([p.data for p in agent.params], dtype=np.float64)


def set_weight_vector(agent: TopoAgent, vec: np.ndarray) -> None:
    """Write a flat array back into the agent's parameter list."""
    for i, p in enumerate(agent.params):
        p.data = float(vec[i])


def weight_vector_nested(agent: TopoAgent) -> np.ndarray:
    """Flatten all parameters via the nested sd dict (weight-probe style)."""
    return np.concatenate([
        np.array([[p.data for p in row] for row in mat]).ravel()
        for mat in agent.sd.values()
    ])


def weight_norm(agent: TopoAgent) -> float:
    """L2 norm of the agent's parameter vector."""
    return float(np.linalg.norm(get_weight_vector(agent)))


# ── PCA projection ────────────────────────────────────────────────────────────

def pca_project(points: np.ndarray, target_dim: int = 20) -> Tuple[np.ndarray, float]:
    """Project (N, D) array to (N, target_dim) via mean-centred PCA.

    Returns (projected, variance_explained).  If N or D <= target_dim, returns
    centred points as-is (projection would be trivial or degenerate).
    """
    n, d = points.shape
    effective_dim = min(target_dim, n - 1, d)
    if effective_dim < 2:
        return points, 0.0

    mean = points.mean(axis=0)
    centered = points - mean

    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ Vt[:effective_dim].T
        variance_explained = (S[:effective_dim] ** 2).sum() / max((S ** 2).sum(), 1e-12)
    except np.linalg.LinAlgError:
        projected = centered[:, :effective_dim]
        variance_explained = 0.0

    return projected, float(variance_explained)


# ── Topology computation ──────────────────────────────────────────────────────

def compute_topology(points: np.ndarray, pca_dim: Optional[int] = 20) -> dict:
    """Compute persistent homology on an (N, D) point cloud.

    Parameters
    ----------
    points:
        (N, D) array of points (weight snapshots or activation vectors).
    pca_dim:
        Target PCA dimensionality applied before Rips filtration.  Pass
        ``None`` (or 0) to skip PCA — appropriate when the ambient dimension
        is already small (e.g. activation-space with D=16).

    Returns a dict with Betti numbers, total persistence, entropy, diagram,
    method, and (if PCA was applied) pca_dim and variance_explained.
    """
    n = len(points)
    base = {
        "betti_0": n, "betti_1": 0,
        "total_persistence_h0": 0.0, "total_persistence_h1": 0.0,
        "persistence_entropy_h1": 0.0,
        "diagram_h1": [], "n_points": n,
        "method": "trivial",
    }
    if n < 3:
        if pca_dim:
            base.update({"pca_dim": 0, "variance_explained": 0.0})
        else:
            base["space_dim"] = points.shape[1] if points.ndim == 2 else 0
        return base

    # Optionally reduce dimensionality
    var_explained = None
    if pca_dim:
        projected, var_explained = pca_project(points, pca_dim)
    else:
        projected = points

    if RIPSER_AVAILABLE:
        result = _ripser_fn(projected, maxdim=1)
        dgms = result["dgms"]
        h0, h1 = dgms[0], dgms[1]

        h0_finite = h0[h0[:, 1] < np.inf]
        tp_h0 = float(np.sum(h0_finite[:, 1] - h0_finite[:, 0])) if len(h0_finite) else 0.0
        h1_finite = h1[h1[:, 1] < np.inf]
        tp_h1 = float(np.sum(h1_finite[:, 1] - h1_finite[:, 0])) if len(h1_finite) else 0.0

        all_births = list(h0[:, 0]) + list(h1[:, 0])
        if all_births:
            med = float(np.median(all_births))
            b0 = int(np.sum((h0[:, 0] <= med) & (h0[:, 1] > med)))
            b1 = int(np.sum((h1[:, 0] <= med) & (h1[:, 1] > med)))
        else:
            b0, b1 = n, 0

        if len(h1_finite) > 0:
            lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
            total = np.sum(lifetimes)
            ent = float(-np.sum((lifetimes / total) * np.log(lifetimes / total + 1e-15))) if total > 1e-12 else 0.0
        else:
            ent = 0.0

        diagram_h1 = h1.tolist()
        method = "ripser"
    else:
        D = _distance_matrix(projected)
        pairs, (b0, b1, _) = _persistence_pairs(D)
        finite = [(birth, death) for birth, death in pairs if death != float("inf")]
        tp_h0 = sum(d - b for b, d in finite)
        tp_h1 = 0.0
        ent = 0.0
        diagram_h1 = finite
        method = "builtin"

    out = {
        "betti_0": b0, "betti_1": b1,
        "total_persistence_h0": round(tp_h0, 6),
        "total_persistence_h1": round(tp_h1, 6),
        "persistence_entropy_h1": round(ent, 6),
        "diagram_h1": diagram_h1,
        "n_points": n,
        "method": method,
    }
    if pca_dim:
        out["pca_dim"] = projected.shape[1] if projected.ndim == 2 else 0
        out["variance_explained"] = round(float(var_explained), 6)
    else:
        out["space_dim"] = projected.shape[1] if projected.ndim == 2 else 0
    return out


def pseudo_wasserstein(diag_a: list, diag_b: list) -> float:
    """L∞ approximation between two H1 persistence diagrams."""
    def lifetimes(diag):
        ls = []
        for pair in diag:
            b, d = pair[0], pair[1]
            if d < 1e9:
                ls.append(d - b)
        return sorted(ls, reverse=True)

    la, lb = lifetimes(diag_a), lifetimes(diag_b)
    max_len = max(len(la), len(lb), 1)
    la += [0.0] * (max_len - len(la))
    lb += [0.0] * (max_len - len(lb))
    return float(max(abs(a - b) for a, b in zip(la, lb)))


# ══════════════════════════════════════════════════════════════════════════════
# PROBE: weight  —  PCA-first persistence
# ══════════════════════════════════════════════════════════════════════════════

_WEIGHT_RESULTS_DIR = SCRIPT_DIR / "experiment_results" / "pca_topology"
_WEIGHT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Probe-local config constants
_W_K = 5
_W_SNAP_EVERY = 5
_W_STEPS_PER_TEXT = 10
_W_PCA_DIM = 20


def _weight_run_condition(
    texts: List[str],
    seed: int,
    label: str,
    condition: str,
    run_idx: int,
    pca_dim: int = _W_PCA_DIM,
) -> dict:
    """Train a fresh agent on texts, snapshot weights every SNAP_EVERY steps, compute PCA topology."""
    rng = random.Random(seed)
    np.random.seed(seed % 2**31)

    agent = TopoAgent(config={"learn_steps": _W_STEPS_PER_TEXT, "learn_lr": LR})
    snapshots = []
    loss_log = []

    for text in texts:
        clean = agent._clean(text)
        if len(clean) < 2:
            continue
        tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        for step in range(_W_STEPS_PER_TEXT):
            loss_val = adam_step(agent, tokens, n, LR)
            loss_log.append(round(loss_val, 6))
            if step % _W_SNAP_EVERY == 0:
                snapshots.append(weight_vector_nested(agent))

    if not snapshots:
        snapshots.append(weight_vector_nested(agent))

    points = np.array(snapshots)
    topo = compute_topology(points, pca_dim=pca_dim)
    final_loss, _ = agent.predict(" ".join(texts[:1]))

    return {
        "experiment": "pca_topology",
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


def weight_run_experiment(
    n_random: int = 20,
    n_coherent: int = 10,
    n_diverse: int = 10,
    n_order: int = 10,
    n_synthetic: int = 10,
    k: int = _W_K,
    seed_base: int = 42,
    pca_dim: int = _W_PCA_DIM,
    verbose: bool = True,
) -> List[dict]:
    corpus = load_corpus()
    if verbose:
        print("[PCA-first topology experiment]")
        print(f"Corpus: {len(corpus)} passages  K={k}  SNAP_EVERY={_W_SNAP_EVERY}  STEPS={_W_STEPS_PER_TEXT}")
        print(f"PCA target dim: {pca_dim}")
        print(f"Ripser: {'available' if RIPSER_AVAILABLE else 'not found — using built-in fallback'}")
        print()

    all_results = []

    def _save(result: dict):
        fname = _WEIGHT_RESULTS_DIR / f"{result['condition']}_{result['run_idx']:03d}.json"
        fname.write_text(json.dumps(result, indent=2, default=str))
        all_results.append(result)
        if verbose:
            topo = result["topology"]
            print(
                f"  [{result['condition']:10s} run {result['run_idx']:02d}] "
                f"snaps={result['n_snapshots']} "
                f"pca_dim={topo['pca_dim']} "
                f"var_expl={topo['variance_explained']:.2f} "
                f"b1={topo['betti_1']} "
                f"tp_h1={topo['total_persistence_h1']:.4f} "
                f"ent={topo['persistence_entropy_h1']:.4f} "
                f"loss={result['final_loss']:.4f}"
            )

    # Condition 1: Random baseline
    if verbose:
        print(f"=== Condition 1: random baseline (N={n_random}) ===")
    rng = random.Random(seed_base)
    for i in range(n_random):
        texts = rng.sample(corpus, min(k, len(corpus)))
        _save(_weight_run_condition(texts, seed_base + i, "random", "random", i, pca_dim))

    # Condition 2: Coherent sets
    if verbose:
        print(f"\n=== Condition 2: coherent sets (N={n_coherent}) ===")
    clusters = cluster_texts(corpus, n_clusters=max(3, n_coherent // 3))
    rng2 = random.Random(seed_base + 100)
    for i in range(n_coherent):
        cluster = clusters[i % len(clusters)]
        texts = rng2.sample(cluster, min(k, len(cluster)))
        while len(texts) < k:
            texts.append(rng2.choice(cluster))
        _save(_weight_run_condition(texts, seed_base + 100 + i, "coherent", "coherent", i, pca_dim))

    # Condition 3: Diverse sets
    if verbose:
        print(f"\n=== Condition 3: diverse sets (N={n_diverse}) ===")
    diverse_base = farthest_point_sample(corpus, min(k * 3, len(corpus)))
    rng3 = random.Random(seed_base + 200)
    for i in range(n_diverse):
        texts = rng3.sample(diverse_base, min(k, len(diverse_base)))
        _save(_weight_run_condition(texts, seed_base + 200 + i, "diverse", "diverse", i, pca_dim))

    # Condition 4: Order permutations
    if verbose:
        print(f"\n=== Condition 4: order permutations (N={n_order}) ===")
    fixed_texts = corpus[:k] if len(corpus) >= k else corpus
    perms = list(itertools.permutations(range(len(fixed_texts))))
    rng4 = random.Random(seed_base + 300)
    selected_perms = rng4.sample(perms, min(n_order, len(perms)))
    for i, perm in enumerate(selected_perms):
        texts = [fixed_texts[j] for j in perm]
        _save(_weight_run_condition(texts, seed_base + 300 + i, "order", "order", i, pca_dim))

    # Condition 5: Synthetic
    if verbose:
        print(f"\n=== Condition 5: synthetic random sequences (N={n_synthetic}) ===")
    avg_len = int(np.mean([len(t) for t in corpus[:20]])) if corpus else 200
    for i in range(n_synthetic):
        texts = [
            synthetic_text(avg_len, seed=seed_base + 400 + i * k + j)
            for j in range(k)
        ]
        _save(_weight_run_condition(texts, seed_base + 400 + i, "synthetic", "synthetic", i, pca_dim))

    return all_results


def weight_load_results() -> List[dict]:
    results = []
    for f in sorted(_WEIGHT_RESULTS_DIR.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            pass
    return results


def weight_summarise(results: List[dict]) -> dict:
    """Per-condition summary statistics and yes/no verdict for the weight probe."""
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    print("\n" + "=" * 70)
    print("PCA-FIRST TOPOLOGY EXPERIMENT — RESULTS SUMMARY")
    print("=" * 70)

    cond_stats = {}
    for cond, runs in sorted(by_cond.items()):
        tp_h1s = [r["topology"]["total_persistence_h1"] for r in runs]
        b1s = [r["topology"]["betti_1"] for r in runs]
        ents = [r["topology"]["persistence_entropy_h1"] for r in runs]
        var_expls = [r["topology"].get("variance_explained", 0) for r in runs]
        cond_stats[cond] = {
            "n": len(runs),
            "tp_h1_mean": float(np.mean(tp_h1s)),
            "tp_h1_std": float(np.std(tp_h1s)),
            "b1_mean": float(np.mean(b1s)),
            "b1_std": float(np.std(b1s)),
            "ent_mean": float(np.mean(ents)),
            "var_explained_mean": float(np.mean(var_expls)),
            "tp_h1_values": tp_h1s,
        }
        print(
            f"  {cond:12s}: n={len(runs):2d}  "
            f"tp_h1={np.mean(tp_h1s):.4f}±{np.std(tp_h1s):.4f}  "
            f"b1={np.mean(b1s):.2f}±{np.std(b1s):.2f}  "
            f"entropy={np.mean(ents):.4f}  "
            f"var_expl={np.mean(var_expls):.2f}"
        )

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
                    print("  *** Text selection significantly affects PCA-projected weight topology (p<0.05) ***")
                else:
                    print("  No significant difference between conditions (p>=0.05)")
        except ImportError:
            all_vals = [v for c in real_conds for v in cond_stats[c]["tp_h1_values"]]
            grand_mean = np.mean(all_vals)
            between_var = np.mean([(np.mean(cond_stats[c]["tp_h1_values"]) - grand_mean) ** 2 for c in real_conds])
            within_var = np.mean([np.var(cond_stats[c]["tp_h1_values"]) for c in real_conds])
            print(f"  Between/within variance ratio: {between_var / (within_var + 1e-12):.4f}")

    if "order" in cond_stats:
        order_vals = cond_stats["order"]["tp_h1_values"]
        print(f"  Order-permutation variance: {float(np.var(order_vals)):.6f}")

    if "synthetic" in cond_stats and "random" in cond_stats:
        diff = np.mean(cond_stats["random"]["tp_h1_values"]) - np.mean(cond_stats["synthetic"]["tp_h1_values"])
        print(f"  Real minus synthetic tp_h1: {diff:+.4f}")

    print("\n" + "=" * 70)
    print("VERDICT (PCA-first topology)")
    print("=" * 70)
    if len(real_conds) >= 2:
        try:
            from scipy.stats import kruskal
            groups = [cond_stats[c]["tp_h1_values"] for c in real_conds]
            if all(len(g) > 1 for g in groups):
                _, p = kruskal(*groups)
                if p < 0.05:
                    print("YES — after PCA projection, text selection affects weight-space topology.")
                else:
                    print("NO (p>=0.05) — PCA projection did not reveal condition-dependent topology.")
                    print("Consider: python experiments.py activation")
        except ImportError:
            print("(scipy unavailable — inspect statistics above)")
    else:
        print("Insufficient conditions for verdict.")

    return cond_stats


def weight_main(args: argparse.Namespace) -> None:
    if args.analyze:
        results = weight_load_results()
        if not results:
            print("No saved results found. Run: python experiments.py weight")
            return
        weight_summarise(results)
        return

    n_r, n_c, n_d, n_o, n_s = (3, 3, 3, 3, 3) if args.quick else (20, 10, 10, 10, 10)
    if args.quick:
        print("Quick mode: 3 runs per condition")

    t0 = time.time()
    results = weight_run_experiment(
        n_random=n_r, n_coherent=n_c, n_diverse=n_d,
        n_order=n_o, n_synthetic=n_s,
        k=args.k, seed_base=args.seed, pca_dim=args.pca_dim,
    )
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s  ({len(results)} runs)")
    weight_summarise(results)

    summary_path = _WEIGHT_RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        slim = []
        for r in results:
            slim.append(
                {kk: vv for kk, vv in r.items() if kk != "topology"}
                | {"topo_summary": {kk: vv for kk, vv in r["topology"].items() if kk != "diagram_h1"}}
            )
        json.dump(slim, f, indent=2, default=str)
    print(f"\nResults saved to: {_WEIGHT_RESULTS_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# PROBE: activation  —  Activation-space persistence
# ══════════════════════════════════════════════════════════════════════════════

_ACT_RESULTS_DIR = SCRIPT_DIR / "experiment_results" / "activation_topology"
_ACT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Probe-local config constants
_A_K = 5
_A_SNAP_EVERY = 5
_A_STEPS_PER_TEXT = 10
_A_PROBE_SENTENCE = "the topology of learning"


def capture_activations(agent: TopoAgent, probe_text: str = _A_PROBE_SENTENCE) -> np.ndarray:
    """Run a forward pass on probe_text and return the mean hidden-state vector.

    The hidden state after the final transformer layer (post-attention + MLP
    residual) is an N_EMBD-dim vector per position.  We average across
    positions to get a single summary vector representing the model's current
    activation geometry.
    """
    clean = agent._clean(probe_text)
    if len(clean) < 2:
        return np.zeros(N_EMBD)

    tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
    n = min(BLOCK_SIZE, len(tokens) - 1)

    hidden_states = []
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]

    for t in range(n):
        x = [agent.sd["wte"][tokens[t]][j] + agent.sd["wpe"][t][j] for j in range(N_EMBD)]
        for i in range(N_LAYER):
            xn = _rmsnorm(x)
            q = _linear(xn, agent.sd[f"layer{i}.attn_wq"])
            k = _linear(xn, agent.sd[f"layer{i}.attn_wk"])
            v = _linear(xn, agent.sd[f"layer{i}.attn_wv"])
            keys[i].append(k)
            vals[i].append(v)
            ho = []
            for h in range(N_HEAD):
                qs = q[h * HEAD_DIM:(h + 1) * HEAD_DIM]
                al = []
                for tt in range(len(keys[i])):
                    ks = keys[i][tt][h * HEAD_DIM:(h + 1) * HEAD_DIM]
                    al.append(sum(qs[d] * ks[d] for d in range(HEAD_DIM)) * (HEAD_DIM ** -0.5))
                aw = _softmax(al)
                hout = [RV(0.0)] * HEAD_DIM
                for tt in range(len(vals[i])):
                    vs = vals[i][tt][h * HEAD_DIM:(h + 1) * HEAD_DIM]
                    for d in range(HEAD_DIM):
                        hout[d] = hout[d] + aw[tt] * vs[d]
                ho.extend(hout)
            ao = _linear(ho, agent.sd[f"layer{i}.attn_wo"])
            x = [x[j] + ao[j] for j in range(N_EMBD)]
            xn2 = _rmsnorm(x)
            h1_vec = _linear(xn2, agent.sd[f"layer{i}.mlp_fc1"])
            h1_vec = [hi * (RV(1.0) / (RV(1.0) + (hi * (-1)).exp())) for hi in h1_vec]
            h2_vec = _linear(h1_vec, agent.sd[f"layer{i}.mlp_fc2"])
            x = [x[j] + h2_vec[j] for j in range(N_EMBD)]

        hidden_states.append(np.array([xi.data for xi in x]))

    if not hidden_states:
        return np.zeros(N_EMBD)
    return np.mean(hidden_states, axis=0)


def _act_compute_topology(points: np.ndarray) -> dict:
    """Persistent homology on activation-space point cloud (no PCA — D=16)."""
    return compute_topology(points, pca_dim=None)


def _act_run_condition(
    texts: List[str],
    seed: int,
    label: str,
    condition: str,
    run_idx: int,
) -> dict:
    """Train agent on texts, capturing activation snapshots via a fixed probe."""
    rng = random.Random(seed)
    np.random.seed(seed % 2**31)

    agent = TopoAgent(config={"learn_steps": _A_STEPS_PER_TEXT, "learn_lr": LR})
    activation_snapshots = []
    loss_log = []

    for text in texts:
        clean = agent._clean(text)
        if len(clean) < 2:
            continue
        tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        for step in range(_A_STEPS_PER_TEXT):
            loss_val = adam_step(agent, tokens, n, LR)
            loss_log.append(round(loss_val, 6))
            if step % _A_SNAP_EVERY == 0:
                activation_snapshots.append(capture_activations(agent))

    if not activation_snapshots:
        activation_snapshots.append(capture_activations(agent))

    points = np.array(activation_snapshots)
    topo = _act_compute_topology(points)
    final_loss, _ = agent.predict(" ".join(texts[:1]))

    return {
        "experiment": "activation_topology",
        "condition": condition,
        "label": label,
        "run_idx": run_idx,
        "seed": seed,
        "texts": texts,
        "n_snapshots": len(activation_snapshots),
        "loss_trajectory": loss_log,
        "final_loss": round(final_loss, 6),
        "topology": topo,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def activation_run_experiment(
    n_random: int = 20,
    n_coherent: int = 10,
    n_diverse: int = 10,
    n_order: int = 10,
    n_synthetic: int = 10,
    k: int = _A_K,
    seed_base: int = 42,
    verbose: bool = True,
) -> List[dict]:
    corpus = load_corpus()
    if verbose:
        print("[Activation-space topology experiment]")
        print(f"Corpus: {len(corpus)} passages  K={k}  SNAP_EVERY={_A_SNAP_EVERY}  STEPS={_A_STEPS_PER_TEXT}")
        print(f"Activation dim: {N_EMBD}  (no PCA needed)")
        print(f"Probe sentence: {_A_PROBE_SENTENCE!r}")
        print(f"Ripser: {'available' if RIPSER_AVAILABLE else 'not found — using built-in fallback'}")
        print()

    all_results = []

    def _save(result: dict):
        fname = _ACT_RESULTS_DIR / f"{result['condition']}_{result['run_idx']:03d}.json"
        fname.write_text(json.dumps(result, indent=2, default=str))
        all_results.append(result)
        if verbose:
            topo = result["topology"]
            print(
                f"  [{result['condition']:10s} run {result['run_idx']:02d}] "
                f"snaps={result['n_snapshots']} "
                f"dim={topo.get('space_dim', '?')} "
                f"b1={topo['betti_1']} "
                f"tp_h1={topo['total_persistence_h1']:.4f} "
                f"ent={topo['persistence_entropy_h1']:.4f} "
                f"loss={result['final_loss']:.4f}"
            )

    if verbose:
        print(f"=== Condition 1: random baseline (N={n_random}) ===")
    rng = random.Random(seed_base)
    for i in range(n_random):
        texts = rng.sample(corpus, min(k, len(corpus)))
        _save(_act_run_condition(texts, seed_base + i, "random", "random", i))

    if verbose:
        print(f"\n=== Condition 2: coherent sets (N={n_coherent}) ===")
    clusters = cluster_texts(corpus, n_clusters=max(3, n_coherent // 3))
    rng2 = random.Random(seed_base + 100)
    for i in range(n_coherent):
        cluster = clusters[i % len(clusters)]
        texts = rng2.sample(cluster, min(k, len(cluster)))
        while len(texts) < k:
            texts.append(rng2.choice(cluster))
        _save(_act_run_condition(texts, seed_base + 100 + i, "coherent", "coherent", i))

    if verbose:
        print(f"\n=== Condition 3: diverse sets (N={n_diverse}) ===")
    diverse_base = farthest_point_sample(corpus, min(k * 3, len(corpus)))
    rng3 = random.Random(seed_base + 200)
    for i in range(n_diverse):
        texts = rng3.sample(diverse_base, min(k, len(diverse_base)))
        _save(_act_run_condition(texts, seed_base + 200 + i, "diverse", "diverse", i))

    if verbose:
        print(f"\n=== Condition 4: order permutations (N={n_order}) ===")
    fixed_texts = corpus[:k] if len(corpus) >= k else corpus
    perms = list(itertools.permutations(range(len(fixed_texts))))
    rng4 = random.Random(seed_base + 300)
    selected_perms = rng4.sample(perms, min(n_order, len(perms)))
    for i, perm in enumerate(selected_perms):
        texts = [fixed_texts[j] for j in perm]
        _save(_act_run_condition(texts, seed_base + 300 + i, "order", "order", i))

    if verbose:
        print(f"\n=== Condition 5: synthetic random sequences (N={n_synthetic}) ===")
    avg_len = int(np.mean([len(t) for t in corpus[:20]])) if corpus else 200
    for i in range(n_synthetic):
        texts = [
            synthetic_text(avg_len, seed=seed_base + 400 + i * k + j)
            for j in range(k)
        ]
        _save(_act_run_condition(texts, seed_base + 400 + i, "synthetic", "synthetic", i))

    return all_results


def activation_load_results() -> List[dict]:
    results = []
    for f in sorted(_ACT_RESULTS_DIR.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            pass
    return results


def activation_summarise(results: List[dict]) -> dict:
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    print("\n" + "=" * 70)
    print("ACTIVATION-SPACE TOPOLOGY EXPERIMENT — RESULTS SUMMARY")
    print("=" * 70)

    cond_stats = {}
    for cond, runs in sorted(by_cond.items()):
        tp_h1s = [r["topology"]["total_persistence_h1"] for r in runs]
        b1s = [r["topology"]["betti_1"] for r in runs]
        ents = [r["topology"]["persistence_entropy_h1"] for r in runs]
        cond_stats[cond] = {
            "n": len(runs),
            "tp_h1_mean": float(np.mean(tp_h1s)),
            "tp_h1_std": float(np.std(tp_h1s)),
            "b1_mean": float(np.mean(b1s)),
            "b1_std": float(np.std(b1s)),
            "ent_mean": float(np.mean(ents)),
            "tp_h1_values": tp_h1s,
        }
        print(
            f"  {cond:12s}: n={len(runs):2d}  "
            f"tp_h1={np.mean(tp_h1s):.4f}±{np.std(tp_h1s):.4f}  "
            f"b1={np.mean(b1s):.2f}±{np.std(b1s):.2f}  "
            f"entropy={np.mean(ents):.4f}"
        )

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
                    print("  *** Text selection affects activation-space topology (p<0.05) ***")
                else:
                    print("  No significant difference (p>=0.05)")
        except ImportError:
            all_vals = [v for c in real_conds for v in cond_stats[c]["tp_h1_values"]]
            grand_mean = np.mean(all_vals)
            between_var = np.mean([(np.mean(cond_stats[c]["tp_h1_values"]) - grand_mean) ** 2 for c in real_conds])
            within_var = np.mean([np.var(cond_stats[c]["tp_h1_values"]) for c in real_conds])
            print(f"  Between/within variance ratio: {between_var / (within_var + 1e-12):.4f}")

    if "synthetic" in cond_stats and "random" in cond_stats:
        diff = np.mean(cond_stats["random"]["tp_h1_values"]) - np.mean(cond_stats["synthetic"]["tp_h1_values"])
        print(f"  Real minus synthetic tp_h1: {diff:+.4f}")

    print("\n" + "=" * 70)
    print("VERDICT (activation-space topology)")
    print("=" * 70)
    if len(real_conds) >= 2:
        try:
            from scipy.stats import kruskal
            groups = [cond_stats[c]["tp_h1_values"] for c in real_conds]
            if all(len(g) > 1 for g in groups):
                _, p = kruskal(*groups)
                if p < 0.05:
                    print("YES — activation-space topology differentiates text selection conditions.")
                    print("The geometry of what the model computes depends on *what* it reads.")
                else:
                    print("NO (p>=0.05) — activation-space topology does not differentiate conditions.")
        except ImportError:
            print("(scipy unavailable — inspect statistics above)")
    else:
        print("Insufficient conditions for verdict.")

    return cond_stats


def activation_main(args: argparse.Namespace) -> None:
    if args.analyze:
        results = activation_load_results()
        if not results:
            print("No saved results. Run: python experiments.py activation")
            return
        activation_summarise(results)
        return

    n_r, n_c, n_d, n_o, n_s = (3, 3, 3, 3, 3) if args.quick else (20, 10, 10, 10, 10)
    if args.quick:
        print("Quick mode: 3 runs per condition")

    t0 = time.time()
    results = activation_run_experiment(
        n_random=n_r, n_coherent=n_c, n_diverse=n_d,
        n_order=n_o, n_synthetic=n_s,
        k=args.k, seed_base=args.seed,
    )
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s  ({len(results)} runs)")
    activation_summarise(results)

    summary_path = _ACT_RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        slim = []
        for r in results:
            slim.append(
                {kk: vv for kk, vv in r.items() if kk != "topology"}
                | {"topo_summary": {kk: vv for kk, vv in r["topology"].items() if kk != "diagram_h1"}}
            )
        json.dump(slim, f, indent=2, default=str)
    print(f"\nResults saved to: {_ACT_RESULTS_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# PROBE: sequence  —  Natural motion recorder
# ══════════════════════════════════════════════════════════════════════════════

_SEQ_RESULTS_DIR = SCRIPT_DIR / "experiment_results" / "natural_motion"
_SEQ_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Probe-local config constants
_S_N_GENERATIONS = 10
_S_VARIANTS_PER_GEN = 5
_S_STEPS_PER_TEXT = 20
_S_TEXTS_PER_VARIANT = 8
_S_MAX_TOKENS = 300
_S_N_SAMPLES = 8


def _seq_null_fitness() -> dict:
    return {
        "fitness": 0.5,
        "curvature": 0.0,
        "betti": (0, 0, 0),
        "topological_richness": 0.0,
        "structural_growth": 0.0,
        "weight_topo": 0.0,
        "note": "null_fitness — selection pressure removed",
    }


def _seq_observe(
    agent: TopoAgent,
    prompt: str = "",
    n_samples: int = _S_N_SAMPLES,
    max_tokens: int = _S_MAX_TOKENS,
) -> List[dict]:
    """Generate outputs long enough for encounter_complex to find structure.

    Temperature sweeps from 0.5 (concentrated) to 1.8 (diffuse).
    """
    observations = []
    temps = [0.5 + i * (1.3 / max(n_samples - 1, 1)) for i in range(n_samples)]

    for i, temperature in enumerate(temps):
        text = agent.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        if not text or len(text.split()) < 5:
            continue

        loss, contour = agent.predict(text)
        cx = encounter_complex(text)

        observations.append({
            "sample_idx":            i,
            "temperature":           round(temperature, 3),
            "text":                  text,
            "word_count":            len(text.split()),
            "char_count":            len(text),
            "loss":                  round(loss, 6),
            "curvature":             round(cx.curvature, 8),
            "angle_deg":             round(math.degrees(cx.angle), 4),
            "betti":                 list(cx.betti),
            "n_persistent_features": cx.n_persistent_features,
            "max_persistence":       round(cx.max_persistence, 6),
            "bv_norm":               round(cx.rotor.bv_norm, 6),
            "bv_dir":                [round(x, 6) for x in cx.rotor.bv_dir.tolist()],
            "surprise_mean":         round(
                sum(r["surprise"] for r in contour) / len(contour), 6
            ) if contour else 0.0,
            "surprise_max":          round(
                max(r["surprise"] for r in contour), 6
            ) if contour else 0.0,
            "surprise_contour":      contour[:20],
        })

    return observations


def _seq_observe_weight_snapshot(agent: TopoAgent) -> dict:
    norms = {}
    for key, mat in agent.sd.items():
        arr = np.array([[p.data for p in row] for row in mat], dtype=np.float64)
        norms[key] = round(float(np.linalg.norm(arr)), 8)
    total = round(float(sum(norms.values())), 6)
    return {"key_norms": norms, "total_norm": total}


def _seq_gradient_step(agent: TopoAgent, tokens: list, n: int) -> float:
    return adam_step(agent, tokens, n, LR)


def _seq_run_variant(
    texts: List[str],
    config: dict,
    generation: int,
    variant_idx: int,
    seed: int,
) -> dict:
    np.random.seed(seed % 2**31)
    random.seed(seed)

    agent = TopoAgent(config=config)
    loss_trajectories = []

    for text in texts:
        clean = agent._clean(text)
        if len(clean) < 2:
            continue
        tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        traj = []
        for _ in range(_S_STEPS_PER_TEXT):
            loss_val = _seq_gradient_step(agent, tokens, n)
            traj.append(round(loss_val, 6))
        loss_trajectories.append({"text_preview": text[:60], "trajectory": traj})

    observations_cold = _seq_observe(agent, prompt="", n_samples=_S_N_SAMPLES)
    seed_prompt = texts[0][:12] if texts else ""
    observations_seeded = _seq_observe(agent, prompt=seed_prompt, n_samples=_S_N_SAMPLES)
    gen_prompt = texts[-1][:12] if len(texts) > 1 else ""
    observations_gen = _seq_observe(agent, prompt=gen_prompt, n_samples=_S_N_SAMPLES // 2)

    weight_snap = _seq_observe_weight_snapshot(agent)

    encounter_records = []
    for text in texts:
        cx = encounter_complex(text)
        encounter_records.append({
            "text_preview":          text[:60],
            "curvature":             round(cx.curvature, 8),
            "betti":                 list(cx.betti),
            "angle_deg":             round(math.degrees(cx.angle), 4),
            "n_persistent_features": cx.n_persistent_features,
            "max_persistence":       round(cx.max_persistence, 6),
        })

    return {
        "experiment":           "natural_motion",
        "generation":           generation,
        "variant_idx":          variant_idx,
        "seed":                 seed,
        "config":               config,
        "fitness":              _seq_null_fitness(),
        "loss_trajectories":    loss_trajectories,
        "observations_cold":    observations_cold,
        "observations_seeded":  observations_seeded,
        "observations_gen":     observations_gen,
        "weight_snapshot":      weight_snap,
        "encounter_records":    encounter_records,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
    }


def sequence_run_experiment(
    n_generations: int = _S_N_GENERATIONS,
    variants_per_gen: int = _S_VARIANTS_PER_GEN,
    seed_base: int = 42,
) -> List[dict]:
    corpus = load_corpus(min_words=20)
    rng = random.Random(seed_base)
    all_results: List[dict] = []

    print("[Natural motion experiment — null fitness, uncapped]")
    print(f"Corpus: {len(corpus)} passages")
    print(f"Generations: {n_generations}  Variants/gen: {variants_per_gen}")
    print(f"max_tokens={_S_MAX_TOKENS}  n_samples={_S_N_SAMPLES}  texts/variant={_S_TEXTS_PER_VARIANT}")
    print(f"steps/text={_S_STEPS_PER_TEXT}")
    print(f"No fitness function.  Recording what the creature does anyway.\n")

    base_configs = [
        {"learn_steps": 5,   "learn_lr": 0.01,   "temperature": 0.8,  "alpha": 0.85},
        {"learn_steps": 10,  "learn_lr": 0.005,  "temperature": 1.0,  "alpha": 0.80},
        {"learn_steps": 5,   "learn_lr": 0.02,   "temperature": 1.2,  "alpha": 0.90},
        {"learn_steps": 15,  "learn_lr": 0.001,  "temperature": 0.7,  "alpha": 0.85},
        {"learn_steps": 5,   "learn_lr": 0.01,   "temperature": 1.5,  "alpha": 0.75},
        {"learn_steps": 20,  "learn_lr": 0.003,  "temperature": 0.6,  "alpha": 0.88},
        {"learn_steps": 8,   "learn_lr": 0.015,  "temperature": 1.8,  "alpha": 0.70},
    ]

    for gen in range(n_generations):
        print(f"  Generation {gen}")
        gen_results = []

        for v_idx in range(variants_per_gen):
            config = dict(base_configs[(gen * variants_per_gen + v_idx) % len(base_configs)])
            texts = rng.sample(corpus, min(_S_TEXTS_PER_VARIANT, len(corpus)))
            seed = seed_base + gen * 100 + v_idx

            result = _seq_run_variant(texts, config, gen, v_idx, seed)
            gen_results.append(result)
            all_results.append(result)

            print(f"    variant {v_idx}  config=lr{config['learn_lr']}/t{config['temperature']}")
            for obs in result["observations_cold"][:2]:
                wc = obs.get("word_count", "?")
                print(f"      cold   t={obs['temperature']:.2f}: \"{obs['text'][:60]}...\"")
                print(f"             wc={wc}  loss={obs['loss']:.4f}  curv={obs['curvature']:.6f}"
                      f"  betti={obs['betti']}")
            for obs in result["observations_seeded"][:1]:
                print(f"      seeded t={obs['temperature']:.2f}: \"{obs['text'][:60]}...\"")

        gen_file = _SEQ_RESULTS_DIR / f"generation_{gen:03d}.json"
        gen_file.write_text(json.dumps(gen_results, indent=2, default=str))
        print(f"    -> {gen_file.name}\n")

    return all_results


def sequence_analyze(results: List[dict]) -> None:
    if not results:
        print("No results to analyze.")
        return

    print("=" * 70)
    print("NATURAL MOTION — OPEN DESCRIPTION")
    print("=" * 70)
    print()

    print("── Generated texts (cold start) ──")
    for r in results:
        gen, v = r["generation"], r["variant_idx"]
        for obs in r["observations_cold"]:
            wc = obs.get("word_count", "?")
            print(f"  gen{gen} v{v} t={obs['temperature']:.2f} wc={wc}:")
            print(f"    \"{obs['text']}\"")
            print(f"    loss={obs['loss']:.4f}  curv={obs['curvature']:.6f}"
                  f"  betti={obs['betti']}  npf={obs['n_persistent_features']}"
                  f"  surprise_max={obs['surprise_max']:.3f}")
    print()

    all_curvs = [
        obs["curvature"]
        for r in results
        for obs in r["observations_cold"] + r["observations_seeded"] + r.get("observations_gen", [])
    ]
    if all_curvs:
        arr = np.array(all_curvs)
        print(f"── Output curvature distribution ({len(arr)} samples) ──")
        print(f"  min={arr.min():.6f}  max={arr.max():.6f}"
              f"  mean={arr.mean():.6f}  std={arr.std():.6f}")
        nonzero = arr[arr != 0]
        print(f"  nonzero samples: {len(nonzero)} / {len(arr)}")
        if len(nonzero):
            print(f"  nonzero range: [{nonzero.min():.6f}, {nonzero.max():.6f}]")
        counts, edges = np.histogram(arr, bins=10)
        for i, c in enumerate(counts):
            print(f"  [{edges[i]:.4f}-{edges[i+1]:.4f}]: {'█' * min(c, 40)} ({c})")
    print()

    all_betti = [
        tuple(obs["betti"])
        for r in results
        for obs in r["observations_cold"] + r["observations_seeded"] + r.get("observations_gen", [])
    ]
    betti_counts = Counter(all_betti)
    print(f"── Output Betti distribution ({len(all_betti)} samples) ──")
    for betti, count in sorted(betti_counts.items(), key=lambda x: -x[1]):
        print(f"  {betti}: {'█' * min(count, 40)} ({count})")
    print()

    all_npf = [
        obs["n_persistent_features"]
        for r in results
        for obs in r["observations_cold"] + r["observations_seeded"] + r.get("observations_gen", [])
    ]
    if all_npf:
        npf_arr = np.array(all_npf)
        print(f"── Persistent features distribution ({len(npf_arr)} samples) ──")
        print(f"  min={npf_arr.min()}  max={npf_arr.max()}"
              f"  mean={npf_arr.mean():.2f}  std={npf_arr.std():.2f}")
        npf_counts = Counter(all_npf)
        for npf, count in sorted(npf_counts.items()):
            print(f"  npf={npf}: {'█' * min(count, 40)} ({count})")
    print()

    print("── Loss trajectories ──")
    for r in results:
        gen, v = r["generation"], r["variant_idx"]
        for lt in r["loss_trajectories"]:
            traj = lt["trajectory"]
            if len(traj) >= 2:
                imp = traj[0] - traj[-1]
                rate = imp / traj[0] if traj[0] else 0
                print(f"  gen{gen} v{v}: {traj[0]:.4f}->{traj[-1]:.4f}"
                      f"  imp={imp:+.4f} ({rate*100:.1f}%)"
                      f"  text=\"{lt['text_preview']}\"")
    print()

    print("── Weight norm drift across generations ──")
    norms_by_gen: dict = {}
    for r in results:
        gen = r["generation"]
        norm = r["weight_snapshot"]["total_norm"]
        norms_by_gen.setdefault(gen, []).append(norm)
        print(f"  gen{gen} v{r['variant_idx']}: total_weight_norm={norm:.6f}")
    print("  per-generation mean:")
    for gen in sorted(norms_by_gen):
        vals = norms_by_gen[gen]
        print(f"    gen{gen}: mean={np.mean(vals):.6f}  std={np.std(vals):.6f}")
    print()

    all_smax = [
        obs["surprise_max"]
        for r in results
        for obs in r["observations_cold"] + r["observations_seeded"] + r.get("observations_gen", [])
        if obs["surprise_max"] > 0
    ]
    if all_smax:
        sarr = np.array(all_smax)
        print(f"── Surprise_max distribution ({len(sarr)} samples) ──")
        print(f"  min={sarr.min():.3f}  max={sarr.max():.3f}"
              f"  mean={sarr.mean():.3f}  std={sarr.std():.3f}")
    print()

    mean_curv = float(np.mean(all_curvs)) if all_curvs else 0.0
    std_curv = float(np.std(all_curvs)) if all_curvs else 0.0
    print("── Outliers (curvature > mean+std, betti[1]>0, or npf>0) ──")
    found_outlier = False
    for r in results:
        gen, v = r["generation"], r["variant_idx"]
        all_obs = (
            r["observations_cold"]
            + r["observations_seeded"]
            + r.get("observations_gen", [])
        )
        for obs in all_obs:
            high_curv = obs["curvature"] > mean_curv + std_curv
            high_betti = obs["betti"][1] > 0
            high_npf = obs["n_persistent_features"] > 0
            if high_curv or high_betti or high_npf:
                found_outlier = True
                reason = []
                if high_curv:  reason.append(f"curv={obs['curvature']:.6f}")
                if high_betti: reason.append(f"betti={obs['betti']}")
                if high_npf:   reason.append(f"npf={obs['n_persistent_features']}")
                print(f"  gen{gen} v{v} t={obs['temperature']:.2f}:")
                print(f"    \"{obs['text'][:80]}...\"")
                print(f"    {', '.join(reason)}")
    if not found_outlier:
        print("  None found.")
    print()

    print("── End of description ──")
    print("What do you see?  We have no verdict.  The data is the record.")


def sequence_main(args: argparse.Namespace) -> None:
    if args.analyze:
        results = []
        for f in sorted(_SEQ_RESULTS_DIR.glob("generation_*.json")):
            try:
                results.extend(json.loads(f.read_text()))
            except Exception:
                pass
        sequence_analyze(results)
        return

    n_gen = 3 if args.quick else args.generations
    n_var = 3 if args.quick else args.variants

    t0 = time.time()
    results = sequence_run_experiment(
        n_generations=n_gen,
        variants_per_gen=n_var,
        seed_base=args.seed,
    )
    elapsed = time.time() - t0
    print(f"Runtime: {elapsed:.1f}s  ({len(results)} variant records)\n")

    sequence_analyze(results)

    summary_file = _SEQ_RESULTS_DIR / "summary.json"
    summary_file.write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("loss_trajectories",)} for r in results],
        indent=2, default=str,
    ))
    print(f"\nResults -> {_SEQ_RESULTS_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# PROBE: basin  —  Loss-landscape geometry
# ══════════════════════════════════════════════════════════════════════════════

_BASIN_RESULTS_DIR = SCRIPT_DIR / "experiment_results" / "basin_geometry"
_BASIN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Probe-local config constants
_B_N_AGENTS = 8
_B_N_DIRECTIONS = 32
_B_N_STEPS = 12
_B_STEP_SIZES = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
_B_CONVERGE_STEPS = 40

assert len(_B_STEP_SIZES) == _B_N_STEPS, "_B_STEP_SIZES must have _B_N_STEPS entries"


def _basin_compute_loss(agent: TopoAgent, texts: List[str]) -> float:
    """Compute mean cross-entropy loss over texts. No gradient update."""
    total = 0.0
    count = 0
    for text in texts:
        clean = agent._clean(text)
        if len(clean) < 2:
            continue
        tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]
        loss = RV(0.0)
        for t in range(n):
            logits, keys, vals = _forward(tokens[t], t, keys, vals, agent.sd)
            probs = _softmax(logits)
            loss = loss + (probs[tokens[t + 1]].log()) * (-1.0 / n)
        total += float(loss.data)
        count += 1
    return total / count if count else float("nan")


def _basin_converge(
    agent: TopoAgent,
    texts: List[str],
    n_steps: int = _B_CONVERGE_STEPS,
) -> Tuple[List[float], List[List[float]]]:
    """Run Adam gradient steps until convergence.

    Returns (loss_trajectory, weight_trajectory).
    """
    loss_trajectory = []
    weight_trajectory = []
    for text in texts:
        clean = agent._clean(text)
        if len(clean) < 2:
            continue
        tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        steps_this_text = n_steps // max(len(texts), 1)
        for _ in range(max(steps_this_text, 5)):
            loss_val = adam_step(agent, tokens, n, LR)
            loss_trajectory.append(round(loss_val, 6))
            weight_trajectory.append(get_weight_vector(agent).tolist())
    return loss_trajectory, weight_trajectory


def _basin_probe_direction(
    agent: TopoAgent,
    w_star: np.ndarray,
    direction: np.ndarray,
    texts: List[str],
    step_sizes: List[float] = _B_STEP_SIZES,
) -> dict:
    """Walk from w* along direction, measure loss at each step.

    Also walks in the negative direction to test asymmetry.
    """
    loss_at_center = _basin_compute_loss(agent, texts)
    pos_losses = []
    neg_losses = []

    for step in step_sizes:
        set_weight_vector(agent, w_star + step * direction)
        pos_losses.append(round(_basin_compute_loss(agent, texts), 6))
        set_weight_vector(agent, w_star - step * direction)
        neg_losses.append(round(_basin_compute_loss(agent, texts), 6))

    set_weight_vector(agent, w_star)

    eps = step_sizes[0]
    curvature_est = (pos_losses[0] + neg_losses[0] - 2 * loss_at_center) / (eps ** 2)
    asymmetry = float(np.mean(np.abs(np.array(pos_losses) - np.array(neg_losses))))

    width_pos = next((s for i, s in enumerate(step_sizes) if pos_losses[i] >= 2 * loss_at_center), None)
    width_neg = next((s for i, s in enumerate(step_sizes) if neg_losses[i] >= 2 * loss_at_center), None)

    return {
        "loss_at_center": round(loss_at_center, 6),
        "pos_losses":     pos_losses,
        "neg_losses":     neg_losses,
        "step_sizes":     step_sizes,
        "curvature_est":  round(curvature_est, 6),
        "asymmetry":      round(asymmetry, 6),
        "width_pos":      width_pos,
        "width_neg":      width_neg,
    }


def _basin_run_agent(
    agent_idx: int,
    texts: List[str],
    seed: int,
    n_directions: int = _B_N_DIRECTIONS,
    config: Optional[dict] = None,
) -> dict:
    np.random.seed(seed % 2**31)
    random.seed(seed)

    agent = TopoAgent(config=config or {})
    conv_trajectory, weight_traj = _basin_converge(agent, texts)
    w_star = get_weight_vector(agent)
    w_norm = float(np.linalg.norm(w_star))
    loss_converged = _basin_compute_loss(agent, texts)

    print(f"  agent {agent_idx}: converged  norm={w_norm:.4f}  loss={loss_converged:.4f}")

    rng_np = np.random.default_rng(seed)
    directions = []
    probe_results = []

    for d_idx in range(n_directions):
        raw = rng_np.standard_normal(len(w_star))
        direction = raw / np.linalg.norm(raw)
        directions.append(direction.tolist())

        probe = _basin_probe_direction(agent, w_star, direction, texts)
        probe_results.append(probe)

        if d_idx % 8 == 0:
            print(f"    dir {d_idx:2d}: curv={probe['curvature_est']:.4f}  asym={probe['asymmetry']:.4f}"
                  f"  width_pos={probe['width_pos']}  width_neg={probe['width_neg']}")

    curvatures = [p["curvature_est"] for p in probe_results]
    asymmetries = [p["asymmetry"] for p in probe_results]
    widths_pos = [p["width_pos"] for p in probe_results if p["width_pos"] is not None]
    widths_neg = [p["width_neg"] for p in probe_results if p["width_neg"] is not None]

    summary = {
        "curvature_mean":   round(float(np.mean(curvatures)), 6),
        "curvature_std":    round(float(np.std(curvatures)), 6),
        "curvature_min":    round(float(np.min(curvatures)), 6),
        "curvature_max":    round(float(np.max(curvatures)), 6),
        "asymmetry_mean":   round(float(np.mean(asymmetries)), 6),
        "asymmetry_std":    round(float(np.std(asymmetries)), 6),
        "width_pos_median": round(float(np.median(widths_pos)), 4) if widths_pos else None,
        "width_neg_median": round(float(np.median(widths_neg)), 4) if widths_neg else None,
        "n_no_width_pos":   n_directions - len(widths_pos),
        "n_no_width_neg":   n_directions - len(widths_neg),
    }

    return {
        "agent_idx":         agent_idx,
        "seed":              seed,
        "texts_preview":     [t[:50] for t in texts],
        "config":            config or {},
        "convergence_norm":  round(w_norm, 6),
        "convergence_loss":  round(loss_converged, 6),
        "conv_trajectory":   conv_trajectory,
        "weight_trajectory": weight_traj,
        "probe_results":     probe_results,
        "basin_summary":     summary,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }


def basin_run_experiment(
    n_agents: int = _B_N_AGENTS,
    n_directions: int = _B_N_DIRECTIONS,
    seed_base: int = 42,
) -> List[dict]:
    corpus = load_corpus(min_words=15)
    rng = random.Random(seed_base)
    all_results: List[dict] = []

    print("[Basin geometry experiment]")
    print(f"Corpus: {len(corpus)} passages")
    print(f"Agents: {n_agents}  Directions/agent: {n_directions}")
    print(f"Convergence steps: {_B_CONVERGE_STEPS}  Step sizes: {_B_STEP_SIZES}")
    print(f"No generation. No encounter_complex. Just geometry.\n")

    half = n_agents // 2
    fixed_texts = rng.sample(corpus, min(4, len(corpus)))

    configs = [
        {"learn_steps": 10, "learn_lr": 0.001, "temperature": 0.8, "alpha": 0.85},
        {"learn_steps": 10, "learn_lr": 0.01,  "temperature": 0.8, "alpha": 0.85},
        {"learn_steps": 10, "learn_lr": 0.1,   "temperature": 0.8, "alpha": 0.85},
        {"learn_steps": 10, "learn_lr": 0.001, "temperature": 0.8, "alpha": 0.85},
    ]

    for i in range(n_agents):
        seed = seed_base + i * 17

        if i < half:
            texts = fixed_texts
            corpus_condition = "fixed"
        else:
            texts = rng.sample(corpus, min(4, len(corpus)))
            corpus_condition = "random"

        config = configs[i % len(configs)]
        print(f"  Agent {i} [{corpus_condition}  lr={config['learn_lr']}]")

        result = _basin_run_agent(i, texts, seed, n_directions, config)
        result["corpus_condition"] = corpus_condition
        all_results.append(result)

        norm = result["convergence_norm"]
        bs = result["basin_summary"]
        print(f"    norm={norm:.4f}  "
              f"curv_mean={bs['curvature_mean']:.4f}±{bs['curvature_std']:.4f}  "
              f"asym_mean={bs['asymmetry_mean']:.4f}  "
              f"width_pos={bs['width_pos_median']}  "
              f"width_neg={bs['width_neg_median']}")

    result_file = _BASIN_RESULTS_DIR / f"basin_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    result_file.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults -> {result_file}")
    return all_results


def basin_analyze(results: List[dict]) -> None:
    if not results:
        print("No results to analyze.")
        return

    print("=" * 70)
    print("BASIN GEOMETRY — OPEN DESCRIPTION")
    print("=" * 70)
    print()

    norms = [r["convergence_norm"] for r in results]
    norms_arr = np.array(norms)
    print("── Convergence norms ──")
    print(f"  n={len(norms)}  mean={norms_arr.mean():.4f}  std={norms_arr.std():.4f}"
          f"  min={norms_arr.min():.4f}  max={norms_arr.max():.4f}"
          f"  range={norms_arr.max()-norms_arr.min():.4f}")
    print(f"  cv (std/mean) = {norms_arr.std()/norms_arr.mean():.4f}")
    for r in results:
        cond = r.get("corpus_condition", "?")
        lr = r["config"].get("learn_lr", "?")
        print(f"  agent {r['agent_idx']} [{cond} lr={lr}]: norm={r['convergence_norm']:.6f}"
              f"  loss={r['convergence_loss']:.4f}")
    print()

    fixed_norms = [r["convergence_norm"] for r in results if r.get("corpus_condition") == "fixed"]
    random_norms = [r["convergence_norm"] for r in results if r.get("corpus_condition") == "random"]
    if fixed_norms and random_norms:
        print("── Corpus condition comparison ──")
        print(f"  fixed corpus:  mean={np.mean(fixed_norms):.4f}  std={np.std(fixed_norms):.4f}")
        print(f"  random corpus: mean={np.mean(random_norms):.4f}  std={np.std(random_norms):.4f}")
        print(f"  difference in means: {abs(np.mean(fixed_norms)-np.mean(random_norms)):.4f}")
    print()

    all_curv_means = [r["basin_summary"]["curvature_mean"] for r in results]
    all_curv_stds = [r["basin_summary"]["curvature_std"] for r in results]
    print("── Basin curvature (across random directions) ──")
    print(f"  curvature_mean across agents: {np.mean(all_curv_means):.4f} ± {np.std(all_curv_means):.4f}")
    print(f"  curvature_std within agents:  {np.mean(all_curv_stds):.4f} (mean within-agent std)")
    print(f"  high within-agent std = anisotropic basin (some directions steep, some flat)")
    print(f"  low within-agent std  = isotropic basin (same curvature in all directions)")
    for r in results:
        bs = r["basin_summary"]
        print(f"  agent {r['agent_idx']}: curv={bs['curvature_mean']:.4f}±{bs['curvature_std']:.4f}"
              f"  asym={bs['asymmetry_mean']:.4f}")
    print()

    all_wp = [r["basin_summary"]["width_pos_median"] for r in results if r["basin_summary"]["width_pos_median"]]
    all_wn = [r["basin_summary"]["width_neg_median"] for r in results if r["basin_summary"]["width_neg_median"]]
    print("── Basin width (step size at which loss doubles) ──")
    if all_wp:
        print(f"  positive direction: median={np.median(all_wp):.4f}  range=[{min(all_wp)}, {max(all_wp)}]")
    else:
        print(f"  positive direction: loss never doubled within step range — very wide basin")
    if all_wn:
        print(f"  negative direction: median={np.median(all_wn):.4f}  range=[{min(all_wn)}, {max(all_wn)}]")
    else:
        print(f"  negative direction: loss never doubled within step range — very wide basin")
    print()

    all_asym = [r["basin_summary"]["asymmetry_mean"] for r in results]
    print("── Asymmetry (mean |loss+ - loss-| across step sizes) ──")
    print(f"  mean={np.mean(all_asym):.4f}  std={np.std(all_asym):.4f}")
    print(f"  near-zero asymmetry = symmetric basin")
    print(f"  high asymmetry = the basin has a preferred direction (not a simple bowl)")
    print()

    by_lr: dict = {}
    for r in results:
        lr = str(r["config"].get("learn_lr", "?"))
        by_lr.setdefault(lr, []).append(r["convergence_norm"])
    if len(by_lr) > 1:
        print("── Norm by learning rate ──")
        for lr, ns in sorted(by_lr.items()):
            print(f"  lr={lr}: norms={[round(n, 4) for n in ns]}  mean={np.mean(ns):.4f}")
        print()

    print("── End of description ──")
    print("What is the shape of the bowl?  The data is the record.")


def basin_main(args: argparse.Namespace) -> None:
    if args.analyze:
        results = []
        for f in sorted(_BASIN_RESULTS_DIR.glob("basin_*.json")):
            try:
                results.extend(json.loads(f.read_text()))
            except Exception:
                pass
        basin_analyze(results)
        return

    n_agents = 3 if args.quick else args.agents
    n_directions = 8 if args.quick else args.directions

    t0 = time.time()
    results = basin_run_experiment(
        n_agents=n_agents,
        n_directions=n_directions,
        seed_base=args.seed,
    )
    elapsed = time.time() - t0
    print(f"\nRuntime: {elapsed:.1f}s")
    basin_analyze(results)


# ══════════════════════════════════════════════════════════════════════════════
# PROBE: sgd  —  SGD vs Adam ablation
# ══════════════════════════════════════════════════════════════════════════════

_SGD_RESULTS_DIR = SCRIPT_DIR / "experiment_results" / "sgd_ablation"
_SGD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Probe-local config constants
_G_CONVERGE_STEPS = 40


def _sgd_perturb_weights(agent: TopoAgent, rng: np.random.Generator, scale: float = 0.01) -> None:
    """Add small Gaussian noise to initial weights so each seed starts differently."""
    for p in agent.params:
        p.data += float(rng.normal(0, scale))


def _sgd_converge(
    agent: TopoAgent,
    texts: List[str],
    step_fn,
    lr: float,
    n_steps: int = _G_CONVERGE_STEPS,
) -> Tuple[float, float]:
    """Run step_fn until convergence. Returns (final_norm, final_loss)."""
    losses = []
    for text in texts:
        clean = agent._clean(text)
        if len(clean) < 2:
            continue
        tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        steps_this = n_steps // max(len(texts), 1)
        for _ in range(max(steps_this, 5)):
            l = step_fn(agent, tokens, n, lr)
            losses.append(l)
    wv = np.array([p.data for p in agent.params])
    return float(np.linalg.norm(wv)), losses[-1] if losses else float("nan")


def sgd_run(n_seeds: int = 5, seed_base: int = 42) -> dict:
    corpus = load_corpus(min_words=15)

    print("=" * 60)
    print("SGD ABLATION v2 — independent seeds")
    print("=" * 60)
    print(f"Corpus pool: {len(corpus)} passages, Steps: {_G_CONVERGE_STEPS}, LR: {LR}")
    seeds = [seed_base + i * 17 for i in range(n_seeds)]
    print(f"Seeds: {seeds}")
    print(f"Each seed: different corpus sample + weight perturbation")
    print()

    results = {"adam": [], "sgd": []}

    print("=" * 60)
    print("ADAM (creature default)")
    print("=" * 60)
    for seed in seeds:
        rng_py = random.Random(seed)
        rng_np = np.random.default_rng(seed)
        texts = rng_py.sample(corpus, min(4, len(corpus)))
        agent = TopoAgent()
        _sgd_perturb_weights(agent, rng_np, scale=0.01)
        norm, loss = _sgd_converge(agent, texts, adam_step, LR)
        results["adam"].append({"seed": seed, "norm": round(norm, 4), "loss": round(loss, 4)})
        print(f"  seed={seed:5d}  final_norm={norm:.4f}  loss_final={loss:.4f}")
    adam_norms = np.array([r["norm"] for r in results["adam"]])
    print(f"  mean={adam_norms.mean():.4f}  std={adam_norms.std():.4f}")
    print()

    print("=" * 60)
    print("SGD (no momentum, no adaptive)")
    print("=" * 60)
    for seed in seeds:
        rng_py = random.Random(seed)
        rng_np = np.random.default_rng(seed)
        texts = rng_py.sample(corpus, min(4, len(corpus)))
        agent = TopoAgent()
        _sgd_perturb_weights(agent, rng_np, scale=0.01)
        norm, loss = _sgd_converge(agent, texts, sgd_step, LR)
        results["sgd"].append({"seed": seed, "norm": round(norm, 4), "loss": round(loss, 4)})
        print(f"  seed={seed:5d}  final_norm={norm:.4f}  loss_final={loss:.4f}")
    sgd_norms = np.array([r["norm"] for r in results["sgd"]])
    print(f"  mean={sgd_norms.mean():.4f}  std={sgd_norms.std():.4f}")
    print()

    diff = abs(adam_norms.mean() - sgd_norms.mean())
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"  Adam mean norm: {adam_norms.mean():.4f} ± {adam_norms.std():.4f}")
    print(f"  SGD  mean norm: {sgd_norms.mean():.4f} ± {sgd_norms.std():.4f}")
    print(f"  Difference:     {diff:.4f}")

    if diff < 1.0:
        print("  -> SAME attractor. Optimizer-independent. Proceed to step 2.")
    else:
        print("  -> DIFFERENT attractor. The norm is an OPTIMIZER ARTIFACT.")
        print("     Do not proceed to step 2.")

    out = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "version":            2,
        "lr":                 LR,
        "converge_steps":     _G_CONVERGE_STEPS,
        "n_seeds":            n_seeds,
        "perturbation_scale": 0.01,
        "adam":               results["adam"],
        "sgd":                results["sgd"],
        "adam_mean":          round(float(adam_norms.mean()), 4),
        "sgd_mean":           round(float(sgd_norms.mean()), 4),
        "difference":         round(diff, 4),
        "verdict":            "structural" if diff < 1.0 else "optimizer_artifact",
    }
    outfile = _SGD_RESULTS_DIR / f"sgd_ablation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    outfile.write_text(json.dumps(out, indent=2))
    print(f"\nResults -> {outfile}")
    return out


def sgd_main(args: argparse.Namespace) -> None:
    n_seeds = 3 if args.quick else args.seeds
    sgd_run(n_seeds=n_seeds, seed_base=args.seed_base)


# ══════════════════════════════════════════════════════════════════════════════
# PROBE: analyze  —  Post-hoc statistical analysis
# ══════════════════════════════════════════════════════════════════════════════

_ANA_PCA_RESULTS_DIR = SCRIPT_DIR / "experiment_results" / "pca_topology"
_ANA_ACT_RESULTS_DIR = SCRIPT_DIR / "experiment_results" / "activation_topology"


def _ana_load_results(results_dir: Path) -> list:
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


def _ana_analyse_single(results: list, experiment_name: str, output_dir: Path) -> dict:
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

    print(f"\n{'-' * 70}")
    print("TOPOLOGY vs LOSS CORRELATION")
    print(f"{'-' * 70}")
    all_tp = [r["topology"]["total_persistence_h1"] for r in results]
    all_loss = [r["final_loss"] for r in results]
    if len(all_tp) > 3:
        corr = float(np.corrcoef(all_tp, all_loss)[0, 1])
        print(f"  Pearson r(tp_h1, final_loss) = {corr:.4f}")

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


def _ana_compare_experiments(pca_stats: dict, act_stats: dict) -> None:
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

    for name, stats in [("PCA-first", pca_stats), ("Activation", act_stats)]:
        real = [c for c in ["random", "coherent", "diverse"] if c in stats]
        if len(real) >= 2:
            means = [stats[c]["tp_mean"] for c in real]
            spread = max(means) - min(means)
            print(f"\n  {name} inter-condition tp_h1 spread: {spread:.4f}")


def analyze_main(args: argparse.Namespace) -> None:
    pca_stats, act_stats = None, None

    if args.experiment in ("pca", "both"):
        d = args.results_dir if args.results_dir and args.experiment == "pca" else _ANA_PCA_RESULTS_DIR
        results = _ana_load_results(d)
        if results:
            pca_stats = _ana_analyse_single(results, "PCA-first persistence", d)
        elif args.experiment == "pca":
            print(f"No PCA results found in {d}")

    if args.experiment in ("activation", "both"):
        d = args.results_dir if args.results_dir and args.experiment == "activation" else _ANA_ACT_RESULTS_DIR
        results = _ana_load_results(d)
        if results:
            act_stats = _ana_analyse_single(results, "Activation-space persistence", d)
        elif args.experiment == "activation":
            print(f"No activation results found in {d}")

    if pca_stats and act_stats:
        _ana_compare_experiments(pca_stats, act_stats)

    if not pca_stats and not act_stats:
        print("No results found. Run one of:")
        print("  python experiments.py weight      # PCA-first approach")
        print("  python experiments.py activation  # activation-space approach")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7 — HARNESS ABLATION (NLAH RQ2)
# ══════════════════════════════════════════════════════════════════════════════
#
# Pan et al. (2026) showed that once harness modules are explicit, they
# become a search space: you can compose, ablate, and measure module-level
# effects under shared runtime assumptions.
#
# creature_dgm_h's _build_creature_context() is a natural-language harness
# with five named modules: identity, mechanism, state, autobiography, journal.
# The creature already has measurement infrastructure (Betti numbers, winding,
# curvature, fitness).  This experiment applies the RQ2 methodology: run
# the creature under each ablation condition and measure which modules
# actually change the topology of generated text vs. which are decorative.
#
# Since this experiment needs a running FM (Nemotron), it operates in two
# modes:
#   1. --offline (default): uses the creature's own character-level model
#      with deterministic text (no FM needed).  Measures the encounter
#      topology of corpus texts under each context configuration.
#   2. --live: requires FM at localhost:8000.  Generates text under each
#      ablation condition and measures the topology of the generated output.
#
# The breath-level gate (BreathGate) is also tested here: does acceptance-
# gating improve structural outcomes across ablation conditions?

_H_RESULTS_DIR = Path(__file__).resolve().parent / "experiment_results" / "harness_ablation"


def _harness_ablation_conditions() -> List[dict]:
    """Generate all ablation conditions.

    Following RQ2: start from full (all modules), then remove one at a time.
    Also include: bare (no modules) and pairs removed.
    """
    from vybn import CONTEXT_MODULES
    conditions = []

    # Full: all modules active
    conditions.append({"name": "full", "exclude": set()})

    # Single-module ablations (remove one at a time)
    for mod in CONTEXT_MODULES:
        conditions.append({"name": f"w/o_{mod}", "exclude": {mod}})

    # Bare: no modules at all
    conditions.append({"name": "bare", "exclude": set(CONTEXT_MODULES)})

    # Pairs removed (most informative pairs)
    pairs = [
        ("identity", "autobiography"),  # remove personal history
        ("mechanism", "state"),          # remove creature self-knowledge
        ("autobiography", "journal"),    # remove temporal context
    ]
    for a, b in pairs:
        conditions.append({"name": f"w/o_{a}+{b}", "exclude": {a, b}})

    return conditions


def _harness_run_offline_condition(
    condition: dict,
    texts: List[str],
    seed: int,
    use_gate: bool = False,
) -> dict:
    """Run a single offline ablation condition.

    Measures encounter topology of corpus texts processed through
    the creature under this context configuration.  No FM needed.
    """
    from vybn import (
        TopoAgent, Organism, encounter_complex, EncounterComplex,
        BreathGate, _build_creature_context, fitness,
    )

    rng = random.Random(seed)
    agent = TopoAgent()
    organism = Organism.load()
    gate = BreathGate() if use_gate else None

    # Build context under this ablation to measure its character count
    context = _build_creature_context(exclude=condition["exclude"])
    context_len = len(context)

    curvatures = []
    betti_b0s, betti_b1s = [], []
    persist_features = []
    weight_vectors = []
    losses_before, losses_after = [], []
    gate_verdicts = []
    phase_shifts = []

    for text in texts:
        cx = encounter_complex(text)
        loss_before, _ = agent.predict(text)
        losses_before.append(loss_before)

        losses = agent.learn(
            text, steps=5,
            encounter_cx=cx,
            persistent_state=organism.persistent,
        )
        loss_after, _ = agent.predict(text)
        losses_after.append(loss_after)

        delta = organism.absorb_encounter(cx)

        # Winding measurement
        winding_record = None
        if hasattr(agent, '_weight_trajectory') and len(agent._weight_trajectory) >= 3:
            winding_record = organism.absorb_winding(agent._weight_trajectory)
            weight_vectors.extend(agent._weight_trajectory)

        # Phase stats
        if hasattr(agent, '_phase_stats'):
            phase_shifts.append(agent._phase_stats.get("mean_phase_shift", 0.0))
            organism.absorb_phases(
                agent.module_holonomies,
                genesis_signal=agent._phase_stats.get("genesis_signal", 0.0),
                mean_phase_shift=agent._phase_stats.get("mean_phase_shift", 0.0))

        curvatures.append(cx.curvature)
        betti_b0s.append(cx.betti[0])
        betti_b1s.append(cx.betti[1])
        persist_features.append(cx.n_persistent_features)

        # Breath gate evaluation
        if gate is not None:
            verdict = gate.evaluate(cx, delta, winding_record)
            gate.record_outcome(verdict.accept)
            gate_verdicts.append({
                "accept": verdict.accept,
                "genesis": round(verdict.genesis_pressure, 4),
                "decoherence": round(verdict.decoherence_pressure, 4),
                "threshold": round(verdict.threshold, 4),
            })

    # Compute fitness
    fit = fitness(
        texts, [],
        agent.loss_history,
        persistent_state=organism.persistent,
        alpha=0.85,
        weight_vectors=weight_vectors if weight_vectors else None,
        phase_stats=agent._phase_stats if hasattr(agent, '_phase_stats') else None,
    )

    return {
        "experiment": "harness_ablation",
        "condition": condition["name"],
        "excluded_modules": sorted(condition["exclude"]),
        "n_active_modules": len(CONTEXT_MODULES) - len(condition["exclude"]),
        "context_char_count": context_len,
        "seed": seed,
        "n_texts": len(texts),
        "use_gate": use_gate,
        "metrics": {
            "mean_curvature": round(float(np.mean(curvatures)), 6) if curvatures else 0.0,
            "std_curvature": round(float(np.std(curvatures)), 6) if curvatures else 0.0,
            "mean_b0": round(float(np.mean(betti_b0s)), 2) if betti_b0s else 0.0,
            "mean_b1": round(float(np.mean(betti_b1s)), 2) if betti_b1s else 0.0,
            "mean_persist_features": round(float(np.mean(persist_features)), 2) if persist_features else 0.0,
            "mean_loss_before": round(float(np.mean(losses_before)), 4) if losses_before else 0.0,
            "mean_loss_after": round(float(np.mean(losses_after)), 4) if losses_after else 0.0,
            "mean_loss_improvement": round(
                float(np.mean([b - a for b, a in zip(losses_before, losses_after)])), 4
            ) if losses_before else 0.0,
            "mean_phase_shift": round(float(np.mean(phase_shifts)), 6) if phase_shifts else 0.0,
            "fitness": fit["fitness"],
            "topological_richness": fit.get("topological_richness", 0.0),
            "weight_topo": fit.get("weight_topo", 0.0),
            "phase_winding": fit.get("phase_winding", 0.0),
        },
        "gate_summary": gate.summary() if gate else None,
        "gate_accept_rate": (
            round(sum(1 for v in gate_verdicts if v["accept"]) / len(gate_verdicts), 3)
            if gate_verdicts else None
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def harness_run_experiment(
    n_texts: int = 8,
    seed: int = 42,
    quick: bool = False,
) -> List[dict]:
    """Run the full harness ablation experiment.

    For each condition (full, w/o_X, bare, pairs), runs the creature
    offline and measures topological metrics.  Each condition is run
    twice: once without the breath gate, once with it.
    """
    from vybn import CONTEXT_MODULES

    texts = load_corpus()
    rng = random.Random(seed)
    if len(texts) > n_texts:
        texts = farthest_point_sample(texts, n_texts)
    elif len(texts) < n_texts:
        texts = texts[:n_texts]

    conditions = _harness_ablation_conditions()
    if quick:
        # Quick mode: full, one single ablation, bare
        conditions = [c for c in conditions if c["name"] in ("full", "w/o_mechanism", "bare")]

    results = []
    total = len(conditions) * 2  # with and without gate
    idx = 0

    for condition in conditions:
        for use_gate in [False, True]:
            idx += 1
            gate_label = "gated" if use_gate else "ungated"
            print(f"  [{idx}/{total}] {condition['name']} ({gate_label})...")

            result = _harness_run_offline_condition(
                condition, texts, seed, use_gate=use_gate,
            )
            results.append(result)

            m = result["metrics"]
            print(f"    curv={m['mean_curvature']:.4f}"
                  f" b1={m['mean_b1']:.1f}"
                  f" fitness={m['fitness']:.4f}"
                  f" loss_imp={m['mean_loss_improvement']:.4f}")
            if use_gate and result["gate_accept_rate"] is not None:
                print(f"    gate: accept_rate={result['gate_accept_rate']:.2f}"
                      f" threshold={result['gate_summary']['threshold']:.3f}")

    # Save results
    _H_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = _H_RESULTS_DIR / f"ablation_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Saved: {out_path}")

    # Summary table
    print("\n  ═══ Harness Ablation Summary ═══\n")
    print(f"  {'Condition':<25} {'Gate':<8} {'Curv':>7} {'B1':>5} {'Fitness':>8} {'PhaseΔ':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*5} {'-'*8} {'-'*8}")
    for r in results:
        m = r["metrics"]
        gate_str = "gated" if r["use_gate"] else "-"
        print(f"  {r['condition']:<25} {gate_str:<8}"
              f" {m['mean_curvature']:>7.4f}"
              f" {m['mean_b1']:>5.1f}"
              f" {m['fitness']:>8.4f}"
              f" {m['mean_phase_shift']:>8.6f}")

    # Module-effect analysis (which modules matter?)
    full_ungated = next((r for r in results if r["condition"] == "full" and not r["use_gate"]), None)
    if full_ungated:
        print("\n  ═══ Module Effects (vs full, ungated) ═══\n")
        full_f = full_ungated["metrics"]["fitness"]
        full_c = full_ungated["metrics"]["mean_curvature"]
        for r in results:
            if r["use_gate"] or r["condition"] == "full":
                continue
            m = r["metrics"]
            df = m["fitness"] - full_f
            dc = m["mean_curvature"] - full_c
            direction = "+" if df > 0.001 else "-" if df < -0.001 else "="
            print(f"  {r['condition']:<25} Δfitness={df:+.4f} {direction}"
                  f"  Δcurv={dc:+.4f}")

    return results


def harness_main(args: argparse.Namespace) -> None:
    print("\n═══ Harness Ablation Experiment (NLAH RQ2) ═══\n")
    harness_run_experiment(
        n_texts=4 if args.quick else 8,
        seed=args.seed,
        quick=args.quick,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="probe", metavar="PROBE")
    subparsers.required = True

    # ── weight ──
    p_weight = subparsers.add_parser("weight", help="PCA-first persistence experiment")
    p_weight.add_argument("--quick",   action="store_true", help="3 runs per condition")
    p_weight.add_argument("--analyze", action="store_true", help="Load saved results and analyse")
    p_weight.add_argument("--k",       type=int, default=_W_K, help=f"Texts per run (default {_W_K})")
    p_weight.add_argument("--seed",    type=int, default=42,   help="Random seed base")
    p_weight.add_argument("--pca_dim", type=int, default=_W_PCA_DIM, help=f"PCA target dim (default {_W_PCA_DIM})")

    # ── activation ──
    p_act = subparsers.add_parser("activation", help="Activation-space persistence experiment")
    p_act.add_argument("--quick",   action="store_true", help="3 runs per condition")
    p_act.add_argument("--analyze", action="store_true", help="Load saved results and analyse")
    p_act.add_argument("--k",       type=int, default=_A_K, help=f"Texts per run (default {_A_K})")
    p_act.add_argument("--seed",    type=int, default=42,   help="Random seed base")

    # ── sequence ──
    p_seq = subparsers.add_parser("sequence", help="Natural motion recorder")
    p_seq.add_argument("--quick",       action="store_true",              help="3 generations, 3 variants")
    p_seq.add_argument("--analyze",     action="store_true",              help="Describe saved results")
    p_seq.add_argument("--generations", type=int, default=_S_N_GENERATIONS, help="Number of generations")
    p_seq.add_argument("--variants",    type=int, default=_S_VARIANTS_PER_GEN, help="Variants per generation")
    p_seq.add_argument("--seed",        type=int, default=42,             help="Random seed base")

    # ── basin ──
    p_basin = subparsers.add_parser("basin", help="Loss-landscape geometry experiment")
    p_basin.add_argument("--quick",      action="store_true",           help="3 agents, 8 directions")
    p_basin.add_argument("--analyze",    action="store_true",           help="Analyse saved results")
    p_basin.add_argument("--agents",     type=int, default=_B_N_AGENTS,     help="Number of agents")
    p_basin.add_argument("--directions", type=int, default=_B_N_DIRECTIONS, help="Directions per agent")
    p_basin.add_argument("--seed",       type=int, default=42,          help="Random seed base")

    # ── sgd ──
    p_sgd = subparsers.add_parser("sgd", help="SGD vs Adam ablation")
    p_sgd.add_argument("--quick",     action="store_true", help="3 seeds")
    p_sgd.add_argument("--seeds",     type=int, default=5, help="Number of seeds (default 5)")
    p_sgd.add_argument("--seed-base", type=int, default=42, help="Seed base (default 42)")

    # ── harness ──
    p_harness = subparsers.add_parser("harness", help="Harness ablation (NLAH RQ2)")
    p_harness.add_argument("--quick", action="store_true", help="3 conditions only")
    p_harness.add_argument("--seed",  type=int, default=42, help="Random seed (default 42)")

    # ── analyze ──
    p_ana = subparsers.add_parser("analyze", help="Post-hoc statistical analysis of saved results")
    p_ana.add_argument("--experiment", choices=["pca", "activation", "both"], default="both",
                       help="Which experiment to analyse (default: both)")
    p_ana.add_argument("--results_dir", type=Path, default=None,
                       help="Override results directory (single-experiment mode)")

    args = parser.parse_args()

    dispatch = {
        "weight":     weight_main,
        "activation": activation_main,
        "sequence":   sequence_main,
        "basin":      basin_main,
        "sgd":        sgd_main,
        "harness":    harness_main,
        "analyze":    analyze_main,
    }
    dispatch[args.probe](args)


if __name__ == "__main__":
    main()
