#!/usr/bin/env python3
"""
experiment_activation_topology.py  —  Activation-space persistence experiment.

Complements experiment_weight_topology.py (PCA-first persistence).  Instead of
tracking weight vectors, this experiment captures hidden-layer activations
during training and computes persistent homology on those.

Rationale: activations are lower-dimensional than weights (N_EMBD=16 vs ~4K
parameters) and are more semantically tied to model behaviour — they represent
what the model *computes*, not what it *stores*.  If text selection shapes the
geometry of understanding, the activation manifold is the most natural place to
look.

Activation capture strategy:
- After each gradient step, run a forward pass on a fixed probe sentence.
- Record the hidden-state vector (post-attention, post-MLP) at each position.
- Average across positions to get a single 16-dim activation summary per step.
- Collect these summaries across all steps → point cloud for topology.

Five conditions (same controlled design):
  1. random     2. coherent   3. diverse   4. order   5. synthetic

Usage:
  python experiment_activation_topology.py              # full experiment
  python experiment_activation_topology.py --quick      # 3 runs per condition
  python experiment_activation_topology.py --analyze    # load saved results
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
sys.path.insert(0, str(REPO_ROOT / "spark"))
sys.path.insert(0, str(SCRIPT_DIR))  # must be first: creature_dgm_h/vybn.py, not spark/vybn.py

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

# ── Config ────────────────────────────────────────────────────────────────
K = 5
SNAP_EVERY = 5
STEPS_PER_TEXT = 10
LR = 0.01
PROBE_SENTENCE = "the topology of learning"  # fixed probe for activation capture
RESULTS_DIR = SCRIPT_DIR / "experiment_results" / "activation_topology"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from ripser import ripser as _ripser_fn
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


# ── Activation capture ───────────────────────────────────────────────────

def capture_activations(agent: TopoAgent, probe_text: str = PROBE_SENTENCE) -> np.ndarray:
    """Run a forward pass on probe_text and return mean hidden-state vector.

    The hidden state after the final transformer layer (post-attention + MLP
    residual) is a 16-dim vector per position.  We average across positions to
    get a single summary vector representing the model's current activation
    geometry.
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
        # Replicate the forward pass but extract the hidden state before lm_head
        x = [agent.sd['wte'][tokens[t]][j] + agent.sd['wpe'][t][j] for j in range(N_EMBD)]
        for i in range(N_LAYER):
            xn = _rmsnorm(x)
            q = _linear(xn, agent.sd[f'layer{i}.attn_wq'])
            k = _linear(xn, agent.sd[f'layer{i}.attn_wk'])
            v = _linear(xn, agent.sd[f'layer{i}.attn_wv'])
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
            ao = _linear(ho, agent.sd[f'layer{i}.attn_wo'])
            x = [x[j] + ao[j] for j in range(N_EMBD)]
            xn2 = _rmsnorm(x)
            h1 = _linear(xn2, agent.sd[f'layer{i}.mlp_fc1'])
            h1 = [hi * (RV(1.0) / (RV(1.0) + (hi * (-1)).exp())) for hi in h1]
            h2 = _linear(h1, agent.sd[f'layer{i}.mlp_fc2'])
            x = [x[j] + h2[j] for j in range(N_EMBD)]

        hidden_states.append(np.array([xi.data for xi in x]))

    if not hidden_states:
        return np.zeros(N_EMBD)

    return np.mean(hidden_states, axis=0)


# ── Topology utilities ────────────────────────────────────────────────────

def compute_topology(points: np.ndarray) -> dict:
    """Persistent homology on an (N, D) activation-space point cloud.

    Since activations are already low-dimensional (D=16), no PCA needed.
    """
    n = len(points)
    if n < 3:
        return {
            "betti_0": n, "betti_1": 0,
            "total_persistence_h0": 0.0, "total_persistence_h1": 0.0,
            "persistence_entropy_h1": 0.0,
            "diagram_h1": [], "n_points": n,
            "method": "trivial", "space_dim": points.shape[1] if points.ndim == 2 else 0,
        }

    if RIPSER_AVAILABLE:
        result = _ripser_fn(points, maxdim=1)
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
        D = _distance_matrix(points)
        pairs, (b0, b1, _) = _persistence_pairs(D)
        finite = [(birth, death) for birth, death in pairs if death != float("inf")]
        tp_h0 = sum(d - b for b, d in finite)
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
        "space_dim": points.shape[1] if points.ndim == 2 else 0,
    }


def pseudo_wasserstein(diag_a: list, diag_b: list) -> float:
    """L∞ approximation between two H1 persistence diagrams."""
    def lifetimes(diag):
        return sorted([d - b for b, d in diag if d < 1e9], reverse=True)
    la, lb = lifetimes(diag_a), lifetimes(diag_b)
    max_len = max(len(la), len(lb), 1)
    la += [0.0] * (max_len - len(la))
    lb += [0.0] * (max_len - len(lb))
    return float(max(abs(a - b) for a, b in zip(la, lb)))


# ── Single run ────────────────────────────────────────────────────────────

def run_condition(
    texts: List[str],
    seed: int,
    label: str,
    condition: str,
    run_idx: int,
) -> dict:
    """Train agent on texts, capturing activation snapshots via a fixed probe."""
    rng = random.Random(seed)
    np.random.seed(seed % 2**31)

    agent = TopoAgent(config={"learn_steps": STEPS_PER_TEXT, "learn_lr": LR})
    activation_snapshots = []
    loss_log = []

    for text in texts:
        clean = agent._clean(text)
        if len(clean) < 2:
            continue
        tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
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
                activation_snapshots.append(capture_activations(agent))

    if not activation_snapshots:
        activation_snapshots.append(capture_activations(agent))

    points = np.array(activation_snapshots)
    topo = compute_topology(points)
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


# ── Corpus helpers (shared with PCA experiment) ──────────────────────────

def load_corpus(min_words: int = 40) -> List[str]:
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
    rng = random.Random(seed)
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(length_chars))


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
        print(f"[Activation-space topology experiment]")
        print(f"Corpus: {len(corpus)} passages  K={k}  SNAP_EVERY={SNAP_EVERY}  STEPS={STEPS_PER_TEXT}")
        print(f"Activation dim: {N_EMBD}  (no PCA needed)")
        print(f"Probe sentence: {PROBE_SENTENCE!r}")
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
                f"snaps={result['n_snapshots']} "
                f"dim={topo['space_dim']} "
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
        _save(run_condition(texts, seed_base + i, "random", "random", i))

    # ── Condition 2: Coherent sets ──
    if verbose:
        print(f"\n=== Condition 2: coherent sets (N={n_coherent}) ===")
    clusters = cluster_texts(corpus, n_clusters=max(3, n_coherent // 3))
    rng2 = random.Random(seed_base + 100)
    for i in range(n_coherent):
        cluster = clusters[i % len(clusters)]
        texts = rng2.sample(cluster, min(k, len(cluster)))
        while len(texts) < k:
            texts.append(rng2.choice(cluster))
        _save(run_condition(texts, seed_base + 100 + i, "coherent", "coherent", i))

    # ── Condition 3: Diverse sets ──
    if verbose:
        print(f"\n=== Condition 3: diverse sets (N={n_diverse}) ===")
    diverse_base = farthest_point_sample(corpus, min(k * 3, len(corpus)))
    rng3 = random.Random(seed_base + 200)
    for i in range(n_diverse):
        texts = rng3.sample(diverse_base, min(k, len(diverse_base)))
        _save(run_condition(texts, seed_base + 200 + i, "diverse", "diverse", i))

    # ── Condition 4: Order permutations ──
    if verbose:
        print(f"\n=== Condition 4: order permutations (N={n_order}) ===")
    import itertools
    fixed_texts = corpus[:k] if len(corpus) >= k else corpus
    perms = list(itertools.permutations(range(len(fixed_texts))))
    rng4 = random.Random(seed_base + 300)
    selected_perms = rng4.sample(perms, min(n_order, len(perms)))
    for i, perm in enumerate(selected_perms):
        texts = [fixed_texts[j] for j in perm]
        _save(run_condition(texts, seed_base + 300 + i, "order", "order", i))

    # ── Condition 5: Synthetic ──
    if verbose:
        print(f"\n=== Condition 5: synthetic random sequences (N={n_synthetic}) ===")
    avg_len = int(np.mean([len(t) for t in corpus[:20]])) if corpus else 200
    for i in range(n_synthetic):
        texts = [
            synthetic_text(avg_len, seed=seed_base + 400 + i * k + j)
            for j in range(k)
        ]
        _save(run_condition(texts, seed_base + 400 + i, "synthetic", "synthetic", i))

    return all_results


# ── Analysis ──────────────────────────────────────────────────────────────

def load_results() -> List[dict]:
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            pass
    return results


def summarise(results: List[dict]) -> dict:
    from collections import defaultdict
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


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Activation-space topology experiment")
    parser.add_argument("--quick", action="store_true", help="3 runs per condition")
    parser.add_argument("--analyze", action="store_true", help="Load saved results")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.analyze:
        results = load_results()
        if not results:
            print("No saved results. Run the experiment first.")
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
        slim = []
        for r in results:
            slim.append({k: v for k, v in r.items() if k != "topology"}
                       | {"topo_summary": {kk: vv for kk, vv in r["topology"].items()
                                           if kk != "diagram_h1"}})
        json.dump(slim, f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
