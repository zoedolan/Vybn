#!/usr/bin/env python3
"""
experiment_sequence_topology.py  —  Final activation-topology experiment.

Previous activation experiments compressed each probe pass to a single
mean vector across token positions, discarding sequential structure before
ripser ever saw it.  This experiment preserves that structure.

Key change:  capture_sequence() returns the full (T, 16) matrix of
per-token hidden states, NOT a mean.  We then build the point cloud by
stacking consecutive per-token snapshots as a sliding window, so the cloud
lives in R^(W*T*16).  This manufactures density from sequential structure
without requiring a larger model.

Design:
  - Two conditions: real (random corpus sample) vs synthetic (random chars)
  - 5 seeds x 2 conditions = 10 runs total
  - Snapshot every gradient step (SNAP_EVERY = 1)
  - 15 steps per text, 3 texts per run  →  ~45 snapshots per run
  - Sliding window W=3 over consecutive snapshots  →  ~43 cloud points
    each in R^(3 * T * 16)
  - ripser on those ~43 points

If H1 does not survive here, topology is not the right instrument for a
1-layer, 16-embedding model and we close this chapter cleanly.

Usage:
  python experiment_sequence_topology.py          # full run
  python experiment_sequence_topology.py --quick  # 2 seeds only
  python experiment_sequence_topology.py --analyze
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "spark"))
sys.path.insert(0, str(SCRIPT_DIR))

from vybn import (
    TopoAgent,
    _persistence_pairs,
    _distance_matrix,
    CORPUS_PATH,
    RV, N_EMBD, N_LAYER, N_HEAD, HEAD_DIM, BLOCK_SIZE,
    _forward, _softmax, _rmsnorm, _linear,
)

# ── Config ────────────────────────────────────────────────────────────────
K               = 3    # texts per run
SNAP_EVERY      = 1    # capture every gradient step
STEPS_PER_TEXT  = 15   # gradient steps per text
LR              = 0.01
WINDOW          = 3    # sliding window width over consecutive snapshots
PROBE_SENTENCE  = "the topology of learning"
N_SEEDS         = 5
RESULTS_DIR     = SCRIPT_DIR / "experiment_results" / "sequence_topology"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from ripser import ripser as _ripser_fn
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


# ── Activation capture: per-token, NOT averaged ─────────────────────────

def capture_sequence(agent: TopoAgent, probe_text: str = PROBE_SENTENCE) -> np.ndarray:
    """Return (T, N_EMBD) array of per-token hidden states.

    Unlike the previous experiment this does NOT average across positions.
    Each row is the final-layer hidden state for one token position.
    """
    clean = agent._clean(probe_text)
    if len(clean) < 2:
        return np.zeros((1, N_EMBD))

    tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
    n = min(BLOCK_SIZE, len(tokens) - 1)

    hidden_states = []
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]

    for t in range(n):
        x = [agent.sd['wte'][tokens[t]][j] + agent.sd['wpe'][t][j]
             for j in range(N_EMBD)]
        for i in range(N_LAYER):
            xn  = _rmsnorm(x)
            q   = _linear(xn, agent.sd[f'layer{i}.attn_wq'])
            k_  = _linear(xn, agent.sd[f'layer{i}.attn_wk'])
            v_  = _linear(xn, agent.sd[f'layer{i}.attn_wv'])
            keys[i].append(k_)
            vals[i].append(v_)
            ho = []
            for h in range(N_HEAD):
                qs = q[h * HEAD_DIM:(h + 1) * HEAD_DIM]
                al = []
                for tt in range(len(keys[i])):
                    ks = keys[i][tt][h * HEAD_DIM:(h + 1) * HEAD_DIM]
                    al.append(
                        sum(qs[d] * ks[d] for d in range(HEAD_DIM))
                        * (HEAD_DIM ** -0.5)
                    )
                aw   = _softmax(al)
                hout = [RV(0.0)] * HEAD_DIM
                for tt in range(len(vals[i])):
                    vs = vals[i][tt][h * HEAD_DIM:(h + 1) * HEAD_DIM]
                    for d in range(HEAD_DIM):
                        hout[d] = hout[d] + aw[tt] * vs[d]
                ho.extend(hout)
            ao = _linear(ho, agent.sd[f'layer{i}.attn_wo'])
            x  = [x[j] + ao[j] for j in range(N_EMBD)]
            xn2 = _rmsnorm(x)
            h1  = _linear(xn2, agent.sd[f'layer{i}.mlp_fc1'])
            h1  = [hi * (RV(1.0) / (RV(1.0) + (hi * (-1)).exp())) for hi in h1]
            h2  = _linear(h1, agent.sd[f'layer{i}.mlp_fc2'])
            x   = [x[j] + h2[j] for j in range(N_EMBD)]
        hidden_states.append(np.array([xi.data for xi in x]))

    return np.array(hidden_states) if hidden_states else np.zeros((1, N_EMBD))


# ── Sliding-window point cloud ───────────────────────────────────────────

def build_point_cloud(snapshots: List[np.ndarray], window: int = WINDOW) -> np.ndarray:
    """Stack W consecutive (T, 16) snapshots into flat cloud points.

    Each point = W * T * 16 floats, preserving both the within-step
    token structure and the across-step temporal structure.

    Falls back to mean-per-snapshot if shapes are inconsistent.
    """
    if len(snapshots) < window:
        return np.array([s.mean(axis=0) for s in snapshots])

    shapes = [s.shape for s in snapshots]
    if len(set(shapes)) == 1:
        points = [
            np.concatenate([snapshots[i + j].ravel() for j in range(window)])
            for i in range(len(snapshots) - window + 1)
        ]
        return np.array(points)
    else:
        means = np.array([s.mean(axis=0) for s in snapshots])
        points = [
            means[i:i + window].ravel()
            for i in range(len(means) - window + 1)
        ]
        return np.array(points)


# ── Topology ─────────────────────────────────────────────────────────────

def compute_topology(points: np.ndarray) -> dict:
    n = len(points)
    if n < 3:
        return {
            "betti_0": n, "betti_1": 0,
            "total_persistence_h0": 0.0, "total_persistence_h1": 0.0,
            "persistence_entropy_h1": 0.0,
            "diagram_h1": [], "n_points": n,
            "space_dim": int(points.shape[-1]) if points.ndim >= 1 else 0,
            "method": "trivial",
        }

    if RIPSER_AVAILABLE:
        result  = _ripser_fn(points, maxdim=1)
        dgms    = result["dgms"]
        h0, h1  = dgms[0], dgms[1]

        h0_f = h0[h0[:, 1] < np.inf]
        h1_f = h1[h1[:, 1] < np.inf]
        tp_h0 = float(np.sum(h0_f[:, 1] - h0_f[:, 0])) if len(h0_f) else 0.0
        tp_h1 = float(np.sum(h1_f[:, 1] - h1_f[:, 0])) if len(h1_f) else 0.0

        all_b = list(h0[:, 0]) + list(h1[:, 0])
        if all_b:
            med = float(np.median(all_b))
            b0  = int(np.sum((h0[:, 0] <= med) & (h0[:, 1] > med)))
            b1  = int(np.sum((h1[:, 0] <= med) & (h1[:, 1] > med)))
        else:
            b0, b1 = n, 0

        ent = 0.0
        if len(h1_f) > 0:
            lt  = h1_f[:, 1] - h1_f[:, 0]
            tot = np.sum(lt)
            if tot > 1e-12:
                p   = lt / tot
                ent = float(-np.sum(p * np.log(p + 1e-15)))

        return {
            "betti_0": b0, "betti_1": b1,
            "total_persistence_h0": round(tp_h0, 6),
            "total_persistence_h1": round(tp_h1, 6),
            "persistence_entropy_h1": round(ent, 6),
            "diagram_h1": h1.tolist(),
            "n_points": n,
            "space_dim": int(points.shape[-1]),
            "method": "ripser",
        }
    else:
        D = _distance_matrix(points)
        pairs, (b0, b1, _) = _persistence_pairs(D)
        finite = [(b, d) for b, d in pairs if d != float("inf")]
        tp_h0  = sum(d - b for b, d in finite)
        return {
            "betti_0": b0, "betti_1": b1,
            "total_persistence_h0": round(tp_h0, 6),
            "total_persistence_h1": 0.0,
            "persistence_entropy_h1": 0.0,
            "diagram_h1": finite,
            "n_points": n,
            "space_dim": int(points.shape[-1]) if points.ndim >= 1 else 0,
            "method": "builtin",
        }


# ── Corpus helpers ────────────────────────────────────────────────────────

def load_corpus(min_words: int = 30) -> List[str]:
    passages = []
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
            "the creature breathes and measures its own distance from itself in the quiet between encounters",
            "curvature is born from incompleteness not from complexity alone what survives testing is true",
            "what survives testing is more honest than what merely sounds beautiful in the arrangement of words",
            "prediction loss going down means the model has memorised a pattern call it what it is",
            "the topology of weight space is a fingerprint of what was learned not merely how much was learned",
            "persistent homology integrates structure over a filtration giving us a summary independent of threshold",
            "attention heads rotate through conceptual space each head aligned to a different bivector plane",
        ]
    return passages


def synthetic_text(length_chars: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(length_chars))


# ── Single run ────────────────────────────────────────────────────────────

def run_condition(
    texts: List[str],
    seed: int,
    condition: str,
    run_idx: int,
) -> dict:
    np.random.seed(seed % 2**31)
    random.seed(seed)

    agent     = TopoAgent(config={"learn_steps": STEPS_PER_TEXT, "learn_lr": LR})
    snapshots: List[np.ndarray] = []
    loss_log  = []

    for text in texts:
        clean = agent._clean(text)
        if len(clean) < 2:
            continue
        tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
        n      = min(BLOCK_SIZE, len(tokens) - 1)

        for step in range(STEPS_PER_TEXT):
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
                g          = p.grad
                agent._m[j] = 0.85 * agent._m[j] + 0.15 * g
                agent._v[j] = 0.99 * agent._v[j] + 0.01 * g ** 2
                mh = agent._m[j] / (1 - 0.85 ** agent._step)
                vh = agent._v[j] / (1 - 0.99 ** agent._step)
                p.data -= LR * mh / (vh ** 0.5 + 1e-8)
            loss_log.append(round(loss.data, 6))

            if step % SNAP_EVERY == 0:
                snapshots.append(capture_sequence(agent))

    if not snapshots:
        snapshots.append(capture_sequence(agent))

    cloud = build_point_cloud(snapshots, window=WINDOW)
    topo  = compute_topology(cloud)
    final_loss, _ = agent.predict(" ".join(texts[:1]))

    return {
        "experiment":      "sequence_topology",
        "condition":       condition,
        "run_idx":         run_idx,
        "seed":            seed,
        "n_snapshots":     len(snapshots),
        "n_cloud_points":  len(cloud),
        "cloud_dim":       int(cloud.shape[-1]) if cloud.ndim == 2 else 0,
        "loss_trajectory": loss_log,
        "final_loss":      round(final_loss, 6),
        "topology":        topo,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }


# ── Main experiment ───────────────────────────────────────────────────────

def run_experiment(n_seeds: int = N_SEEDS, k: int = K, seed_base: int = 42) -> List[dict]:
    corpus  = load_corpus()
    avg_len = int(np.mean([len(t) for t in corpus[:20]])) if corpus else 200

    print(f"[Sequence-topology experiment — final run]")
    print(f"Corpus: {len(corpus)} passages  K={k}  SNAP_EVERY={SNAP_EVERY}  "
          f"STEPS={STEPS_PER_TEXT}  WINDOW={WINDOW}")
    expected_snaps = k * STEPS_PER_TEXT
    print(f"Expected snapshots/run: ~{expected_snaps}  "
          f"cloud points: ~{expected_snaps - WINDOW + 1}  "
          f"cloud dim: {WINDOW * min(BLOCK_SIZE, 26) * N_EMBD} (approx)")
    print(f"Ripser: {'available' if RIPSER_AVAILABLE else 'builtin fallback'}")
    print()

    all_results = []
    rng = random.Random(seed_base)

    for seed_offset in range(n_seeds):
        seed = seed_base + seed_offset

        # real
        texts = rng.sample(corpus, min(k, len(corpus)))
        r = run_condition(texts, seed, "real", seed_offset)
        all_results.append(r)
        t = r["topology"]
        print(
            f"  seed {seed}  real      | "
            f"pts={r['n_cloud_points']} dim={r['cloud_dim']}  "
            f"b1={t['betti_1']}  tp_h1={t['total_persistence_h1']:.6f}  "
            f"loss={r['final_loss']:.3f}"
        )
        (RESULTS_DIR / f"real_{seed_offset:03d}.json").write_text(
            json.dumps(r, indent=2, default=str)
        )

        # synthetic
        syn_texts = [
            synthetic_text(avg_len, seed=seed * 1000 + j) for j in range(k)
        ]
        r2 = run_condition(syn_texts, seed, "synthetic", seed_offset)
        all_results.append(r2)
        t2 = r2["topology"]
        print(
            f"  seed {seed}  synthetic | "
            f"pts={r2['n_cloud_points']} dim={r2['cloud_dim']}  "
            f"b1={t2['betti_1']}  tp_h1={t2['total_persistence_h1']:.6f}  "
            f"loss={r2['final_loss']:.3f}"
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
    print("SEQUENCE-TOPOLOGY EXPERIMENT — RESULTS SUMMARY")
    print("=" * 70)

    for cond in ("real", "synthetic"):
        runs = by_cond.get(cond, [])
        if not runs:
            continue
        tp = [r["topology"]["total_persistence_h1"] for r in runs]
        b1 = [r["topology"]["betti_1"] for r in runs]
        print(
            f"  {cond:10s}: n={len(runs)}  "
            f"tp_h1={np.mean(tp):.6f}±{np.std(tp):.6f}  "
            f"b1_mean={np.mean(b1):.2f}  "
            f"b1_nonzero={sum(b > 0 for b in b1)}/{len(b1)}"
        )

    real_tp = [r["topology"]["total_persistence_h1"]
               for r in by_cond.get("real", [])]
    syn_tp  = [r["topology"]["total_persistence_h1"]
               for r in by_cond.get("synthetic", [])]

    print()
    if real_tp and syn_tp:
        diff    = np.mean(real_tp) - np.mean(syn_tp)
        any_h1  = any(v > 0 for v in real_tp + syn_tp)
        real_h1 = any(v > 0 for v in real_tp)
        syn_h1  = any(v > 0 for v in syn_tp)
        print(f"  Real − synthetic mean tp_h1: {diff:+.6f}")
        print()
        if not any_h1:
            print("  VERDICT: zero H1 across all runs.")
            print("  Topology is not the right instrument for this model at this scale.")
            print("  The geometry story (curvature, rotor, holonomy) stands.")
            print("  Closing the topology chapter.")
        elif real_h1 and not syn_h1:
            print("  VERDICT: H1 present in real, absent in synthetic.")
            print("  Text selection shapes activation-space topology. Signal confirmed.")
        elif real_h1 and syn_h1:
            print("  VERDICT: H1 appears in both conditions — likely noise at this scale.")
            print("  Inspect diagram_h1 lifetimes; consider this chapter inconclusive.")
        else:
            print("  VERDICT: ambiguous. Inspect per-seed diagrams.")


def main():
    parser = argparse.ArgumentParser(description="Final sequence-topology experiment")
    parser.add_argument("--quick",   action="store_true",
                        help="2 seeds only (fast sanity check)")
    parser.add_argument("--analyze", action="store_true",
                        help="Load saved results only")
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
                {
                    **{k: v for k, v in r.items()
                       if k not in ("loss_trajectory",)},
                    "diagram_h1_count": len(r["topology"].get("diagram_h1", [])),
                }
                for r in results
            ],
            indent=2, default=str,
        )
    )
    print(f"Results → {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
