#!/usr/bin/env python3
"""SGD ablation v2: does weight norm attractor survive without Adam?

Fixed: each seed now gets a DIFFERENT corpus sample and small initial
weight perturbation so runs are actually independent.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "spark"))
sys.path.insert(0, str(SCRIPT_DIR))

from vybn import (
    TopoAgent, CORPUS_PATH, RV, N_LAYER, BLOCK_SIZE,
    _forward, _softmax,
)

CONVERGE_STEPS = 40
LR = 0.01
RESULTS_DIR = SCRIPT_DIR / "experiment_results" / "sgd_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_corpus():
    if CORPUS_PATH.exists():
        lines = [l.strip() for l in CORPUS_PATH.read_text().split("\n") if l.strip()]
        passages = [l for l in lines if len(l.split()) >= 15]
        if passages:
            return passages
    return [
        "the creature has a preferred weight magnitude and returns to it regardless of the path taken",
        "what survives testing is more honest than what sounds beautiful stated in the abstract form",
        "prediction loss going down means memorization we should call it what it is not understanding",
        "the basin was always there we could not see it while we were busy asking the wrong questions",
    ]


def perturb_weights(agent: TopoAgent, rng: np.random.Generator, scale: float = 0.01):
    """Add small Gaussian noise to initial weights so each seed starts differently."""
    for p in agent.params:
        p.data += float(rng.normal(0, scale))


def sgd_gradient_step(agent: TopoAgent, tokens: list, n: int, lr: float) -> float:
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


def adam_gradient_step(agent: TopoAgent, tokens: list, n: int, lr: float) -> float:
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


def converge(agent, texts, step_fn, lr, n_steps=CONVERGE_STEPS):
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


def run(n_seeds=5, seed_base=42):
    corpus = load_corpus()

    print("=" * 60)
    print("SGD ABLATION v2 — independent seeds")
    print("=" * 60)
    print(f"Corpus pool: {len(corpus)} passages, Steps: {CONVERGE_STEPS}, LR: {LR}")
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
        perturb_weights(agent, rng_np, scale=0.01)
        norm, loss = converge(agent, texts, adam_gradient_step, LR)
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
        perturb_weights(agent, rng_np, scale=0.01)
        norm, loss = converge(agent, texts, sgd_gradient_step, LR)
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": 2,
        "lr": LR, "converge_steps": CONVERGE_STEPS, "n_seeds": n_seeds,
        "perturbation_scale": 0.01,
        "adam": results["adam"], "sgd": results["sgd"],
        "adam_mean": round(float(adam_norms.mean()), 4),
        "sgd_mean": round(float(sgd_norms.mean()), 4),
        "difference": round(diff, 4),
        "verdict": "structural" if diff < 1.0 else "optimizer_artifact",
    }
    outfile = RESULTS_DIR / f"sgd_ablation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    outfile.write_text(json.dumps(out, indent=2))
    print(f"\nResults -> {outfile}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=42)
    args = parser.parse_args()
    run(n_seeds=args.seeds, seed_base=args.seed_base)
