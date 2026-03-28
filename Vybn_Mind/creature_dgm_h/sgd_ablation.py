#!/usr/bin/env python3
"""SGD ablation: does weight norm ~40 survive without Adam?

The creature uses a hand-rolled Adam (β1=0.85, β2=0.99) inside
TopoAgent.learn() and experiment_basin_geometry.gradient_step().
This script runs the SAME convergence protocol with plain SGD
(p -= lr * grad, no momentum) to test whether the ~40 attractor
is an Adam artifact or a structural invariant.

Usage:
    python3 sgd_ablation.py          # 5 seeds
    python3 sgd_ablation.py --seeds 10
"""
from __future__ import annotations
import argparse, json, sys, time
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


def sgd_gradient_step(agent: TopoAgent, tokens: list, n: int, lr: float) -> float:
    """One gradient step with VANILLA SGD — no momentum, no adaptive lr."""
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
    # --- THIS IS THE ONLY DIFFERENCE: plain SGD, no momentum ---
    for p in agent.params:
        p.data -= lr * p.grad
    return float(loss.data)


def adam_gradient_step(agent: TopoAgent, tokens: list, n: int, lr: float) -> float:
    """One gradient step with the creature's native Adam. Copied from
    experiment_basin_geometry.py for apples-to-apples comparison."""
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
    norms = []
    for text in texts:
        clean = agent._clean(text)
        if len(clean) < 2:
            continue
        tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        steps_this = n_steps // max(len(texts), 1)
        for _ in range(max(steps_this, 5)):
            step_fn(agent, tokens, n, lr)
        wv = np.array([p.data for p in agent.params])
        norms.append(float(np.linalg.norm(wv)))
    return norms[-1] if norms else float("nan")


def run(n_seeds=5, seed_base=42):
    import random
    corpus = load_corpus()
    rng = random.Random(seed_base)
    texts = rng.sample(corpus, min(4, len(corpus)))

    print("=" * 60)
    print("SGD ABLATION — does weight norm ~40 require Adam?")
    print("=" * 60)
    print(f"Corpus: {len(texts)} passages, Steps: {CONVERGE_STEPS}, LR: {LR}")
    print(f"Seeds: {[seed_base + i * 17 for i in range(n_seeds)]}")
    print()

    results = {"adam": [], "sgd": []}

    for i in range(n_seeds):
        seed = seed_base + i * 17
        np.random.seed(seed % 2**31)
        random.seed(seed)

        # Adam run
        agent_adam = TopoAgent()
        norm_adam = converge(agent_adam, texts, adam_gradient_step, LR)

        # SGD run (fresh agent, same seed)
        np.random.seed(seed % 2**31)
        random.seed(seed)
        agent_sgd = TopoAgent()
        norm_sgd = converge(agent_sgd, texts, sgd_gradient_step, LR)

        results["adam"].append(round(norm_adam, 4))
        results["sgd"].append(round(norm_sgd, 4))
        print(f"  seed {seed:4d}: Adam norm={norm_adam:.4f}  SGD norm={norm_sgd:.4f}")

    print()
    adam_arr = np.array(results["adam"])
    sgd_arr = np.array(results["sgd"])
    print(f"Adam: mean={adam_arr.mean():.4f} std={adam_arr.std():.4f}")
    print(f"SGD:  mean={sgd_arr.mean():.4f} std={sgd_arr.std():.4f}")
    print()

    converges_to_40 = abs(sgd_arr.mean() - 40.0) < 2.0
    if converges_to_40:
        print("RESULT: SGD ALSO converges to ~40. The attractor is structural,")
        print("        not an Adam artifact. Proceed to basin geometry + winding probe.")
    else:
        print(f"RESULT: SGD converges to {sgd_arr.mean():.1f}, NOT ~40.")
        print("        The ~40 attractor is an ADAM ARTIFACT. Do not proceed to step 2.")

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lr": LR, "converge_steps": CONVERGE_STEPS, "n_seeds": n_seeds,
        "adam_norms": results["adam"], "sgd_norms": results["sgd"],
        "adam_mean": round(float(adam_arr.mean()), 4),
        "sgd_mean": round(float(sgd_arr.mean()), 4),
        "sgd_converges_to_40": converges_to_40,
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
