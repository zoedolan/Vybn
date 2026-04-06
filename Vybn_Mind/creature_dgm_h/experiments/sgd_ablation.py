#!/usr/bin/env python3
"""SGD ablation: does weight norm ~40 survive without Adam?

Same architecture, same corpus, same lr, same steps.
Replace Adam update with plain SGD (no momentum).
"""
import json, math, sys, random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "spark"))
sys.path.insert(0, str(SCRIPT_DIR))

from vybn import (
    TopoAgent, CORPUS_PATH, CHECKPOINT_PATH,
    RV, N_LAYER, BLOCK_SIZE,
    _forward, _softmax,
)
import numpy as np

CONVERGE_STEPS = 40
LR = 0.01
SEEDS = [42, 137, 2025, 7, 99]

def load_corpus():
    raw = Path(CORPUS_PATH).read_text(encoding="utf-8", errors="replace")
    return raw[:2000]

def weight_norm(params):
    return math.sqrt(sum(p.data**2 for p in params))

def train_sgd(seed, corpus, steps=CONVERGE_STEPS, lr=LR):
    """Train with plain SGD — no momentum, no adaptive rates."""
    random.seed(seed)
    np.random.seed(seed)
    
    agent = TopoAgent()
    clean = agent._clean(corpus)
    if len(clean) < 2:
        return None
    tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
    n = min(BLOCK_SIZE, len(tokens) - 1)
    
    norms = []
    losses = []
    for step in range(steps):
        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]
        loss = RV(0.0)
        for t in range(n):
            logits, keys, vals = _forward(tokens[t], t, keys, vals, agent.sd, None)
            probs = _softmax(logits)
            loss = loss + (probs[tokens[t+1]].log()) * (-1.0 / n)
        
        for p in agent.params:
            p.grad = 0.0
        loss.backward()
        
        # Plain SGD: w = w - lr * grad
        for p in agent.params:
            p.data -= lr * p.grad
        
        norm = weight_norm(agent.params)
        norms.append(round(norm, 4))
        losses.append(round(loss.data, 6))
    
    return {"seed": seed, "final_norm": norms[-1], "norms": norms, "losses": losses}

def train_adam(seed, corpus, steps=CONVERGE_STEPS, lr=LR):
    """Train with the creature's built-in Adam for comparison."""
    random.seed(seed)
    np.random.seed(seed)
    
    agent = TopoAgent()
    agent.learn(corpus, steps=steps, lr=lr)
    norm = weight_norm(agent.params)
    return {"seed": seed, "final_norm": round(norm, 4)}

def main():
    corpus = load_corpus()
    print(f"Corpus length: {len(corpus)} chars")
    print(f"Steps: {CONVERGE_STEPS}, LR: {LR}, Seeds: {SEEDS}\n")
    
    print("=" * 60)
    print("ADAM (creature default)")
    print("=" * 60)
    adam_norms = []
    for seed in SEEDS:
        r = train_adam(seed, corpus)
        adam_norms.append(r["final_norm"])
        print(f"  seed={seed:>5}  final_norm={r['final_norm']:.4f}")
    print(f"  mean={np.mean(adam_norms):.4f}  std={np.std(adam_norms):.4f}\n")
    
    print("=" * 60)
    print("SGD (no momentum, no adaptive)")
    print("=" * 60)
    sgd_norms = []
    for seed in SEEDS:
        r = train_sgd(seed, corpus)
        sgd_norms.append(r["final_norm"])
        print(f"  seed={seed:>5}  final_norm={r['final_norm']:.4f}  loss_final={r['losses'][-1]:.4f}")
    print(f"  mean={np.mean(sgd_norms):.4f}  std={np.std(sgd_norms):.4f}\n")
    
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    adam_mean = np.mean(adam_norms)
    sgd_mean = np.mean(sgd_norms)
    delta = abs(adam_mean - sgd_mean)
    print(f"  Adam mean norm: {adam_mean:.4f}")
    print(f"  SGD  mean norm: {sgd_mean:.4f}")
    print(f"  Delta:          {delta:.4f}")
    if delta < 1.0:
        print(f"  SGD converges to same neighborhood => NOT Adam-specific")
        print(f"  Proceed to basin geometry + winding probe")
    else:
        print(f"  SGD diverges from Adam norm => the ~40 is Adam's doing")
        print(f"  STOP. Do not proceed to step two.")

if __name__ == "__main__":
    main()
