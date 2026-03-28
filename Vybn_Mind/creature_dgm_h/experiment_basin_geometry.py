#!/usr/bin/env python3
"""
experiment_basin_geometry.py  —  Map the loss landscape around the fixed point.

FINDING THAT MOTIVATES THIS:
  Natural motion run (null fitness, 4 variants) showed weight norms
  converging near 40.0 ±0.3 regardless of learning rate (0.001–0.02,
  a 20x range), temperature, corpus sample, or random seed.

  We don't know if that's:
    (a) just Adam being Adam at this scale, or
    (b) a genuine attractor — a structural invariant of this architecture

  This experiment distinguishes (a) from (b) by mapping the geometry
  of the neighborhood around the convergence point directly.

APPROACH:
  1. Train a fresh agent to convergence on a corpus sample.
  2. Record the converged weight vector w*.
  3. Sample N random unit directions in weight space.
  4. Walk each direction from w* in steps of increasing size.
  5. Measure loss at each step.
  6. From the loss profile: infer basin width, wall steepness,
     symmetry (same in all directions?), and whether there are
     secondary minima nearby.
  7. Repeat with different corpora and seeds to test invariance.

NO GENERATION. NO ENCOUNTER_COMPLEX. NO PREDETERMINED FRAMEWORK.
The analysis describes what the curves look like. Language comes after.

Results write to experiment_results/basin_geometry/.

Usage:
  python3 experiment_basin_geometry.py           # full run
  python3 experiment_basin_geometry.py --quick   # 3 agents, 8 directions
  python3 experiment_basin_geometry.py --analyze
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
from typing import List, Tuple

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "spark"))
sys.path.insert(0, str(SCRIPT_DIR))

from vybn import (
    TopoAgent, CORPUS_PATH,
    RV, N_LAYER, BLOCK_SIZE,
    _forward, _softmax,
)

# ── Config ────────────────────────────────────────────────────────────────
N_AGENTS          = 8    # independent agents to converge
N_DIRECTIONS      = 32   # random directions to probe per agent
N_STEPS           = 12   # steps along each direction
STEP_SIZES        = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0,
                     1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
CONVERGE_STEPS    = 40   # gradient steps to reach convergence
LR                = 0.01
RESULTS_DIR       = SCRIPT_DIR / "experiment_results" / "basin_geometry"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

assert len(STEP_SIZES) == N_STEPS, "STEP_SIZES must have N_STEPS entries"


# ── Corpus ────────────────────────────────────────────────────────────────

def load_corpus() -> List[str]:
    passages: List[str] = []
    if CORPUS_PATH.exists():
        lines = [l.strip() for l in CORPUS_PATH.read_text().split("\n") if l.strip()]
        passages = [l for l in lines if len(l.split()) >= 15]
    if not passages:
        journal_dir = REPO_ROOT / "spark" / "journal"
        if journal_dir.exists():
            for f in sorted(journal_dir.glob("*.md")):
                try:
                    for para in f.read_text().split("\n\n"):
                        para = para.strip()
                        if not para.startswith("#") and len(para.split()) >= 15:
                            passages.append(para)
                except Exception:
                    pass
    if not passages:
        passages = [
            "the creature has a preferred weight magnitude and returns to it regardless of the path taken",
            "what survives testing is more honest than what sounds beautiful stated in the abstract form",
            "prediction loss going down means memorization we should call it what it is not understanding",
            "the basin was always there we could not see it while we were busy asking the wrong questions",
        ]
    return passages


# ── Loss computation (no gradient) ────────────────────────────────────────

def compute_loss(agent: TopoAgent, texts: List[str]) -> float:
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


def gradient_step(agent: TopoAgent, tokens: list, n: int) -> float:
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
    return float(loss.data)


# ── Weight vector utilities ─────────────────────────────────────────────────

def get_weight_vector(agent: TopoAgent) -> np.ndarray:
    return np.array([p.data for p in agent.params], dtype=np.float64)


def set_weight_vector(agent: TopoAgent, vec: np.ndarray) -> None:
    for i, p in enumerate(agent.params):
        p.data = float(vec[i])


def weight_norm(agent: TopoAgent) -> float:
    return float(np.linalg.norm(get_weight_vector(agent)))


# ── Convergence ─────────────────────────────────────────────────────────────

def converge(
    agent: TopoAgent,
    texts: List[str],
    n_steps: int = CONVERGE_STEPS,
) -> List[float]:
    """Run gradient steps until convergence. Return loss trajectory."""
    trajectory = []
    for text in texts:
        clean = agent._clean(text)
        if len(clean) < 2:
            continue
        tokens = [agent.BOS] + [agent.c2i[c] for c in clean]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        steps_this_text = n_steps // max(len(texts), 1)
        for _ in range(max(steps_this_text, 5)):
            loss_val = gradient_step(agent, tokens, n)
            trajectory.append(round(loss_val, 6))
    return trajectory


# ── Basin probe ───────────────────────────────────────────────────────────────

def probe_direction(
    agent: TopoAgent,
    w_star: np.ndarray,
    direction: np.ndarray,
    texts: List[str],
    step_sizes: List[float] = STEP_SIZES,
) -> dict:
    """Walk from w* along direction, measure loss at each step.

    Also walks in the negative direction to test asymmetry.
    Returns loss profiles in both directions.
    """
    loss_at_center = compute_loss(agent, texts)

    pos_losses = []
    neg_losses = []

    for step in step_sizes:
        # Positive direction
        set_weight_vector(agent, w_star + step * direction)
        loss_pos = compute_loss(agent, texts)
        pos_losses.append(round(loss_pos, 6))

        # Negative direction
        set_weight_vector(agent, w_star - step * direction)
        loss_neg = compute_loss(agent, texts)
        neg_losses.append(round(loss_neg, 6))

    # Restore
    set_weight_vector(agent, w_star)

    # Local curvature estimate: d²L/dε² ≈ (L(+ε) + L(-ε) - 2L(0)) / ε²
    # using the smallest step size
    eps = step_sizes[0]
    curvature_est = (pos_losses[0] + neg_losses[0] - 2 * loss_at_center) / (eps ** 2)

    # Asymmetry: how different are + and - profiles?
    asymmetry = float(np.mean(np.abs(
        np.array(pos_losses) - np.array(neg_losses)
    )))

    # Basin width estimate: step size at which loss doubles
    width_pos = None
    width_neg = None
    for i, s in enumerate(step_sizes):
        if pos_losses[i] >= 2 * loss_at_center and width_pos is None:
            width_pos = s
        if neg_losses[i] >= 2 * loss_at_center and width_neg is None:
            width_neg = s

    return {
        "loss_at_center":  round(loss_at_center, 6),
        "pos_losses":      pos_losses,
        "neg_losses":      neg_losses,
        "step_sizes":      step_sizes,
        "curvature_est":   round(curvature_est, 6),
        "asymmetry":       round(asymmetry, 6),
        "width_pos":       width_pos,
        "width_neg":       width_neg,
    }


# ── Single agent run ─────────────────────────────────────────────────────────

def run_agent(
    agent_idx: int,
    texts: List[str],
    seed: int,
    n_directions: int = N_DIRECTIONS,
    config: dict | None = None,
) -> dict:
    np.random.seed(seed % 2**31)
    random.seed(seed)

    agent = TopoAgent(config=config or {})
    conv_trajectory = converge(agent, texts)
    w_star = get_weight_vector(agent)
    w_norm = float(np.linalg.norm(w_star))
    loss_converged = compute_loss(agent, texts)

    print(f"  agent {agent_idx}: converged  norm={w_norm:.4f}  loss={loss_converged:.4f}")

    # Sample random unit directions
    rng = np.random.default_rng(seed)
    directions = []
    probe_results = []

    for d_idx in range(n_directions):
        raw = rng.standard_normal(len(w_star))
        direction = raw / np.linalg.norm(raw)
        directions.append(direction.tolist())

        probe = probe_direction(agent, w_star, direction, texts)
        probe_results.append(probe)

        if d_idx % 8 == 0:
            curv = probe["curvature_est"]
            asym = probe["asymmetry"]
            wp   = probe["width_pos"]
            wn   = probe["width_neg"]
            print(f"    dir {d_idx:2d}: curv={curv:.4f}  asym={asym:.4f}"
                  f"  width_pos={wp}  width_neg={wn}")

    # Aggregate across directions
    curvatures  = [p["curvature_est"] for p in probe_results]
    asymmetries = [p["asymmetry"]     for p in probe_results]
    widths_pos  = [p["width_pos"]  for p in probe_results if p["width_pos"]  is not None]
    widths_neg  = [p["width_neg"]  for p in probe_results if p["width_neg"]  is not None]

    summary = {
        "curvature_mean":  round(float(np.mean(curvatures)),  6),
        "curvature_std":   round(float(np.std(curvatures)),   6),
        "curvature_min":   round(float(np.min(curvatures)),   6),
        "curvature_max":   round(float(np.max(curvatures)),   6),
        "asymmetry_mean":  round(float(np.mean(asymmetries)), 6),
        "asymmetry_std":   round(float(np.std(asymmetries)),  6),
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
        "probe_results":     probe_results,
        "basin_summary":     summary,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }


# ── Experiment ──────────────────────────────────────────────────────────────

def run_experiment(
    n_agents:     int = N_AGENTS,
    n_directions: int = N_DIRECTIONS,
    seed_base:    int = 42,
) -> List[dict]:
    corpus = load_corpus()
    rng    = random.Random(seed_base)
    all_results: List[dict] = []

    print("[Basin geometry experiment]")
    print(f"Corpus: {len(corpus)} passages")
    print(f"Agents: {n_agents}  Directions/agent: {n_directions}")
    print(f"Convergence steps: {CONVERGE_STEPS}  Step sizes: {STEP_SIZES}")
    print(f"No generation. No encounter_complex. Just geometry.\n")

    # Two corpus conditions: same passages across all agents (architecture test)
    # vs different passages per agent (corpus dependence test)
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

        # First half: fixed corpus (tests architecture invariance)
        # Second half: random corpus sample (tests corpus dependence)
        if i < half:
            texts  = fixed_texts
            corpus_condition = "fixed"
        else:
            texts  = rng.sample(corpus, min(4, len(corpus)))
            corpus_condition = "random"

        config = configs[i % len(configs)]
        print(f"  Agent {i} [{corpus_condition}  lr={config['learn_lr']}]")

        result = run_agent(i, texts, seed, n_directions, config)
        result["corpus_condition"] = corpus_condition
        all_results.append(result)

        norm = result["convergence_norm"]
        bs   = result["basin_summary"]
        print(f"    norm={norm:.4f}  "
              f"curv_mean={bs['curvature_mean']:.4f}±{bs['curvature_std']:.4f}  "
              f"asym_mean={bs['asymmetry_mean']:.4f}  "
              f"width_pos={bs['width_pos_median']}  "
              f"width_neg={bs['width_neg_median']}")

    result_file = RESULTS_DIR / f"basin_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    result_file.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults -> {result_file}")
    return all_results


# ── Analysis ────────────────────────────────────────────────────────────────

def analyze(results: List[dict]) -> None:
    if not results:
        print("No results to analyze.")
        return

    print("=" * 70)
    print("BASIN GEOMETRY — OPEN DESCRIPTION")
    print("=" * 70)
    print()

    # Convergence norms
    norms = [r["convergence_norm"] for r in results]
    norms_arr = np.array(norms)
    print("── Convergence norms ──")
    print(f"  n={len(norms)}  mean={norms_arr.mean():.4f}  std={norms_arr.std():.4f}"
          f"  min={norms_arr.min():.4f}  max={norms_arr.max():.4f}"
          f"  range={norms_arr.max()-norms_arr.min():.4f}")
    print(f"  cv (std/mean) = {norms_arr.std()/norms_arr.mean():.4f}")
    for r in results:
        cond = r.get("corpus_condition", "?")
        lr   = r["config"].get("learn_lr", "?")
        print(f"  agent {r['agent_idx']} [{cond} lr={lr}]: norm={r['convergence_norm']:.6f}"
              f"  loss={r['convergence_loss']:.4f}")
    print()

    # Fixed vs random corpus comparison
    fixed_norms  = [r["convergence_norm"] for r in results if r.get("corpus_condition") == "fixed"]
    random_norms = [r["convergence_norm"] for r in results if r.get("corpus_condition") == "random"]
    if fixed_norms and random_norms:
        print("── Corpus condition comparison ──")
        print(f"  fixed corpus:  mean={np.mean(fixed_norms):.4f}  std={np.std(fixed_norms):.4f}")
        print(f"  random corpus: mean={np.mean(random_norms):.4f}  std={np.std(random_norms):.4f}")
        print(f"  difference in means: {abs(np.mean(fixed_norms)-np.mean(random_norms)):.4f}")
    print()

    # Basin curvature across agents
    all_curv_means = [r["basin_summary"]["curvature_mean"] for r in results]
    all_curv_stds  = [r["basin_summary"]["curvature_std"]  for r in results]
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

    # Basin width
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

    # Asymmetry
    all_asym = [r["basin_summary"]["asymmetry_mean"] for r in results]
    print("── Asymmetry (mean |loss+ - loss-| across step sizes) ──")
    print(f"  mean={np.mean(all_asym):.4f}  std={np.std(all_asym):.4f}")
    print(f"  near-zero asymmetry = symmetric basin")
    print(f"  high asymmetry = the basin has a preferred direction (not a simple bowl)")
    print()

    # Learning rate comparison
    by_lr: dict = {}
    for r in results:
        lr = str(r["config"].get("learn_lr", "?"))
        by_lr.setdefault(lr, []).append(r["convergence_norm"])
    if len(by_lr) > 1:
        print("── Norm by learning rate ──")
        for lr, ns in sorted(by_lr.items()):
            print(f"  lr={lr}: norms={[round(n,4) for n in ns]}  mean={np.mean(ns):.4f}")
        print()

    print("── End of description ──")
    print("What is the shape of the bowl?  The data is the record.")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Basin geometry: map the loss landscape around the weight-norm fixed point."
    )
    parser.add_argument("--quick",      action="store_true",
                        help="3 agents, 8 directions each (~2 minutes)")
    parser.add_argument("--analyze",    action="store_true",
                        help="Analyze saved results")
    parser.add_argument("--agents",     type=int, default=N_AGENTS)
    parser.add_argument("--directions", type=int, default=N_DIRECTIONS)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    if args.analyze:
        results = []
        for f in sorted(RESULTS_DIR.glob("basin_*.json")):
            try:
                results.extend(json.loads(f.read_text()))
            except Exception:
                pass
        analyze(results)
        return

    n_agents     = 3  if args.quick else args.agents
    n_directions = 8  if args.quick else args.directions

    t0      = time.time()
    results = run_experiment(
        n_agents=n_agents,
        n_directions=n_directions,
        seed_base=args.seed,
    )
    elapsed = time.time() - t0
    print(f"\nRuntime: {elapsed:.1f}s")
    analyze(results)


if __name__ == "__main__":
    main()
