#!/usr/bin/env python3
"""
experiment_sequence_topology.py  —  Natural motion recorder.

PRIOR CHAPTER (closed):
  Six experiments measured internal geometry under real vs synthetic conditioning:
    1. Weight-space H1:             zero
    2. PCA H1:                      zero
    3. Activation H1:               zero
    4. Sequence H1:                 zero
    5. Curvature trajectory:        flat
    6. Weight SVD geometry:         indistinguishable
  Verdict: the creature's internal geometry is invariant under meaning at 4K
  parameters.  We were measuring the map.  The territory is elsewhere.

THIS CHAPTER:
  Instead of asking the creature to demonstrate something we expect, we remove
  the expectation.  Fitness is held constant (all variants score equally).
  Selection pressure selects for nothing.  We record what the creature does
  anyway — what configurations persist, what trajectories emerge, what the
  system's natural motion looks like when we are not grading it.

  The analysis script has no predetermined categories.  It describes whatever
  it finds.  Language for the results comes after, not before.

  Results write to experiment_results/natural_motion/.

Usage:
  python experiment_sequence_topology.py          # full run (5 generations)
  python experiment_sequence_topology.py --quick  # 2 generations
  python experiment_sequence_topology.py --analyze
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "spark"))
sys.path.insert(0, str(SCRIPT_DIR))

from vybn import (
    TopoAgent, Organism, encounter_complex,
    _load_prose_corpus, CORPUS_PATH,
    RV, N_LAYER, BLOCK_SIZE,
    _forward, _softmax,
)

# ── Config ────────────────────────────────────────────────────────────────
N_GENERATIONS     = 5
VARIANTS_PER_GEN  = 3
STEPS_PER_TEXT    = 8
TEXTS_PER_VARIANT = 2
LR                = 0.01
RESULTS_DIR       = SCRIPT_DIR / "experiment_results" / "natural_motion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Null fitness: constant for all variants ───────────────────────────────

def null_fitness() -> dict:
    """Returns identical fitness for every variant.
    Selection pressure selects for nothing.
    The creature evolves, but not toward anything we defined."""
    return {
        "fitness": 0.5,
        "curvature": 0.0,
        "betti": (0, 0, 0),
        "topological_richness": 0.0,
        "structural_growth": 0.0,
        "weight_topo": 0.0,
        "note": "null_fitness — selection pressure removed",
    }


# ── Observation: what does the creature actually produce? ─────────────────

def observe(agent: TopoAgent, prompt: str = "", n_samples: int = 5) -> List[dict]:
    """Generate n_samples outputs and record everything observable about each.

    We do not decide in advance what is interesting.  We record:
    - the raw generated text
    - its encounter_complex (rotor, curvature, betti, persistence)
    - the loss the agent assigns to its own output
    - the surprise contour (per-character)

    The analysis step decides what matters.
    """
    observations = []
    for i in range(n_samples):
        temperature = 0.7 + i * 0.15
        text = agent.generate(prompt=prompt, max_tokens=40, temperature=temperature)
        if not text or len(text.split()) < 3:
            continue

        loss, contour = agent.predict(text)
        cx = encounter_complex(text)

        observations.append({
            "sample_idx": i,
            "temperature": round(temperature, 3),
            "text": text,
            "loss": round(loss, 6),
            "curvature": round(cx.curvature, 8),
            "angle_deg": round(math.degrees(cx.angle), 4),
            "betti": list(cx.betti),
            "n_persistent_features": cx.n_persistent_features,
            "max_persistence": round(cx.max_persistence, 6),
            "bv_norm": round(cx.rotor.bv_norm, 6),
            "bv_dir": [round(x, 6) for x in cx.rotor.bv_dir.tolist()],
            "surprise_mean": round(
                sum(r["surprise"] for r in contour) / len(contour), 6
            ) if contour else 0.0,
            "surprise_max": round(
                max(r["surprise"] for r in contour), 6
            ) if contour else 0.0,
            "surprise_contour": contour[:8],
        })

    return observations


def observe_weight_snapshot(agent: TopoAgent) -> dict:
    """Record a compact snapshot of weight magnitudes.
    Not for topology — just to see if configurations drift, cluster,
    or stabilize over generations without fitness pressure."""
    norms = {}
    for key, mat in agent.sd.items():
        arr = np.array([[p.data for p in row] for row in mat], dtype=np.float64)
        norms[key] = round(float(np.linalg.norm(arr)), 8)
    total = round(float(sum(norms.values())), 6)
    return {"key_norms": norms, "total_norm": total}


# ── Corpus ────────────────────────────────────────────────────────────────

def load_corpus() -> List[str]:
    passages: List[str] = []
    if CORPUS_PATH.exists():
        lines = [l.strip() for l in CORPUS_PATH.read_text().split("\n") if l.strip()]
        passages = [l for l in lines if len(l.split()) >= 20]
    if not passages:
        passages = _load_prose_corpus(min_words=20, max_passages=50)
    if not passages:
        passages = [
            "the creature breathes and measures its own distance from itself",
            "curvature is born from incompleteness not from complexity alone",
            "what survives testing is more honest than what sounds beautiful",
            "prediction loss going down means memorization call it what it is",
        ]
    return passages


# ── Single gradient step ──────────────────────────────────────────────────

def _gradient_step(agent: TopoAgent, tokens: list, n: int) -> float:
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


# ── Single variant run ────────────────────────────────────────────────────

def run_variant(
    texts: List[str],
    config: dict,
    generation: int,
    variant_idx: int,
    seed: int,
) -> dict:
    """Run one variant: learn from texts, observe outputs, record everything.
    Fitness is null — we do not score this variant by any criterion we set."""

    np.random.seed(seed % 2 ** 31)
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
        for _ in range(STEPS_PER_TEXT):
            loss_val = _gradient_step(agent, tokens, n)
            traj.append(round(loss_val, 6))
        loss_trajectories.append({"text_preview": text[:40], "trajectory": traj})

    # Observe: what does this creature produce unprompted?
    observations_cold = observe(agent, prompt="", n_samples=3)

    # Observe: what does it produce seeded with its own training text?
    seed_prompt = texts[0][:8] if texts else ""
    observations_seeded = observe(agent, prompt=seed_prompt, n_samples=3)

    weight_snap = observe_weight_snapshot(agent)

    encounter_records = []
    for text in texts:
        cx = encounter_complex(text)
        encounter_records.append({
            "text_preview": text[:40],
            "curvature": round(cx.curvature, 8),
            "betti": list(cx.betti),
            "angle_deg": round(math.degrees(cx.angle), 4),
            "n_persistent_features": cx.n_persistent_features,
        })

    return {
        "experiment": "natural_motion",
        "generation": generation,
        "variant_idx": variant_idx,
        "seed": seed,
        "config": config,
        "fitness": null_fitness(),
        "loss_trajectories": loss_trajectories,
        "observations_cold": observations_cold,
        "observations_seeded": observations_seeded,
        "weight_snapshot": weight_snap,
        "encounter_records": encounter_records,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Generation loop ───────────────────────────────────────────────────────

def run_experiment(
    n_generations: int = N_GENERATIONS,
    variants_per_gen: int = VARIANTS_PER_GEN,
    seed_base: int = 42,
) -> List[dict]:
    corpus = load_corpus()
    rng = random.Random(seed_base)
    all_results: List[dict] = []

    print("[Natural motion experiment — null fitness]")
    print(f"Corpus: {len(corpus)} passages")
    print(f"Generations: {n_generations}  Variants/gen: {variants_per_gen}")
    print(f"No fitness function.  Recording what the creature does anyway.\n")

    # Config pool: vary hyperparameters without any fitness signal
    # telling us which is better.  We simply watch what each config produces.
    base_configs = [
        {"learn_steps": 5,  "learn_lr": 0.01,  "temperature": 0.8,  "alpha": 0.85},
        {"learn_steps": 8,  "learn_lr": 0.005, "temperature": 1.0,  "alpha": 0.80},
        {"learn_steps": 3,  "learn_lr": 0.02,  "temperature": 1.2,  "alpha": 0.90},
        {"learn_steps": 10, "learn_lr": 0.001, "temperature": 0.7,  "alpha": 0.85},
        {"learn_steps": 5,  "learn_lr": 0.01,  "temperature": 1.5,  "alpha": 0.75},
    ]

    for gen in range(n_generations):
        print(f"  Generation {gen}")
        gen_results = []

        for v_idx in range(variants_per_gen):
            config = dict(base_configs[(gen * variants_per_gen + v_idx) % len(base_configs)])
            texts = rng.sample(corpus, min(TEXTS_PER_VARIANT, len(corpus)))
            seed = seed_base + gen * 100 + v_idx

            result = run_variant(texts, config, gen, v_idx, seed)
            gen_results.append(result)
            all_results.append(result)

            print(f"    variant {v_idx}  config=lr{config['learn_lr']}/t{config['temperature']}")
            for obs in result["observations_cold"][:2]:
                print(f"      cold:   \"{obs['text']}\"")
                print(f"              loss={obs['loss']:.4f}  curv={obs['curvature']:.6f}"
                      f"  betti={obs['betti']}")
            for obs in result["observations_seeded"][:1]:
                print(f"      seeded: \"{obs['text']}\"")

        gen_file = RESULTS_DIR / f"generation_{gen:03d}.json"
        gen_file.write_text(json.dumps(gen_results, indent=2, default=str))
        print(f"    -> {gen_file.name}\n")

    return all_results


# ── Analysis: describe, don't test ───────────────────────────────────────

def analyze(results: List[dict]) -> None:
    """Describe the natural motion without predetermined categories.

    We are not testing a hypothesis.  We are reading what happened.
    """
    if not results:
        print("No results to analyze.")
        return

    print("=" * 70)
    print("NATURAL MOTION — OPEN DESCRIPTION")
    print("=" * 70)
    print()

    # All generated texts — just print them, in order
    print("── Generated texts (cold start, all variants, all generations) ──")
    for r in results:
        gen, v = r["generation"], r["variant_idx"]
        for obs in r["observations_cold"]:
            print(f"  gen{gen} v{v} t={obs['temperature']:.2f}: \"{obs['text']}\"")
            print(f"          loss={obs['loss']:.4f}  curv={obs['curvature']:.6f}"
                  f"  betti={obs['betti']}  surprise_max={obs['surprise_max']:.3f}")
    print()

    # Distribution of curvature in outputs
    all_curvs = [
        obs["curvature"]
        for r in results
        for obs in r["observations_cold"] + r["observations_seeded"]
    ]
    if all_curvs:
        arr = np.array(all_curvs)
        print(f"── Output curvature distribution ({len(arr)} samples) ──")
        print(f"  min={arr.min():.6f}  max={arr.max():.6f}"
              f"  mean={arr.mean():.6f}  std={arr.std():.6f}")
        counts, edges = np.histogram(arr, bins=6)
        for i, c in enumerate(counts):
            bar = "█" * c
            print(f"  [{edges[i]:.4f}-{edges[i+1]:.4f}]: {bar} ({c})")
    print()

    # Betti number distribution in outputs
    all_betti = [
        tuple(obs["betti"])
        for r in results
        for obs in r["observations_cold"] + r["observations_seeded"]
    ]
    betti_counts = Counter(all_betti)
    print(f"── Output Betti distribution ({len(all_betti)} samples) ──")
    for betti, count in sorted(betti_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {betti}: {bar} ({count})")
    print()

    # Loss trajectories
    print("── Loss trajectories ──")
    for r in results:
        gen, v = r["generation"], r["variant_idx"]
        for lt in r["loss_trajectories"]:
            traj = lt["trajectory"]
            if len(traj) >= 2:
                imp = traj[0] - traj[-1]
                print(f"  gen{gen} v{v}: {traj[0]:.4f}->{traj[-1]:.4f}"
                      f"  improvement={imp:+.4f}  text=\"{lt['text_preview']}\"")
    print()

    # Weight norm drift across generations
    print("── Weight norm drift across generations ──")
    for r in results:
        gen, v = r["generation"], r["variant_idx"]
        norm = r["weight_snapshot"]["total_norm"]
        print(f"  gen{gen} v{v}: total_weight_norm={norm:.6f}")
    print()

    # Outliers — outputs with unusual curvature or non-trivial betti
    mean_curv = float(np.mean(all_curvs)) if all_curvs else 0.0
    std_curv  = float(np.std(all_curvs))  if all_curvs else 0.0
    print("── Outliers (curvature > mean+std or betti[1] > 0) ──")
    found_outlier = False
    for r in results:
        gen, v = r["generation"], r["variant_idx"]
        for obs in r["observations_cold"] + r["observations_seeded"]:
            high_curv  = obs["curvature"] > mean_curv + std_curv
            high_betti = obs["betti"][1] > 0
            if high_curv or high_betti:
                found_outlier = True
                reason = []
                if high_curv:  reason.append(f"curv={obs['curvature']:.6f}")
                if high_betti: reason.append(f"betti={obs['betti']}")
                print(f"  gen{gen} v{v}: \"{obs['text']}\"")
                print(f"    {', '.join(reason)}")
    if not found_outlier:
        print("  None found — outputs clustered near mean.")
    print()

    print("── End of description ──")
    print("What do you see?  We have no verdict.  The data is the record.")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Natural motion recorder: null-fitness evolution. "
                    "No hypothesis.  No verdict.  Just observation."
    )
    parser.add_argument("--quick",       action="store_true",
                        help="2 generations, 2 variants each")
    parser.add_argument("--analyze",     action="store_true",
                        help="Describe saved results without running experiment")
    parser.add_argument("--generations", type=int, default=N_GENERATIONS)
    parser.add_argument("--variants",    type=int, default=VARIANTS_PER_GEN)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    if args.analyze:
        results = []
        for f in sorted(RESULTS_DIR.glob("generation_*.json")):
            try:
                results.extend(json.loads(f.read_text()))
            except Exception:
                pass
        analyze(results)
        return

    n_gen = 2 if args.quick else args.generations
    n_var = 2 if args.quick else args.variants

    t0 = time.time()
    results = run_experiment(
        n_generations=n_gen,
        variants_per_gen=n_var,
        seed_base=args.seed,
    )
    elapsed = time.time() - t0
    print(f"Runtime: {elapsed:.1f}s  ({len(results)} variant records)\n")

    analyze(results)

    summary_file = RESULTS_DIR / "summary.json"
    summary_file.write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("loss_trajectories",)}
         for r in results],
        indent=2, default=str,
    ))
    print(f"\nResults -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
