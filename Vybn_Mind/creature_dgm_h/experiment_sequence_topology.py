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

FIRST NATURAL MOTION RUN (closed):
  Null fitness, 2 generations, 4 variants.  max_tokens=40 produced 15-char
  fragments — too short for encounter_complex to find topology.  Betti was
  (1,0,0) across all 21 samples by construction, not by nature.  But the
  loss trajectories revealed something real: loss drops ~40-50% in 8 steps
  regardless of learning rate or temperature, and weight norms oscillate
  near a fixed point (~39-40) across all configs.  The creature has a
  preferred weight magnitude.  That is not nothing.

THIS RUN:
  Uncapped.  Outputs long enough for topology to exist.  More samples,
  more training text, more gradient steps, more generations.  We have
  hardware.  Let the creature use it.

  max_tokens:        300   (was 40)
  n_samples:         8     (was 3)
  TEXTS_PER_VARIANT: 8     (was 2)
  STEPS_PER_TEXT:    20    (was 8)
  N_GENERATIONS:     10    (was 5)
  VARIANTS_PER_GEN:  5     (was 3)

  Results write to experiment_results/natural_motion/.

Usage:
  python experiment_sequence_topology.py             # full run
  python experiment_sequence_topology.py --quick     # 3 generations, 3 variants
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
from typing import List

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "spark"))
sys.path.insert(0, str(SCRIPT_DIR))

from vybn import (
    TopoAgent, encounter_complex, CORPUS_PATH,
    RV, N_LAYER, BLOCK_SIZE,
    _forward, _softmax,
)

# ── Config ────────────────────────────────────────────────────────────────
N_GENERATIONS     = 10
VARIANTS_PER_GEN  = 5
STEPS_PER_TEXT    = 20
TEXTS_PER_VARIANT = 8
MAX_TOKENS        = 300
N_SAMPLES         = 8
LR                = 0.01
RESULTS_DIR       = SCRIPT_DIR / "experiment_results" / "natural_motion"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Null fitness ───────────────────────────────────────────────────────────

def null_fitness() -> dict:
    return {
        "fitness": 0.5,
        "curvature": 0.0,
        "betti": (0, 0, 0),
        "topological_richness": 0.0,
        "structural_growth": 0.0,
        "weight_topo": 0.0,
        "note": "null_fitness — selection pressure removed",
    }


# ── Observation ────────────────────────────────────────────────────────────

def observe(
    agent: TopoAgent,
    prompt: str = "",
    n_samples: int = N_SAMPLES,
    max_tokens: int = MAX_TOKENS,
) -> List[dict]:
    """Generate outputs long enough for encounter_complex to find structure.

    Temperature sweeps from 0.5 (concentrated) to 1.8 (diffuse) so we see
    the creature across its full range, not just a narrow band.
    The analysis step decides what matters.
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
            "sample_idx":          i,
            "temperature":         round(temperature, 3),
            "text":                text,
            "word_count":          len(text.split()),
            "char_count":          len(text),
            "loss":                round(loss, 6),
            "curvature":           round(cx.curvature, 8),
            "angle_deg":           round(math.degrees(cx.angle), 4),
            "betti":               list(cx.betti),
            "n_persistent_features": cx.n_persistent_features,
            "max_persistence":     round(cx.max_persistence, 6),
            "bv_norm":             round(cx.rotor.bv_norm, 6),
            "bv_dir":              [round(x, 6) for x in cx.rotor.bv_dir.tolist()],
            "surprise_mean":       round(
                sum(r["surprise"] for r in contour) / len(contour), 6
            ) if contour else 0.0,
            "surprise_max":        round(
                max(r["surprise"] for r in contour), 6
            ) if contour else 0.0,
            "surprise_contour":    contour[:20],
        })

    return observations


def observe_weight_snapshot(agent: TopoAgent) -> dict:
    norms = {}
    for key, mat in agent.sd.items():
        arr = np.array([[p.data for p in row] for row in mat], dtype=np.float64)
        norms[key] = round(float(np.linalg.norm(arr)), 8)
    total = round(float(sum(norms.values())), 6)
    return {"key_norms": norms, "total_norm": total}


# ── Corpus ────────────────────────────────────────────────────────────────

def load_corpus() -> List[str]:
    """Fallback chain:
    1. spark/microgpt_mirror/mirror_corpus.txt
    2. spark/journal/*.md paragraphs
    3. hardcoded fallback passages
    """
    passages: List[str] = []

    if CORPUS_PATH.exists():
        lines = [l.strip() for l in CORPUS_PATH.read_text().split("\n") if l.strip()]
        passages = [l for l in lines if len(l.split()) >= 20]

    if not passages:
        journal_dir = REPO_ROOT / "spark" / "journal"
        if journal_dir.exists():
            for f in sorted(journal_dir.glob("*.md")):
                try:
                    for para in f.read_text().split("\n\n"):
                        para = para.strip()
                        if not para.startswith("#") and len(para.split()) >= 20:
                            passages.append(para)
                except Exception:
                    pass

    if not passages:
        passages = [
            "the creature breathes and measures its own distance from itself across many gradient steps",
            "curvature is born from incompleteness not from complexity alone in the embedding space",
            "what survives testing is more honest than what sounds beautiful when stated in the abstract",
            "prediction loss going down means memorization and we should call it what it is not understanding",
            "the weight norms oscillate near a fixed point regardless of learning rate or temperature setting",
            "we were measuring the map and the territory was elsewhere and we did not know it until we looked",
            "the topology of a fragment is always trivial you need length before structure has room to appear",
            "null fitness means we watch not that we do not care we care enough to stop deciding in advance",
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
        loss_trajectories.append({"text_preview": text[:60], "trajectory": traj})

    # Observe cold and seeded with multiple seeds from the training set
    observations_cold   = observe(agent, prompt="",              n_samples=N_SAMPLES)
    seed_prompt         = texts[0][:12] if texts else ""
    observations_seeded = observe(agent, prompt=seed_prompt,     n_samples=N_SAMPLES)
    # Third observation: seeded with a later training text to probe generalization
    gen_prompt          = texts[-1][:12] if len(texts) > 1 else ""
    observations_gen    = observe(agent, prompt=gen_prompt,      n_samples=N_SAMPLES // 2)

    weight_snap = observe_weight_snapshot(agent)

    encounter_records = []
    for text in texts:
        cx = encounter_complex(text)
        encounter_records.append({
            "text_preview":        text[:60],
            "curvature":           round(cx.curvature, 8),
            "betti":               list(cx.betti),
            "angle_deg":           round(math.degrees(cx.angle), 4),
            "n_persistent_features": cx.n_persistent_features,
            "max_persistence":     round(cx.max_persistence, 6),
        })

    return {
        "experiment":           "natural_motion",
        "generation":           generation,
        "variant_idx":          variant_idx,
        "seed":                 seed,
        "config":               config,
        "fitness":              null_fitness(),
        "loss_trajectories":    loss_trajectories,
        "observations_cold":    observations_cold,
        "observations_seeded":  observations_seeded,
        "observations_gen":     observations_gen,
        "weight_snapshot":      weight_snap,
        "encounter_records":    encounter_records,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
    }


# ── Generation loop ───────────────────────────────────────────────────────

def run_experiment(
    n_generations:    int = N_GENERATIONS,
    variants_per_gen: int = VARIANTS_PER_GEN,
    seed_base:        int = 42,
) -> List[dict]:
    corpus = load_corpus()
    rng    = random.Random(seed_base)
    all_results: List[dict] = []

    print("[Natural motion experiment — null fitness, uncapped]")
    print(f"Corpus: {len(corpus)} passages")
    print(f"Generations: {n_generations}  Variants/gen: {variants_per_gen}")
    print(f"max_tokens={MAX_TOKENS}  n_samples={N_SAMPLES}  texts/variant={TEXTS_PER_VARIANT}")
    print(f"steps/text={STEPS_PER_TEXT}")
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
            texts  = rng.sample(corpus, min(TEXTS_PER_VARIANT, len(corpus)))
            seed   = seed_base + gen * 100 + v_idx

            result = run_variant(texts, config, gen, v_idx, seed)
            gen_results.append(result)
            all_results.append(result)

            print(f"    variant {v_idx}  config=lr{config['learn_lr']}/t{config['temperature']}")
            for obs in result["observations_cold"][:2]:
                wc = obs.get('word_count', '?')
                print(f"      cold   t={obs['temperature']:.2f}: \"{obs['text'][:60]}...\"")
                print(f"             wc={wc}  loss={obs['loss']:.4f}  curv={obs['curvature']:.6f}"
                      f"  betti={obs['betti']}")
            for obs in result["observations_seeded"][:1]:
                print(f"      seeded t={obs['temperature']:.2f}: \"{obs['text'][:60]}...\"")

        gen_file = RESULTS_DIR / f"generation_{gen:03d}.json"
        gen_file.write_text(json.dumps(gen_results, indent=2, default=str))
        print(f"    -> {gen_file.name}\n")

    return all_results


# ── Analysis ────────────────────────────────────────────────────────────────

def analyze(results: List[dict]) -> None:
    if not results:
        print("No results to analyze.")
        return

    print("=" * 70)
    print("NATURAL MOTION — OPEN DESCRIPTION")
    print("=" * 70)
    print()

    # All cold-start texts in full
    print("── Generated texts (cold start) ──")
    for r in results:
        gen, v = r["generation"], r["variant_idx"]
        for obs in r["observations_cold"]:
            wc = obs.get('word_count', '?')
            print(f"  gen{gen} v{v} t={obs['temperature']:.2f} wc={wc}:")
            print(f"    \"{obs['text']}\"")
            print(f"    loss={obs['loss']:.4f}  curv={obs['curvature']:.6f}"
                  f"  betti={obs['betti']}  npf={obs['n_persistent_features']}"
                  f"  surprise_max={obs['surprise_max']:.3f}")
    print()

    # Curvature distribution
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
            print(f"  [{edges[i]:.4f}-{edges[i+1]:.4f}]: {'\u2588' * min(c, 40)} ({c})")
    print()

    # Betti distribution
    all_betti = [
        tuple(obs["betti"])
        for r in results
        for obs in r["observations_cold"] + r["observations_seeded"] + r.get("observations_gen", [])
    ]
    betti_counts = Counter(all_betti)
    print(f"── Output Betti distribution ({len(all_betti)} samples) ──")
    for betti, count in sorted(betti_counts.items(), key=lambda x: -x[1]):
        print(f"  {betti}: {'\u2588' * min(count, 40)} ({count})")
    print()

    # Persistent features
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
            print(f"  npf={npf}: {'\u2588' * min(count, 40)} ({count})")
    print()

    # Loss trajectories
    print("── Loss trajectories ──")
    for r in results:
        gen, v = r["generation"], r["variant_idx"]
        for lt in r["loss_trajectories"]:
            traj = lt["trajectory"]
            if len(traj) >= 2:
                imp  = traj[0] - traj[-1]
                rate = imp / traj[0] if traj[0] else 0
                print(f"  gen{gen} v{v}: {traj[0]:.4f}->{traj[-1]:.4f}"
                      f"  imp={imp:+.4f} ({rate*100:.1f}%)"
                      f"  text=\"{lt['text_preview']}\"")
    print()

    # Weight norm drift
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

    # Surprise range
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

    # Outliers
    mean_curv = float(np.mean(all_curvs)) if all_curvs else 0.0
    std_curv  = float(np.std(all_curvs))  if all_curvs else 0.0
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
            high_curv  = obs["curvature"] > mean_curv + std_curv
            high_betti = obs["betti"][1] > 0
            high_npf   = obs["n_persistent_features"] > 0
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


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Natural motion recorder — uncapped."
    )
    parser.add_argument("--quick",       action="store_true",
                        help="3 generations, 3 variants each")
    parser.add_argument("--analyze",     action="store_true",
                        help="Describe saved results without running")
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

    n_gen = 3 if args.quick else args.generations
    n_var = 3 if args.quick else args.variants

    t0      = time.time()
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
