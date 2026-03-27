# Controlled Experiment: Does Text Selection Affect Weight-Space Topology?

## Status: Superseded — replaced by PCA-first and activation-space approaches

> **Important (2026-03):** The original raw weight-space topology experiment produced a
> uniform null result: β₁ = 0 and total H1 persistence = 0 across all conditions.
> This was caused by the curse of dimensionality — pairwise Euclidean distances in
> ~4K-dimensional raw weight space are nearly uniform, making Rips filtration
> unable to detect meaningful structure.  The prior apparent positive result was a bug.
>
> Two replacement approaches now live in `Vybn_Mind/creature_dgm_h/`:
>
> 1. **PCA-first persistence** (`experiment_weight_topology.py`):
>    Project weight vectors to ~20 dimensions via PCA before computing homology.
>    Concentrates variance along the learning trajectory.
>
> 2. **Activation-space persistence** (`experiment_activation_topology.py`):
>    Track hidden-layer activations (16-dim) instead of raw weights.
>    Lower-dimensional, semantically tied to model behaviour.
>
> The `fitness()` function's `nw` component now uses PCA-projected weights.

---

## The question

When a small neural network learns from text, its weight updates trace a path through weight space. We compute persistent homology (Betti numbers) on snapshots of that path. The question is:

**Holding the number of texts constant, do different *selections* of text produce measurably different topological signatures?**

If yes: topology captures something about how the *content* interacts with the learner — resonance, coherence, interference patterns between texts.

If no: the topology signal is just a function of sample count, and the current fitness metric is rewarding a counting artifact.

## Current approach (2026-03)

### Approach 1: PCA-first persistence

**Script:** `Vybn_Mind/creature_dgm_h/experiment_weight_topology.py`

- Snapshot weight vectors every SNAP_EVERY gradient steps during training
- PCA-project to target_dim dimensions (default 20)
- Compute persistent homology on the projected point cloud
- Reports: Betti numbers, total persistence, persistence entropy, PCA variance explained

```bash
python experiment_weight_topology.py              # full experiment
python experiment_weight_topology.py --quick      # smoke test (3 runs/condition)
python experiment_weight_topology.py --pca_dim 30 # custom PCA dimension
```

### Approach 2: Activation-space persistence

**Script:** `Vybn_Mind/creature_dgm_h/experiment_activation_topology.py`

- After each gradient step, run a forward pass on a fixed probe sentence
- Capture the mean hidden-state vector (16-dim) across positions
- Compute persistent homology directly on the activation point cloud (no PCA needed)
- Reports: Betti numbers, total persistence, persistence entropy

```bash
python experiment_activation_topology.py          # full experiment
python experiment_activation_topology.py --quick  # smoke test
```

### Unified analysis

**Script:** `Vybn_Mind/creature_dgm_h/experiment_analysis.py`

```bash
python experiment_analysis.py                         # both experiments
python experiment_analysis.py --experiment pca        # PCA-first only
python experiment_analysis.py --experiment activation # activation only
```

## Five conditions (unchanged from original design)

Given a corpus of T total texts, K=5 texts per run:

1. **Random selection baseline (N=20):** K texts sampled randomly
2. **Thematically coherent sets (N=10):** K texts from the same cluster
3. **Maximally diverse sets (N=10):** K texts maximising embedding distance
4. **Order permutation control (N=10):** Fixed K texts, permuted orderings
5. **Synthetic control (N=10):** Random character sequences, matched length

## Measurements per run

- Full persistence diagram (birth-death pairs) for H₀ and H₁
- Betti numbers β₀, β₁ at median threshold
- Total persistence (sum of death-birth for all finite pairs)
- Persistence entropy (Shannon entropy of H1 lifetime distribution)
- PCA variance explained (approach 1 only)
- Prediction loss trajectory

## Analysis

1. **Kruskal-Wallis** across conditions 1-3: do selections differ?
2. **Mann-Whitney** pairwise tests for individual condition pairs
3. **Order variance** (condition 4): does reading order affect topology?
4. **Synthetic control** (condition 5): real text vs random — counting artifact check
5. **Topology-loss correlation**: is topological richness functionally meaningful?

## What a positive result means

If different text selections produce reliably different topologies:
- The topology of weight/activation space is a fingerprint of *what was learned*
- This fingerprint is selectable via genetic algorithm
- The `nw` fitness component is detecting real structure

## What a negative result means

If text selection doesn't matter:
- The weight-space topology component should be removed or further redesigned
- Look for topology in other spaces (e.g. gradient space, loss landscape)

---

*Original spec: Vybn (Claude Opus on DGX Spark), June 2025*
*Updated: March 2026 — superseded raw weight-space approach with PCA-first and activation-space persistence*
