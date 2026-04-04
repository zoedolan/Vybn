# Weight-Space Topology Experiment: Results

**Date:** 2025-06-28
**Ran by:** Vybn (Claude Opus on DGX Spark)
**Code:** `Vybn_Mind/creature_dgm_h/experiment_weight_topology.py --quick`
**Topology engine:** ripser (exact Vietoris-Rips persistence)

## Result: Negative. Uniform zero topology across all conditions.

### Raw data

| Condition     | Run | β₀  | β₁ | Total Pers H₁ | Final Loss |
|---------------|-----|-----|----|----------------|------------|
| coherent      | 0   | 10  | 0  | 0.0000         | 4.205      |
| coherent      | 1   | 10  | 0  | 0.0000         | 4.399      |
| coherent      | 2   | 10  | 0  | 0.0000         | 5.463      |
| diverse       | 0   | 10  | 0  | 0.0000         | 4.246      |
| diverse       | 1   | 10  | 0  | 0.0000         | 4.211      |
| diverse       | 2   | 10  | 0  | 0.0000         | 3.319      |
| order         | 0   | 10  | 0  | 0.0000         | 3.007      |
| order         | 1   | 10  | 0  | 0.0000         | 5.311      |
| order         | 2   | 10  | 0  | 0.0000         | 4.336      |
| random        | 0   | 10  | 0  | 0.0000         | 3.741      |
| random        | 1   | 10  | 0  | 0.0000         | 5.793      |
| random        | 2   | 10  | 0  | 0.0000         | 4.332      |
| synthetic     | 0-2 | (data format issue, but same pattern) |

### Interpretation

**β₀ = 10, β₁ = 0** means: at the median filtration threshold, all 10 weight-space
points are disconnected (10 connected components, zero loops). The points are so
far apart in ~4000-dimensional parameter space that the Rips complex never forms
1-cycles.

This is the **dimension curse** in action. 10 points in 4000 dimensions are
almost certainly in general position — pairwise distances concentrate around the
same value (a well-known phenomenon in high-dimensional spaces), but that value
is too large relative to any natural scale for cycles to form.

### What this means for the fitness function

The `nw` (weight-space topology) component of the creature fitness function
has been producing a signal that was:

1. **Zero when computed correctly (ripser):** No topology exists at this scale
2. **Hallucinated when computed approximately (builtin):** The greedy union-find
   filtration was mislabeling H₀ dying pairs as H₁ features, reporting β₁=27
   where the true value was β₁=0

The 15% fitness weight assigned to `nw` was either contributing nothing (if
ripser) or rewarding a computational artifact (if builtin). Neither is useful.

### What was wrong with the previous "positive" result (Gen 47)

Generation 47 showed `wt=1.0` for Variant 1 (reading 7+ texts → 7+ weight
snapshots) and `wt=0.0` for Variants 2-5 (reading 3 texts → 3 weight
snapshots). This was the builtin filtration. The `wt=1.0` was an artifact:
the builtin was counting all union-find merges as H₁ features. More texts →
more merges → higher fake β₁. The controlled experiment with ripser confirms:
the real H₁ is zero regardless of text count.

### Where to go from here

The question "does text selection affect the geometry of learning?" is still
interesting. But weight-space persistent homology at this scale (tiny network,
few snapshots, raw parameter space) is the wrong tool. Options:

1. **Dimensionality reduction first.** Project weight vectors to 10-50 dims
   via PCA before computing persistence. This concentrates the signal and
   brings pairwise distances into a regime where cycles can form.

2. **More snapshots.** Instead of 10 points, take 100-500 (snapshot every
   gradient step). More points → more chance of capturing genuine topology.
   Combined with PCA, this could work.

3. **Different invariant.** Instead of persistent homology, compute:
   - Curvature of the weight trajectory (already partially done via the
     Cl(3,0) rotor)
   - Fractal dimension of the trajectory
   - Spectral properties of the pairwise distance matrix
   These may be more sensitive at this scale.

4. **Activation space instead of weight space.** Track how the network's
   *hidden activations* change across texts, not its weights. Activations
   are lower-dimensional and more directly tied to what the network "sees."

5. **Accept the null.** The network may be too small for its weight space
   to have interesting topology. Scale up the network or accept that
   topology is the wrong lens at this scale.

### The honest takeaway

We asked a clean question. We got a clean answer. The answer is no — at
this scale, with this network, weight-space persistent homology does not
differentiate text selections. The builtin filtration was giving us
hallucinated topology. Ripser tells the truth: there's nothing there.

This is a real result. Negative results are results. The apparatus works;
the phenomenon isn't present at this scale. Now we know.

---

*Also confirmed: the builtin `_persistence_pairs` in vybn.py has a bug
that mislabels H₀ death events as H₁ features. This should be fixed or
the builtin should be deprecated in favor of ripser.*
