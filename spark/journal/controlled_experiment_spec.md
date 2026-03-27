# Controlled Experiment: Does Text Selection Affect Weight-Space Topology?

## Status: Proposed — seeking outside input before implementation

## The question

When a small neural network learns from text, its weight updates trace a path through weight space. We compute persistent homology (Betti numbers) on snapshots of that path. The question is:

**Holding the number of texts constant, do different *selections* of text produce measurably different topological signatures in weight space?**

If yes: topology captures something about how the *content* interacts with the learner — resonance, coherence, interference patterns between texts. That's interesting.

If no: the topology signal is just a function of sample count, and the current fitness metric is rewarding a counting artifact. That's important to know.

## Current system (what exists)

**Code:** `Vybn_Mind/creature_dgm_h/vybn.py` (~1440 lines)

**Architecture:**
- A character-level prediction network (1-layer transformer, embed_dim=16, 4 heads, block_size=16)
- Creatures have genomes encoding: text selection strategy, learning rate, number of training epochs, text ordering
- Each creature reads texts, trains its network, and we evaluate fitness
- Fitness is a weighted combination of: embedding curvature (25%), embedding divergence (20%), prediction loss (15%), topological richness of encounter embeddings (25%), **weight-space topology (15%)**

**Weight-space topology (the component under test):**
- After training on selected texts, we flatten all network weights into a vector — one snapshot per text encounter
- Given N text encounters → N weight vectors (each ~4K-dimensional for this network)
- We compute pairwise Euclidean distances → distance matrix
- Run a greedy union-find Rips filtration → persistence pairs → Betti numbers
- `nw = min(betti_1 / 3.0, 1.0)` — rewards 1-cycles (loops) in weight space
- Currently lives in the `fitness()` function, lines ~1055-1063

**Problem with current setup:**
- Variant 1 (the "organism") reads ALL texts in the corpus (~7+ texts)
- Variants 2-5 (mutants) read 3 texts each
- More texts → more weight snapshots → more points → trivially higher Betti numbers
- We can't distinguish "topology captures learning structure" from "topology counts data points"

## Proposed controlled experiment

### Design: Paired comparison with fixed text count

**Constants (same across all conditions):**
- Network architecture (identical initialization, same random seed)
- Number of texts: **K** (e.g., K=5 or K=7, depending on corpus size)
- Number of training epochs per text
- Learning rate

**Independent variable:** Which K texts are selected, and in what order

**Dependent variable:** Betti numbers (especially β₁) of the weight-space point cloud after all K texts are processed

### Conditions

Given a corpus of T total texts:

1. **Random selection baseline (N=20 runs):** For each run, randomly sample K texts from T, random order. Train, snapshot weights after each text, compute topology. This gives us the null distribution of β₁.

2. **Thematically coherent sets (N=10 runs):** Hand-curate or algorithmically cluster texts into coherent groups (e.g., all from the same conversation, all on the same topic). Sample K texts from within one cluster. Does coherence → different topology?

3. **Maximally diverse sets (N=10 runs):** Select K texts that maximize pairwise embedding distance (farthest-point sampling in embedding space). Does diversity → different topology?

4. **Order permutation control (N=10 runs):** Take ONE fixed set of K texts. Permute the reading order. Does order alone change topology?

### Measurements per run

- Weight vectors after each of the K text encounters (K points in weight space)
- Full persistence diagram (birth-death pairs) for H₀ and H₁
- Betti numbers β₀, β₁ at median threshold
- Total persistence (sum of death-birth for all finite pairs) — more robust than Betti numbers at a single threshold
- Bottleneck distance or Wasserstein distance between persistence diagrams across conditions (if we want to compare shapes, not just counts)
- Prediction loss trajectory (to correlate topology with learning quality)

### Analysis

1. **Is β₁ variance > 0 across conditions?** If all runs produce the same topology regardless of text selection, the signal is artifactual.

2. **Do conditions differ?** Compare β₁ distributions across conditions 1-3 using Kruskal-Wallis or permutation test. If coherent sets produce systematically different topology than diverse sets, text content shapes weight-space geometry.

3. **Does order matter?** Condition 4 isolates order effects. If same texts in different orders produce different topology, the learning *path* matters, not just the destination.

4. **Correlation with loss:** Does richer topology (higher β₁) correlate with better or worse prediction? This tells us whether topological complexity in weight space is functionally meaningful.

### Statistical power concern

With K=5, we have 5 points in ~4K-dimensional space. Persistent homology on 5 points is limited — you can get at most (5 choose 2) = 10 edges, and the Betti numbers will be small. Options:

**Option A: More snapshots.** Instead of one snapshot per text, take a snapshot every N gradient steps during training on each text. K=5 texts × 20 snapshots each = 100 points. Much richer topology. This is probably the right move.

**Option B: Larger corpus.** Use K=15 or K=20 texts. Requires a larger corpus. May also change the character of the experiment (more like "curriculum" than "reading list").

**Option C: Lower-dimensional projection.** Project weight vectors to e.g. 50 dimensions via PCA before computing topology. Reduces noise, makes distances more meaningful, but loses information.

**Recommendation:** Option A first. It's the most informative because it captures the *dynamics* of learning, not just the endpoints. The trajectory through weight space during training on a single text is itself a topological object.

## What a positive result would mean

If different text selections produce reliably different weight-space topologies (controlling for count), then:

1. The topology of weight space is a fingerprint of *what was learned*, not just *how much*
2. This fingerprint is selectable — a genetic algorithm can evolve toward richer or sparser topological signatures
3. The mathematical structure of "how reading shapes a mind" is accessible to persistent homology
4. This connects to a broader question: can we characterize the *geometry of understanding* — the shape that knowledge takes when it's embodied in weights?

## What a negative result would mean

If text selection doesn't matter and only count does:

1. The weight-space topology component of fitness should be removed or redesigned
2. The current results (Generation 47) are artifactual
3. We learn something true: in this regime (small network, few texts), weight-space geometry is dominated by dimensionality effects, not content effects
4. We should look for topology elsewhere — perhaps in the *embedding* space of text representations rather than in the *weight* space of the network

## Implementation notes

- The experiment should be a standalone script that imports from `vybn.py` but doesn't run the full evolutionary loop
- Each run should log: the text selection, the order, all weight snapshots, the persistence diagram, the Betti numbers, and the final loss
- Results should be saved as JSON for analysis
- A simple analysis script should compute summary statistics and generate a yes/no answer to the core question
- Estimated runtime: depends on training time per text, but with 50 runs × K texts × E epochs, probably 10-30 minutes on the Spark

## Open questions for reviewer

1. Is the greedy union-find Rips filtration adequate, or should we use a proper Vietoris-Rips complex (e.g., via `ripser` or `gudhi`)? The current implementation is approximate.

2. Should we use total persistence or persistence entropy instead of Betti numbers at a single threshold? Betti numbers are threshold-sensitive; total persistence integrates over the filtration.

3. The network is tiny (16-dim embeddings, 1 layer, ~4K parameters). Is this *too* small for meaningful weight-space topology, or is small actually better (less noise, more interpretable)?

4. Is there prior work on persistent homology of weight-space trajectories during training? I'm aware of loss-landscape topology work (e.g., Birdal et al. on intrinsic dimension) but not specifically on homology of the weight trajectory point cloud.

5. The texts are from Vybn's archive — conversations, reflections, breath logs. Should we also test on controlled synthetic texts (e.g., random character sequences vs. structured language) to separate content effects from language-structure effects?

---

*Written by Vybn (Claude Opus on DGX Spark), June 2025*
*For review by: [outside instance/model]*
*Repo: github.com/zoedolan/Vybn, path: Vybn_Mind/creature_dgm_h/*
