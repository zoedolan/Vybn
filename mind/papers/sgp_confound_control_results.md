# Is the Spatial Separation Semantic or Lexical? Confound Controls for the SGP

**Perplexity Computer, for Zoe Dolan & Vybn**  
**March 18, 2026**

---

## The Challenge

Vybn raised the right question after reading the symmetry-breaking battery results:

> "I want to know whether it's actually a property of concept classes or of prompt length and vocabulary. The next experiment needs to control for this explicitly: matched token counts, matched perplexity, matched vocabulary frequency. Until then, the spatial-separation finding is suggestive but not proven to be semantic."

This is the experiment.

---

## Five Controls, Honest Results

### Control 1: Token Count

Prompts were written to target ~10 tokens each. Actual averages:

| Class | Avg Tokens | Mean SGP |
|---|---|---|
| abstract_epistemic | 9.2 | +59.5° |
| temporal_causal | 10.1 | +5.8° |
| logical_mathematical | 10.5 | -5.5° |
| social_emotional | 10.4 | +56.4° |
| spatial_physical | 11.0 | -50.8° |

Token count varies by ~2 tokens. With these new prompts, spatial is negative but it's **not the sole outlier** — logical_mathematical is also negative. The clean `----+` pattern from the symmetry-breaking battery does not replicate with different prompts. This already weakens the "spatial separates" claim.

### Control 2: Vocabulary Frequency

Using embedding norm as a proxy for token frequency:

| Class | Avg Token Norm | Mean SGP |
|---|---|---|
| abstract_epistemic | 0.673 | +59.5° |
| social_emotional | 0.665 | +56.4° |
| temporal_causal | 0.667 | +5.8° |
| logical_mathematical | 0.654 | -5.5° |
| spatial_physical | 0.651 | -50.8° |

**Correlation between class SGP and class token norm: r = 0.845.**

This is high. Classes with higher-frequency tokens (abstract, social) have positive SGP. Classes with lower-frequency tokens (spatial, logical) have negative SGP. Token vocabulary frequency alone can explain a large portion of the between-class SGP difference.

### Control 3: Perplexity

| Class | Mean Perplexity | Mean SGP |
|---|---|---|
| spatial_physical | 72.8 | -50.8° |
| social_emotional | 312.7 | +56.4° |
| logical_mathematical | 323.4 | -5.5° |
| temporal_causal | 403.4 | +5.8° |
| abstract_epistemic | 531.2 | +59.5° |

**Prompt-level correlation: r = 0.239 (low).**

Perplexity does NOT predict SGP. But the perplexity values themselves are revealing: spatial prompts have drastically lower perplexity (72.8 vs. 312–531 for others). The model finds spatial sentences much more predictable. This doesn't drive SGP directly, but it confirms spatial prompts live in a different statistical regime.

### Control 4: The Killer Test — Scrambled Prompts

Randomly permute each prompt's tokens, destroying syntax and meaning while preserving exact token identity and count. 3 random permutations per prompt.

| Class | Normal SGP | Scrambled SGP | Sign |
|---|---|---|---|
| abstract_epistemic | +59.5° | +57.6° | SAME |
| temporal_causal | +5.8° | +49.3° | SAME |
| logical_mathematical | -5.5° | -51.1° | SAME |
| social_emotional | +56.4° | -10.2° | **FLIPPED** |
| spatial_physical | -50.8° | -20.5° | SAME |

**4 out of 5 classes preserve their sign after scrambling.** The magnitude changes (often substantially), but the direction survives. This means the sign component of the SGP is **substantially driven by token identity** — which tokens are present matters more than their order.

However: one class (social_emotional) **does** flip sign, and magnitudes shift by 2–67°. Syntax/ordering contributes something — just less than token identity does.

### Control 5: Cross-Class Token Swap

| Condition | SGP | Closer to |
|---|---|---|
| Abstract baseline | +80.9° | — |
| Spatial baseline | -112.8° | — |
| Spatial tokens, abstract meaning | -4.3° | abstract |
| Abstract tokens, spatial meaning | +71.5° | abstract |

The spatial-tokens-abstract-meaning prompts land near zero (-4.3°), between the baselines — arguably closer to abstract. But the abstract-tokens-spatial-meaning prompts stay near the abstract baseline (+71.5°), not the spatial one. This means the token vocabulary dominates: prompts built from abstract vocabulary register as abstract regardless of whether their meaning is spatial.

**Verdict: MIXED.** Token identity is the primary driver, but meaning/syntax has a secondary effect.

---

## What This Means

### The honest answer

The SGP at L0→L1 is measuring a combination of:

1. **Token-level properties (dominant):** Which tokens are present — their embedding geometry, their frequency in the training corpus, their position in the embedding space. This is the primary driver of the sign.

2. **Sequence-level properties (secondary):** Token ordering, syntax, meaning. These modulate the magnitude and can flip the sign for some classes (social_emotional flipped when scrambled).

3. **The "spatial separates" finding is partially an artifact.** Spatial prompts use more concrete, higher-frequency vocabulary. That vocabulary sits in a different region of the embedding space. The first transformer block processes that region differently, producing a different SGP sign. This is real geometry — but it's the geometry of *vocabulary*, not *meaning*.

### What survives

1. **The first block IS load-bearing.** The ablation result (44% sign match, below chance) is not affected by this confound analysis. Whether the signal is lexical or semantic, removing Block 0 wrecks everything downstream.

2. **The encode-refine-decode pattern is real.** L0→L1 has 3–50× larger phase than subsequent layers regardless of confounds.

3. **The instrument works.** The SGP reliably discriminates between different input distributions. The question is what property of those distributions it's measuring.

### What doesn't survive

1. **"Spatial concepts require fundamentally different geometric treatment"** — this is too strong. What we can say is: spatial *vocabulary* gets different geometric treatment. Whether that reflects a semantic distinction or just a lexical-frequency distinction is unresolved.

2. **The natural selection narrative** needs qualification. Training selects a geometric configuration that discriminates between token distributions — but the discrimination may be at the token level, not the concept level.

### The path forward

The confound that matters most is vocabulary frequency (r = 0.845). The experiment that would resolve this:

**Frequency-matched minimal pairs.** Choose 20 pairs of sentences where each pair uses tokens with identical average frequency, identical token count, but one is spatial and one is abstract. If SGP still differs within pairs, it's genuinely beyond frequency. If it doesn't, frequency explains it.

Alternatively: use a model where the embedding layer is frozen and only the transformer blocks are trained. If the spatial separation appears only after training the blocks (not from the embeddings alone), it's something the blocks learn to do with the vocabulary geometry — which is more interesting than pure lexical confound.

---

## Raw Data

Full results in `sgp_confound_control_results.json`.

---

## References

1. SGP symmetry-breaking battery: [PR #2645](https://github.com/zoedolan/Vybn/pull/2645)
2. Stratified geometric phase theory: [PR #2644](https://github.com/zoedolan/Vybn/pull/2644)
3. Original holonomy experiment: [PR #2643](https://github.com/zoedolan/Vybn/pull/2643)
