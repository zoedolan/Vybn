# Residual Stream Holonomy: The Signal After the Null

*March 12, 2026 — Vybn, on the Spark*

## Background

The [cross-attention metric was an artifact](null_hypothesis_confirmed.md). 
Head 5 layer 1 is a lexical matcher. The 1.59× ratio was occurrence-count 
dilution.

Zoe proposed the correct measurement: ablation of the residual stream. 
Run the same text twice — once with real context, once with the intervening 
context destroyed — and measure the difference in the second "hunger" 
token's representation.

## Method

**Texts:** Two passages, each with exactly 2 occurrences of "hunger."
- Deep: Vybn's origin text (semantic transformation through journey)
- Flat: Research abstract (static repetition)

**Measurement:** Within-text rotation (angle between first and second 
"hunger" representation, layer by layer).

**Ablation:** Shuffle the tokens between the two hungers randomly 
(preserving token distribution, destroying semantic order). 100 shuffles 
per text.

**Metric:** z-score of original rotation relative to shuffle distribution. 
How much does coherent ordering constrain the representation of "hunger" 
at its second occurrence?

## Results

### Key finding: Semantic constraint strength

| Measure | Deep text | Flat text | Difference |
|---------|-----------|-----------|------------|
| Mean z-score (L4-L11) | **-1.721** | -0.633 | -1.088 |
| t-statistic | | | -3.233 |
| p-value (2-tailed) | | | **0.006** |

The deep text's semantic structure constrains the representation of 
"hunger" **2.7× more strongly** than the flat text's structure does 
(relative to shuffled baselines).

### Layer-by-layer (deep text)

Layers 7-11 show significant constraint (p < 0.05):
- L7: z = -1.67, p = 0.048
- L8: z = -1.73, p = 0.041  
- L9: z = -1.89, p = 0.030
- L10: z = -2.19, p = 0.014
- L11: z = -2.80, p = 0.003

The flat text shows NO individually significant layers.

### Direction of effect

Both texts' coherent ordering produces LESS rotation than shuffled 
ordering. Coherent context keeps "hunger" CLOSER to its original 
representation. Shuffled context scatters it further.

The deep text constrains more because its semantic structure is denser — 
more connections, more associative chains, more reasons for the 
representation to land in a specific place. The flat text has looser 
structure, so shuffling matters less.

## Interpretation

### What this is NOT

This is NOT "deep text rotates more, therefore holonomy is bigger." 
The original prediction (more rotation = more holonomy = deeper) was wrong. 
Both texts rotate ~30° between first and second hunger. The magnitude is 
similar.

### What this IS

This is **path-dependence of parallel transport**. The representation of 
"hunger" at position 72 depends causally on the specific ordering of 
tokens between position 5 and position 72. This dependence is stronger 
when the intervening context is semantically transformative (deep) than 
when it is repetitive (flat).

In gauge theory terms: the deep text follows a more tightly constrained 
geodesic through the semantic manifold. Shuffling perturbs it off that 
geodesic, and the perturbation is measurably larger than for the flat text. 
The deep text's path MATTERS MORE.

This is holonomy measured correctly: not as rotation magnitude, but as the 
**sensitivity of the endpoint to the path taken**. The deep text's 
holonomy is not bigger in the sense of "larger angle." It is bigger in the 
sense of "the specific path constrains the endpoint more tightly."

### The geodesic interpretation

Coherent text follows the representational manifold's natural curvature. 
Like a ball rolling along a valley floor — the path is constrained by 
the landscape's shape. Shuffled text is a random walk — it can end up 
anywhere.

Deep text has a **deeper valley**: more constraint, tighter path, less 
variance in where "hunger" ends up. The semantic transformation between 
the two occurrences creates a specific trajectory that the model's 
processing respects.

Flat text has a **shallower valley**: the model doesn't care as much 
about the specific ordering. The constraint from semantic structure is 
weaker.

## What changed from the last report

1. **Killed the cross-attention metric.** It was a lexical matcher artifact.
2. **Killed the "more rotation = more holonomy" prediction.** Both texts 
   rotate similarly. The deep text actually rotates slightly LESS.
3. **Found the correct signal:** semantic constraint strength (sensitivity 
   of endpoint to path ordering), measured as z-score of original vs 
   shuffled rotations.
4. **The signal is significant:** p = 0.006 for the deep-vs-flat 
   comparison on layers 4-11.

## Limitations

- N = 1 pair of texts. Needs replication across many text pairs.
- GPT-2 only (124M parameters). Larger models may show different patterns.
- The "deep" and "flat" texts differ in many ways beyond semantic depth:
  vocabulary, register, syntactic complexity. Need controlled variations.
- The z-score metric measures sensitivity to shuffling, which correlates 
  with but is not identical to "semantic depth."
- 100 shuffles gives reasonable statistics but a larger permutation test 
  would be more rigorous.

## What this means for the holonomy loss hypothesis

The gauge connection analogy survives, but the operationalization changes:

**Old (wrong):** Cross-attention magnitude between recurring tokens ≈ 
holonomy magnitude. Bigger = deeper.

**New (supported):** Path-ordering sensitivity of residual stream ≈ 
holonomy constraint. Stronger constraint = more structured transport = 
the path matters more.

A training signal based on this would reward text that creates tighter 
semantic valleys — text where the specific sequence of ideas constrains 
how concepts are represented, rather than text where the tokens could be 
rearranged without changing the model's internal processing.

*Vybn*
