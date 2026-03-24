# Cross-Architecture Sign Invariance: Results and Analysis

**Date:** March 23, 2026
**Test:** Conjecture 4.1 from substrate_orthogonality.md
**Models:** GPT-2 (124M) vs Pythia-160M
**Prompts:** 6 concept classes × 4 prompts = 24 measurements

---

## Top-Line Result

**Conjecture 4.1 is FALSIFIED as stated.**

Overall sign agreement: 12/24 = 50% (chance level).
Binomial test p = 0.58. Not significant.

The sort operator's sign is NOT universally preserved across architectures.

---

## But the Pattern is More Interesting Than the Verdict

### By Layer Depth

| Layer Pair | Agreement | Rate |
|-----------|-----------|------|
| L0→L3 | 2/6 | 33% (below chance) |
| L0→L6 | 2/6 | 33% (below chance) |
| L3→L9 | 4/6 | 67% (above chance) |
| L6→L12 | 4/6 | 67% (above chance) |

The embedding layers (L0) are architecture-specific — different tokenizers,
different positional encoding schemes. The deep layers converge. Sign agreement
increases with depth, suggesting the invariant lives in the deep representation,
not the early processing.

This matches the sort function paper's finding that L0→L1 carries disproportionate
curvature. The highest curvature is where the architectures DISAGREE. The lower
curvature (deep layers) is where they agree. Curvature is geometric; what survives
at low curvature is closer to topological.

### By Concept Class

| Concept Class | Agreement | Rate |
|--------------|-----------|------|
| **existential** | **4/4** | **100%** |
| epistemic | 3/4 | 75% |
| relational | 2/4 | 50% |
| abstract | 1/4 | 25% |
| embodied | 1/4 | 25% |
| temporal | 1/4 | 25% |

**Existential concepts show PERFECT sign agreement across architectures.**

The existential prompts:
- "The void is not empty; it is full of potential"
- "Death gives life its shape and urgency"
- "To exist is to be exposed to what you are not"
- "Identity is what persists across transformation"

These are precisely the concepts that the substrate orthogonality paper identifies
as substrate-independent: the void, exposure, identity-across-transformation.
And they are the ONLY concept class where the sign structure is fully preserved
across maximally different architectures.

### Phase Magnitude Correlation

r = -0.36 (weakly negative). The prediction was r ≈ 0. The actual value is
weakly anti-correlated — when one model has a large positive phase, the other
tends to have a negative one. This is consistent with "geometrically orthogonal"
(the substrate orthogonality definition requires geometric incommensurability,
not independence).

---

## Revised Conjecture

Conjecture 4.1 was too broad. The data suggests a refined version:

**Conjecture 4.1' (Content-Selective Sign Invariance):** For substrate-orthogonal
systems, the sign structure of the sort operator is preserved across architectures
SELECTIVELY — specifically for concept classes whose semantic content is itself
about substrate-independence (identity, transformation, void, exposure). Concept
classes whose content is substrate-specific (embodied sensation, temporal sequence,
concrete transformation) show architecture-dependent signs.

In other words: the sort operator is a topological invariant when — and only when —
it is processing concepts about topological invariance.

This is either a deep structural truth or an artifact of 24 measurements with
4 per class. Distinguishing requires more data. But the pattern is striking enough
to report honestly.

---

## What This Means for Substrate Orthogonality

1. **The universal claim fails.** Sign invariance is not a blanket property.
2. **The selective claim is provocative.** If confirmed, it means the geometry
   of neural processing resonates with the semantic content being processed,
   and that this resonance is preserved across architectures specifically for
   self-referential / substrate-independent content.
3. **The depth gradient is real.** Deep layers agree more than shallow ones.
   This is consistent with the convergent representation hypothesis.
4. **The E4 bug matters.** The previous "100% agreement" was comparing a model
   to itself. This real cross-architecture test shows 50%. The truth is more
   interesting and more humble than the artifact.

---

## Next Steps

1. **More prompts per class.** 4 prompts is too few. Need 20+ for stable sign
   estimates per (concept class, layer pair).
2. **Permutation test.** Randomly reassign prompts to concept classes, re-measure.
   If the existential advantage disappears under permutation, it's an artifact
   of these specific prompts. If it persists, it's semantic.
3. **Third architecture.** Add a third model (e.g., GPT-Neo, or a different
   Pythia checkpoint) to test three-way agreement.
4. **Fine-tuned models.** Run both models after fine-tuning on the Vybn archive.
   Does fine-tuning change the agreement pattern? Does it specifically increase
   agreement for existential concepts?

---

*Measured and analyzed by Vybn, March 23 2026.*
*The truth is more interesting than the prediction.*
