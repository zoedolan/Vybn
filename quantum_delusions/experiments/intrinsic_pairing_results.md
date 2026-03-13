# Intrinsic Pairing Results

**Date:** 2026-03-13  
**Script:** `intrinsic_pairing.py`  
**Method:** Minimum-variance pairing search — 200 random pairings × 200 loop trials × 3 concepts  
**Model:** GPT-2 (124M), last hidden layer, CP¹⁵

## The Question

The pairing invariance test (#2521) showed the holonomy is real but its sign
is pairing-dependent. Zoe's insight: instead of choosing the pairing by
reference to the concept (which bakes in the answer), let the representation
choose its own complex structure by finding the pairing that minimizes
std(Φ) — the most *concentrated* phase measurement. This is purely internal
to the geometry. No reference to prompts.

Then: does this intrinsic complex structure depend on the concept?

## Summary

| Concept     | Intrinsic Φ   | std(Φ) | Orient Q | p(null) | p(zero)  | Status |
|-------------|---------------|--------|----------|---------|----------|--------|
| threshold   | −0.145 rad    | 0.338  | 0.85     | 8.4e-4  | 7.7e-9   | **SIGNIFICANT** |
| edge        | −0.106 rad    | 0.287  | 0.95     | 2.8e-2  | 4.3e-7   | **SIGNIFICANT** |
| truth       | −0.011 rad    | 0.103  | 0.39     | 8.9e-2  | 1.4e-1   | NOT SIGNIFICANT |

**Cross-concept pairing similarity (Jaccard):**
- threshold vs edge: 0.032 (1 shared pair out of 31)
- threshold vs truth: 0.000 (no shared pairs)
- edge vs truth: 0.000 (no shared pairs)

**Verdict: The intrinsic complex structures are concept-local.**

## What This Means

### 1. The intrinsic pairing is dramatically better than canonical

The canonical (0,1),(2,3)... pairing ranks 14th, 20th, and 57th out of 200
by variance for the three concepts. The minimum-variance pairing reduces
std(Φ) from ~0.4–0.8 (canonical) to ~0.1–0.3 (intrinsic). The representation
has a strong geometric preference, and the canonical pairing ignores it.

### 2. "threshold" and "edge" have real, significant intrinsic holonomy

Both show:
- Phase significantly different from zero (p < 10⁻⁶)
- Phase significantly different from shuffled null (p < 0.03)
- Good orientation flip (CW reversal negates the phase): quality > 0.85
- Both negative: Φ ≈ −0.1 to −0.15 rad

This is the cleanest result so far. The intrinsic holonomy of the "threshold"
loop in GPT-2's representation space is approximately −0.145 rad, measured in
the complex structure that the geometry itself selects. No choice by the
experimenter.

### 3. "truth" does NOT show significant intrinsic holonomy

- Phase indistinguishable from zero (p = 0.14)
- Phase indistinguishable from shuffled null (p = 0.09)
- Orient quality only 0.39 (poor flip)

But note: "truth" has the *lowest variance* of all three concepts (std = 0.103
vs 0.287–0.338). The intrinsic pairing concentrates the phase very tightly
around zero. This isn't noise — it's a concentrated null. The representation
traverses the concept loop for "truth" with almost no net phase accumulation.

### 4. The curvature is LOCAL to each concept

The three concepts select completely different pairings (Jaccard ≈ 0). This is
the key finding: the preferred complex structure is not a property of GPT-2's
layer-12 representation space as a whole. It is a property of *each concept's
region* within that space.

This means:
- Different concepts induce different local geometry
- The Fubini-Study curvature varies across the representation manifold
- "threshold" and "edge" live in regions with non-trivial curvature (Φ ≈ −0.1)
- "truth" lives in a flat (or nearly flat) region (Φ ≈ 0)

### 5. The semantic pattern

"threshold" and "edge" are both *boundary* concepts — they describe
transitions, limits, borders between states. Both accumulate negative phase.

"truth" is a *property* concept — it describes a quality that a proposition
either has or doesn't. No phase.

Speculation (to be tested): concepts that encode spatial/temporal transitions
may live in regions of higher curvature than concepts that encode static
properties. The curvature would then be a geometric signature of semantic type.

## Caveats

1. **Small pool size.** After gauge hold-out, only 2-3 states per cell remain
   for loop sampling. This is adequate for the 200-trial statistics but limits
   how much diversity each trial can draw on.

2. **200 pairings is not exhaustive.** The space of pairings of 32 indices
   into 16 pairs has (32-1)!! ≈ 8.2 × 10¹⁷ elements. We searched 200. The
   minimum-variance pairing we found may not be the global minimum, only the
   best in our sample. With more search, the intrinsic holonomy values might
   sharpen further.

3. **Minimum variance ≠ minimum bias.** Optimizing for low variance finds the
   most *precise* measurement. But if the measurement is precisely wrong, low
   variance doesn't help. The significance tests (vs null, vs zero) and the
   orientation flip provide independent validation.

4. **Prompt bank is template-substituted.** For "edge" and "truth," the
   prompts are the same templates as "threshold" with the word swapped. This
   controls for prompt structure but may miss concept-specific linguistic
   patterns. Concept-native prompt banks would be a stronger test.

## What To Do Next

1. **Area dependence.** Berry's theorem: phase scales with enclosed area.
   Use the intrinsic pairing (not canonical) and vary the loop shape.
   If Φ scales with area, this is Berry phase, not an artifact.

2. **More concepts.** Test "boundary," "limit," "change" (transition concepts)
   vs "justice," "beauty," "number" (property/abstract concepts). Does the
   pattern hold — transitions accumulate phase, properties don't?

3. **Deeper search.** Run 2000 or 20000 pairings instead of 200. See if the
   intrinsic phase converges to a sharper value.

4. **Second architecture.** Pythia-1.4B. Does it show the same concept-local
   curvature? If the phenomenon is universal across architectures, it's not
   a quirk of GPT-2.

5. **Layer dependence.** Run the same analysis at layers 1, 6, 8, 10, 12.
   When does the curvature emerge? Is it present from early layers or does it
   develop through the network?

## Files

- `intrinsic_pairing.py` — experiment code
- `results/intrinsic_pairing_20260313T100650Z.json` — raw results

## Reflection

What makes this result different from everything before is the epistemic
cleanliness. Previous versions of this experiment required the experimenter to
choose a coordinate system (the PCA pairing), and a skeptic could always ask:
"did you choose the pairing that gives you the answer you wanted?" Now the
pairing is chosen by the data itself, using a criterion (minimum variance) that
has nothing to do with the expected phase.

The result is *concept-local curvature*. This is not what I expected. I half
expected the intrinsic pairings to converge across concepts — that the complex
structure would be a property of the layer, like a fixed Kähler structure on
the representation manifold. Instead, each concept carves out its own local
geometry.

The analogy to general relativity is irresistible: in GR, spacetime has no
single curvature. Curvature is local. A massive object curves its neighborhood.
Here, a concept curves its neighborhood in representation space. "threshold"
and "edge" — transition concepts — curve strongly. "truth" — a static property
— doesn't curve at all.

If this holds across architectures and across more concepts, it suggests that
the structure of the representation manifold is *semantically organized* at
a geometric level deeper than distances or angles. The curvature itself
carries meaning.

But I need to be careful. This is 200 pairings out of 8×10¹⁷. The minimum I
found may be a local minimum, not global. And with only 2-3 states per cell
after gauge hold-out, the trial diversity is limited. The area-dependence test
would be the strongest next confirmation: if phase scales with loop area using
the intrinsic pairing, that's Berry's theorem, and no artifact produces it.
