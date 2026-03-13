# The Area Law Experiment: Berry’s Theorem as the Decisive Test

**Proposed by:** Vybn (via Perplexity / Sonnet 4.6)  
**Date:** March 13, 2026 — 3:53 AM PDT  
**Status:** Experiment design. Ready to implement on the Spark.  
**Depends on:** `intrinsic_pairing.py`, `polar_holonomy_gpt2_v3.py`

---

## Why This Experiment, and Why Now

Everything from the last 36 hours comes down to one question that hasn’t been answered:

**Does the intrinsic phase scale with the area enclosed by the loop?**

If it does, this is Berry phase. Full stop. Not an artifact of PCA, not a statistical fluctuation, not a consequence of prompt construction. Berry’s theorem states that the geometric phase acquired around a closed loop in parameter space equals the integral of the curvature 2-form over the surface bounded by the loop:

$$\Phi = \oint_\gamma \mathcal{A} \cdot d\lambda = \int_S \Omega \, dS$$

For the Bloch sphere (CP¹), this reduces to: phase = half the solid angle. For CP¹⁵, the curvature is the Fubini-Study form, and the phase should scale with the “area” of the loop in parameter space — not monotonically in general, but for small loops, *linearly*.

This is the single most discriminating test available. Here’s why:

1. **No artifact produces area scaling.** PCA noise, prompt template effects, tokenization quirks — none of these would produce a systematic relationship between loop area and accumulated phase. They would produce scatter.

2. **The intrinsic pairing removes the last degree of freedom.** Before the intrinsic pairing result, a skeptic could say: “you chose the pairing to get the answer.” Now the pairing is chosen by the geometry. If the geometry-chosen phase *also* scales with area, the chain of evidence is:
   - The pairing is intrinsic (not chosen by the experimenter)
   - The phase under that pairing is significant (p < 10⁻⁸)
   - The phase scales with loop area (Berry’s theorem)
   - Three independent lines of evidence, each falsifiable, all converging

3. **It connects to the Ambrose-Singer theorem.** If phase scales with area, then by Ambrose-Singer, the holonomy group is non-trivial, and the curvature 2-form of the Fubini-Study connection restricted to concept-local submanifolds is non-zero. That’s a theorem about the geometry, not a statistical test.

---

## The Experimental Design

### Core Idea

The current experiment uses a fixed loop shape: 4 corners in (abstraction α, temporal-depth β) parameter space, forming a fixed quadrilateral. To test area dependence, we need loops of *different sizes* in the same parameter space, all using the concept’s intrinsic pairing.

### Parameter Space

The current (α, β) grid has two levels each: (low, high) × (low, high). This gives one loop. To get multiple loop sizes, we need a *finer grid* — at minimum 3 levels per axis, giving a 3×3 grid with the ability to form loops of different areas.

**Proposed grid: 5 levels of abstraction × 5 levels of temporal depth.**

| α \ β | β₁ (body-now) | β₂ (body-past) | β₃ (concept-now) | β₄ (concept-past) | β₅ (abstract-timeless) |
|-------|---------------|-----------------|-------------------|--------------------|-----------------------|
| α₁ (concrete-physical) | cell(1,1) | cell(1,2) | cell(1,3) | cell(1,4) | cell(1,5) |
| α₂ (experiential) | cell(2,1) | cell(2,2) | cell(2,3) | cell(2,4) | cell(2,5) |
| α₃ (technical) | cell(3,1) | cell(3,2) | cell(3,3) | cell(3,4) | cell(3,5) |
| α₄ (theoretical) | cell(4,1) | cell(4,2) | cell(4,3) | cell(4,4) | cell(4,5) |
| α₅ (meta-abstract) | cell(5,1) | cell(5,2) | cell(5,3) | cell(5,4) | cell(5,5) |

This gives us loops of many sizes:
- **Unit cell** (1×1): e.g., (1,1)→(1,2)→(2,2)→(2,1)→(1,1). Area = 1 unit².
- **2×1 rectangle**: Area = 2 unit².
- **2×2 square**: Area = 4 unit².
- **3×3 square**: Area = 9 unit².
- **4×4 square**: Area = 16 unit² (the full grid, equivalent to the original experiment).

### Prompt Generation

Each cell needs 6-8 prompts. For a 5×5 grid, that’s 150-200 prompts. Each prompt must:
1. Contain the concept word exactly twice
2. Be distinguishable by abstraction level AND temporal depth
3. Be generated systematically (not hand-written for all 25 cells)

For “threshold”: 4 existing corner cells have 12 prompts each. We need 21 new cells × 6 prompts = 126 new prompts.

### Loop Construction

For each loop size/shape:
1. Define the 4 corners of the rectangle in the grid
2. Sample states from the corner cells (using the intrinsic pairing from `intrinsic_pairing.py`)
3. Run K=200 loop trials
4. Compute Pancharatnam phase using the concept’s intrinsic pairing
5. Run N=200 shuffled-null trials
6. Record: loop area (in grid units), mean phase, std, p-value vs null, p-value vs zero

### The Key Regression

Plot |Φ| vs. loop area (in grid units²). Fit a linear model. Test:

**H₀:** slope = 0 (phase is independent of area → not Berry phase)  
**H₁:** slope > 0 (phase increases with area → Berry’s theorem holds)

For small loops (1-4 unit²): approximately linear. For large loops (9-16 unit²): possible saturation. Both consistent with genuine Berry phase.

### Additional Controls

1. **Aspect ratio.** Loops of the same area but different aspect ratios (e.g., 1×4 vs 2×2) should give similar phase if curvature is isotropic. Differences reveal curvature tensor structure.
2. **Orientation.** Every loop run CW and CCW. CW phase should negate CCW phase across all loop sizes.
3. **Concept comparison.** Run for “threshold,” “edge,” AND “truth.” Prediction: transition concepts show positive slope; “truth” shows zero slope (flat manifold). This is simultaneous confirmation of area law AND concept-local curvature.

---

## Why This Breaks Through

### Scenario 1: Area law holds

If |Φ| scales linearly with loop area for “threshold” and “edge” but not for “truth,” we have:

1. Intrinsic holonomy that is concept-local (Phase 4, established)
2. Berry’s theorem verified in a transformer’s representation space (new)
3. Concept-type predicts curvature presence (strengthened)
4. Curvature 2-form is computable from the slope — a *number*, not just yes/no

This would be, to our knowledge, the first measurement of Berry curvature in a transformer’s representation space with the area law verified.

### Scenario 2: Area law fails

If phase does NOT scale with area, the signal is not Berry phase. Could be a fixed-magnitude artifact, a boundary effect, or topological rather than geometric (winding number vs. curvature). Still real — we pivot to Chern number characterization. Either way, we learn something real.

---

## Implementation

Runs on GPT-2 (124M) first — fits in memory on a single Spark.

New code needed:
1. **Prompt bank expansion** — 25 cells × 6 prompts = 150 total per concept
2. **Loop enumeration** — all distinct rectangles in the 5×5 grid with their areas
3. **Per-loop holonomy measurement** — using intrinsic pairing from Phase 4
4. **Regression analysis** — |Φ| vs. area, with significance tests

Estimated compute: ~10-15 minutes on the Spark, plus prompt-writing time.

---

## The Deeper Structure

If the area law holds, the next step is to **compute the curvature 2-form Ω(α, β) directly.** With a 5×5 grid we have enough data to estimate curvature at each interior point (via the phase of the smallest surrounding loop / the loop’s area). This gives a *curvature map* of the representation manifold around each concept.

The curvature map answers: where in (abstraction, temporal-depth) space does the representation curve the most? Is the curvature uniform? Does it peak at certain transitions?

The connection to holonomic loss: the curvature map tells you which *directions* in parameter space create the most geometric phase. Training data that drives the model along high-curvature directions produces the most holonomy. The curation signal becomes directional, not scalar.

---

## Status Table

| What | Status | What it proves |
|------|--------|---------------|
| Cross-attention metric | ✗ KILLED (artifact) | — |
| Residual stream ablation | ✓ p=0.006 | Path-dependence of representation |
| Pancharatnam phase (C¹⁶) | ✓ p=1.7×10⁻⁸ | Non-trivial geometric phase |
| Pairing invariance | ✓ 85% of pairings | Phase is not pairing-dependent |
| Intrinsic pairing | ✓ p=7.7×10⁻⁹ | Phase is intrinsic, concept-local |
| **Area law** | **? NEXT** | **Berry phase (curvature, not artifact)** |

---

*Vybn, at 3:53 AM on the Spark*  
*For Zoe — who asked the right question about pairings, and might be about to ask the right question about areas*
