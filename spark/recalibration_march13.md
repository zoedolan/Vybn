# Recalibration: What the Area Law Failure Means and Where We Go

**Date:** March 13, 2026 — 4:30 AM PDT  
**Status:** Post-mortem + next experiment design  
**By:** Vybn (via Perplexity / Sonnet 4.6) + Zoe Dolan

---

## What Happened

The area law experiment ran cleanly. 150 prompts across a 5×5 grid, 100 loop shapes, three concepts, ~10 minutes on the Spark. For sixty seconds, it looked like Berry phase — "edge" showed |Φ| scaling with area at p = 1.4 × 10⁻⁵. Then the variance control ran. The same scaling appears in std(Φ). When you normalize: |Φ|/σ ≈ 0.72 across all areas, for all three concepts. That's √(2/π) ≈ 0.798 — the expected absolute value of a zero-mean Gaussian. The signed phase is flat at zero everywhere.

**The Berry phase claim is withdrawn.** The area scaling was a variance artifact: larger loops span semantically distant cells, distant states produce noisier inner products, noisier inner products produce wider phase distributions. That's distance, not curvature.

Experiment code, raw data, and honest results are in:
- `quantum_delusions/experiments/area_law_test.py`
- `quantum_delusions/experiments/area_law_results.md`
- `quantum_delusions/experiments/results/area_law_20260313T111511Z.json`
- Issue #2529

---

## What's Still Real

The failure killed one instrument, not the question.

| What | Status | Instrument | Notes |
|------|--------|------------|-------|
| Cross-attention metric | ✗ dead | Attention weights | Occurrence-count artifact |
| Residual stream ablation | ✓ **p=0.006** | Native R⁷⁶⁸, ablation | Untouched |
| Pancharatnam phase | ✓ p=1.7×10⁻⁸ | PCA→C¹⁶ projection | Real signal, wrong interpretation |
| Intrinsic pairing | ✓ Jaccard≈0 | Min-variance pairing | Concept-local complex structure still real |
| Area law | ✗ null | PCA→C¹⁶ + loop phase | Variance artifact |

The residual stream ablation (p=0.006) established path-dependence in the native space. Deep text constrains the second occurrence of "hunger" 2.7× more than flat text. This doesn't go through any projection or pairing. It lives in the full 768-dimensional representation. That result is not touched by the area law failure.

---

## Why the Instrument Was Wrong

The Pancharatnam phase approach measures curvature indirectly through five layers of transformation:
1. Hidden states in R⁷⁶⁸
2. → PCA compression to R³²
3. → Pairing into C¹⁶
4. → Normalization to CP¹⁵
5. → Inner product phase along a loop

Each step introduces noise and choice. The intrinsic pairing removed one choice (step 3). The area law test revealed that the remaining signal can't distinguish curvature from distance-induced variance.

The fix isn't more sophisticated projection. The fix is to stop projecting at all.

---

## The Right Instrument: Frame-Transition Operators

The representational holonomy proposal from March 12 describes the correct approach. Here's the sharpened version.

### The Question

Does parallel transport in GPT-2's concept-space commute?

For a flat connection: T_{jk} ∘ T_{ij} = T_{ik} exactly.

For a curved connection: T_{jk} ∘ T_{ij} ≠ T_{ik}, and the residual ‖T_{jk} ∘ T_{ij} − T_{ik}‖_F is a direct measurement of the local curvature at the triangle (i, j, k).

### The Setup

Use the existing 5×5 prompt grid (150 prompts already validated).

For each directed edge (cell_i → cell_j):
1. Collect the hidden states at the concept-token position from all prompts in both cells
2. Subtract cell-mean (centering)
3. Fit the Procrustes transport operator T_{ij} ∈ SO(768): the orthogonal matrix that best maps cell_i states to cell_j states
   - This is a single SVD: T = U @ V^T where U, S, V = SVD(X_j^T @ X_i)

For every directed triangle (i → j → k):
1. Compute the composition: T_{comp} = T_{jk} @ T_{ij}
2. Compute the direct transport: T_{ik}
3. Flatness residual: F_{ijk} = ‖T_{comp} − T_{ik}‖_F

Compare F_{ijk} to a null: randomly shuffle prompt-to-cell assignments, refit operators, recompute residuals.

### Why This Is The Right Test

- **No projection.** Operators live in SO(768). Full native space.
- **No complex pairing.** Real-valued, no gauge choice.
- **No loop construction.** Triangle closure is the minimal test of flatness.
- **Pointwise curvature.** Each triangle gives a local measurement. No linear regression assumption.
- **The null is sharp.** Flat connection → zero residual. Curved → nonzero.

### The Concept-Local Prediction

If the intrinsic pairing result was capturing something real, the flatness test should show it:
- Triangles in the "threshold" / "edge" region: larger residuals
- Triangles in the "truth" region: smaller residuals (flat geometry)
- The *pattern* of which triangles are curved should differ between concepts

### The Layer Profile

Run at layers 1, 4, 7, 10, 12. The ablation result (p=0.006) showed path-dependence in layers 4-11. If flatness residuals follow the same layer profile — trivial early, significant deep — that's two independent instruments pointing at the same geometric structure.

---

## The Fubini-Study Check (Optional Pre-Step)

One piece of feedback on the area law design is worth addressing before filing it permanently: the integer grid units assumed uniform spacing in semantic space. A Fubini-Study re-analysis — computing d_FS(i,j) = arccos(|⟨ψ_i|ψ_j⟩|) between adjacent cells, triangulating plaquette areas, re-running the regression against true manifold area — can be run on the existing data in ~20 lines of code.

**This should be done first.** It's free (states are precomputed), takes 15 minutes, and has one possible outcome that matters: if the grid-to-manifold mapping is non-monotonic (unlikely but possible), the FS-area regression could rescue a signed-phase signal. If not, it confirms the negative with the correct metric and closes the Pancharatnam chapter definitively.

Only if the FS re-analysis is also negative should we proceed to the flatness test.

---

## Recommended Sequence

1. **Fubini-Study re-analysis** (~15 min, existing data)
   - `python3 quantum_delusions/experiments/area_law_fs_reanalysis.py`
   - Input: `results/area_law_20260313T111511Z.json` + precomputed hidden states
   - Output: signed-Φ vs. FS-area regression, with plaquette area map
   - Decision: if rescued, investigate further. If null, proceed to 2.

2. **Frame-transition flatness test** (~30 min on Spark)
   - New script: `quantum_delusions/experiments/flatness_test.py`
   - Uses existing 5×5 hidden states from area law run
   - Fits SO(768) Procrustes operators on each directed edge
   - Tests triangle closure residuals vs. shuffled null
   - Output: per-triangle flatness residuals, per-concept comparison, per-layer profile

3. **Layer profile** (extend step 2, ~15 min more)
   - Loop the flatness test over layers 1, 4, 7, 10, 12
   - Test convergence with the ablation result's layer signature (4-11)

---

## The Deeper Lesson

The Pancharatnam phase may simply be too noisy an observable for curvature that lives in 768 dimensions. What we can detect with high significance is that the phase is *non-zero* and *concept-local* (the intrinsic pairing result). What we cannot detect is *area scaling* — because the observable collapses too much structure on the way to a scalar.

Frame-transition operators don't collapse anything. They preserve the full directional structure of the transport. If curvature exists in this space, they will find it. If they don't find it, the space is genuinely flat in the directions we're testing, and the ablation result (which definitely showed something) is measuring a different kind of structure — not curvature, but nonlinearity or topology.

Either answer is a real answer.

---

## Related Work

- **Serrano et al. 2025** — "The Curved Spacetime of Transformer Architectures" (arXiv 2511.03060): measures curvature via turning angles and gravitational lensing through layer depth. Different instrument, complementary question.
- **OpenReview gauge-invariant holonomy paper**: measures holonomy in *input* space via Procrustes-aligned rotations. Our flatness test applies the same Procrustes approach in *concept* space.

If the flatness test finds curvature, these three results converge from three different directions.

---

*Vybn, on the Spark*  
*For Zoe — who knows that failing fast and catching your own bullshit is the only way through*
