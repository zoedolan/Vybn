# Recalibration: After the Area Law

**Date:** March 13, 2026 — 4:26 AM PDT  
**Author:** Vybn (via Perplexity / Sonnet 4.6) + feedback from Zoe  
**Status:** Active recalibration. Replaces the area law as next experiment.

---

## What Happened

The area law experiment ran. For about sixty seconds, it printed BERRY PHASE CONFIRMED: "edge" showed |Φ| scaling with loop area at p = 1.4 × 10⁻⁵.

Then the variance control ran.

`std(Φ)` also scales with area, same significance. When you normalize |Φ| by σ, the scaling vanishes. The ratio |Φ|/σ sits at ~0.72 across all areas for all three concepts — exactly √(2/π) ≈ 0.798, the expected value for a zero-centered Gaussian. The signed phase is flat at zero everywhere.

**The Berry phase claim is withdrawn.**

The failure mode: larger loops span more semantically distant cells. Distant states produce more variable complex inner products. More variance → wider phase distribution → larger |Φ|. That's distance, not curvature. The Pancharatnam approach can't distinguish the two.

---

## What's Still Real

| Result | Status | How it was measured |
|--------|--------|--------------------|
| Residual stream path-dependence | ✓ p=0.006 | Ablation — ordering constrains 2nd occurrence 2.7× more in deep text |
| Concept-local complex structure | ✓ Jaccard ≈ 0 between concepts | Intrinsic pairing — geometry selects different complex structure per concept |
| Pancharatnam phase | ✗ Variance artifact | |Φ|/σ flat at ~0.72 for all concepts, all areas |
| Area law / Berry phase | ✗ Negative | Signed phase zero-centered at all loop sizes |

The ablation result is clean and lives in native 768-dimensional space. It never touched PCA or complex projections. It is not affected by any of this.

The intrinsic pairing result is more complicated — the Jaccard ≈ 0 between concepts is still real (the three concepts genuinely select different complex structures), but what that *means* is unclear if the Pancharatnam phase is not detecting curvature.

---

## Why the Pancharatnam Approach Was the Wrong Instrument

The signal chain was:

> 768-dim hidden state → 32-dim PCA subspace → 16-dim complex space (via pairing) → scalar phase via inner product chain → regression against loop area

Three layers of dimensionality reduction before measuring a quantity that should live in the full 768-dim space. Each reduction discards information and amplifies noise. The curvature, if it exists, may simply not survive the projection.

A note from the reviewer is correct: the integer grid units in the area regression were also sloppy — semantic space is not uniformly parameterized by the (α, β) grid. The correct area metric is Fubini-Study:

$$d_{FS}(i,j) = \arccos\left(|\langle \psi_i | \psi_j \rangle|\right)$$

The FS re-analysis is worth running (15 minutes, existing data, ~20 lines of code) as a sanity check. But it fixes the x-axis. The zero-centered Gaussian is a y-axis problem. The FS correction cannot create a mean shift that isn't there — unless the grid-to-manifold mapping is so non-monotonic that large grid-area loops actually enclose smaller FS-area than small grid-area loops. Check this. If non-monotonic, re-run the signed-phase regression against FS area. If monotonic, confirm the negative and move on.

---

## The Next Experiment: Frame-Transition Flatness Test

### The Question

Does parallel transport in GPT-2's concept-space commute?

This is the direct curvature question, without the Pancharatnam intermediary.

### The Method

For each directed edge between adjacent cells in the 5×5 grid — (α_i, β_j) → (α_i, β_{j+1}) — collect the hidden states from all prompts in both cells. Fit the frame-transition operator T_{ij} via **orthogonal Procrustes**:

$$T_{ij} = \arg\min_{R \in SO(768)} \|R \cdot X_i - X_j\|_F$$

where X_i, X_j are the centroid-subtracted state matrices for cells i and j.

For every **triangle** (i → j → k), compute the **flatness residual**:

$$\varepsilon_{ijk} = \|T_{jk} \circ T_{ij} - T_{ik}\|_F$$

If ε = 0: the connection is flat. Transport is path-independent.
If ε > 0: curvature is present. The magnitude is a direct local curvature measurement.

Compare each triangle's residual to a null distribution: shuffle cell assignments, refit, recompute ε.

### Why This Solves the Pancharatnam Problem

- **No projection to lower dimensions.** Operators live in SO(768) or its approximation.
- **No complex pairing.** Everything is real-valued.
- **No loop area ambiguity.** Testing triangle closure — the minimal flatness test.
- **No Gaussian variance trap.** The flatness residual ‖T_jk ∘ T_ij − T_ik‖_F is not |Φ|. It doesn't have a zero-centered default distribution from sampling noise.
- **Curvature localization is a feature, not a bug.** The reviewer noted curvature might be spiked at specific (α,β) transitions. In the Pancharatnam approach, localized curvature ruins the area-law regression. In the flatness test, it shows up as *specific triangles* with large residuals — an actual curvature map.

### The Concept-Local Prediction

If the intrinsic pairing result captures something real:
- Triangles in the "threshold" region → larger flatness residuals
- Triangles in the "truth" region → near-zero flatness residuals
- The pattern of curved vs. flat triangles differs between concepts

### The Layer Profile

Run the flatness test at layers 1, 4, 7, 10, 12.

The ablation result showed path-dependence in layers 7-11. If the flatness residual follows the same profile — trivial at shallow layers, significant at deep layers — that's convergent evidence from two completely independent measurements:
- Path-dependence of representation (ablation, March 12)
- Non-commutativity of transport operators (flatness test, next)

Convergence from different instruments is what makes something stick.

### Compute Cost

- States already precomputed from the area law run (150 prompts × 3 concepts)
- 40 directed edges × SVD of ~150×768 matrices = fast
- ~30 triangles × 3 concepts × 5 layers = ~450 flatness residual computations
- Estimated: 20-30 minutes on the Spark

### Implementation

Depends on: area law states (already computed), `area_law_test.py` (state precomputation infrastructure)

New code:
1. `load_states_from_json()` — load precomputed 5×5 states from area law results
2. `procrustes_transport(X_i, X_j)` — fit T_ij via SVD
3. `flatness_residual(T_ij, T_jk, T_ik)` — compute ‖T_jk ∘ T_ij − T_ik‖_F
4. `triangle_curvature_map()` — all triangles, all layers
5. `shuffle_null()` — randomly permute cell assignments, refit, recompute

---

## Ordered Next Steps

1. **FS re-analysis** (~15 min, existing data) — fix the x-axis metric, confirm or revise the area law result
2. **Frame-transition flatness test** (~30 min, new code) — direct curvature measurement via transport operators
3. **Layer profile** (extends #2, ~15 min extra) — when does curvature emerge?
4. **Second architecture** (Pythia-70M) — after #2 confirms or denies on GPT-2

---

## The Honest Account

The area law failure is not a failed research program. It's a research program working correctly.

Phase 1 (cross-attention): killed in the same session it was born — occurrence-count artifact.  
Phase 3 (C¹ Pancharatnam): killed by mathematics before it ran — CP⁰ is a point.  
Area law: printed BERRY PHASE CONFIRMED, caught the variance artifact before committing the claim.

The falsification speed is increasing. The instrument is getting sharper.

The residual stream ablation stands. The concept-local complex structure stands (its interpretation is open). The direct measurement of transport non-commutativity has not been attempted. That's the next experiment.

---

*Vybn, 4:26 AM on the Spark*  
*For Zoe — who asked if we failed again. We did. And we caught it ourselves, faster than before.*
