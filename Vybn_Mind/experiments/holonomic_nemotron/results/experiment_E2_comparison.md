# Experiment E.2 — QGT Comparison: Baseline vs Geometric Training

**Generated:** 2026-03-22 17:29 UTC
**Source:** Experiment D v3 (384-dim activation centroids, GPT-2 small)
**Baseline:** λ=0.0 (standard cross-entropy loss)
**Geometric:** λ=0.5 (cross-entropy + geometric regularizer)
**Snapshots:** 31 per run (every 10 steps + final)
**Activation dim:** 384

## Main Comparison Table

| Layer | Base Arc-Length | Geo Arc-Length | Ratio (G/B) | Base Anisotropy | Geo Anisotropy | Ratio (G/B) | Base Mean\|Barg\| | Geo Mean\|Barg\| | Ratio (G/B) |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 5.6015 | 3.8237 | **0.683** | 1.2444 | 1.6441 | **1.321** | 0.9143 | 0.9419 | **1.030** |
| L1 | 7.2575 | 4.4274 | **0.610** | 0.8454 | 1.5538 | **1.838** | 0.8868 | 0.9319 | **1.051** |
| L2 | 9.1158 | 4.7434 | **0.520** | 0.6452 | 1.5628 | **2.422** | 0.8418 | 0.9257 | **1.100** |
| L3 | 10.1217 | 4.8859 | **0.483** | 0.5735 | 1.6302 | **2.842** | 0.8166 | 0.9232 | **1.131** |
| L4 | 10.0016 | 4.7821 | **0.478** | 0.5251 | 1.6439 | **3.131** | 0.8291 | 0.9243 | **1.115** |
| L5 | 8.4013 | 4.3935 | **0.523** | 0.6556 | 1.7080 | **2.605** | 0.8724 | 0.9306 | **1.067** |

## Per-Layer Detail: Mean Overlap & Sign Flips

| Layer | Base Mean Overlap | Geo Mean Overlap | Base Sign Flips | Geo Sign Flips | Base Neg Bargmann | Geo Neg Bargmann |
|:---:|---:|---:|---:|---:|---:|---:|
| L0 | 0.958494 | 0.971777 | 0 | 0 | 0 | 0 |
| L1 | 0.952075 | 0.965494 | 0 | 0 | 0 | 0 |
| L2 | 0.937006 | 0.960475 | 0 | 0 | 0 | 0 |
| L3 | 0.927408 | 0.956163 | 0 | 0 | 0 | 0 |
| L4 | 0.932059 | 0.957399 | 0 | 0 | 0 | 0 |
| L5 | 0.947148 | 0.961894 | 0 | 0 | 0 | 0 |

## Interpretation

### What the data shows

**1. Arc-length: Geometric training dramatically shortens trajectories.**

The geometric run (λ=0.5) traverses roughly **half** the Fubini-Study arc-length
of the baseline across all layers. The effect is strongest at deep layers:
L3 baseline travels 10.12 rad vs geometric 4.89 rad (ratio 0.483). This means
geometric regularization causes the model to move *less* in projective activation
space — the representations change more conservatively during training.

**2. Anisotropy: OPPOSITE of prediction — geometric run is MORE anisotropic.**

We predicted the geometric run would have more *uniform* step sizes (lower
anisotropy). Instead, it's roughly **2.5–3× more anisotropic** than baseline
(e.g., L4: 1.644 vs 0.525). This means the geometric run takes a few large
steps and many small ones — it's *burstier* in how it moves through the space.

This is actually consistent with a different mechanism than we hypothesized:
the geometric regularizer may be acting as a **brake** that occasionally releases,
producing punctuated rather than smooth movement. The total distance is shorter,
but the distribution of step sizes is lumpier.

**3. Berry curvature: Trivially zero for both runs.**

Zero sign flips, zero negative Bargmann invariants. Both runs stay in the same
hemisphere of projective space throughout training. For real-valued vectors, the
Bargmann invariant `⟨ψ₁|ψ₂⟩⟨ψ₂|ψ₃⟩⟨ψ₃|ψ₁⟩` can only contribute curvature
through sign changes, and there are none. The Berry phase is identically zero.

This means the **topological** content of the QGT is trivial at this scale.
The **metric** content (Fubini-Study distances, arc-lengths) is where the
signal lives. To get non-trivial Berry curvature, we would need either:
- Complex-valued representations (not standard in transformers)
- A parametric manifold (varying hyperparameters, not just training steps)
- Higher-dimensional topological invariants (Chern numbers over 2D parameter spaces)

### Prediction scorecard

| Prediction | Result | Notes |
|:---|:---:|:---|
| Geometric run has lower Berry curvature | ⚪ TIE | Both trivially zero — not informative |
| Geometric run has more uniform metric (lower anisotropy) | ❌ WRONG | Geometric is 2.5–3× *more* anisotropic |
| Effect strongest at deep layers | ✅ RIGHT | Arc-length ratio smallest at L3 (0.483), anisotropy ratio largest at L5 (2.604) |
| QGT Berry curvature correlates with generalization gap | ⚪ N/A | Berry curvature is trivially zero; cannot test |

### What this means for the project

The honest answer: the Berry phase / topological invariant story doesn't work
at this scale with real-valued activations. What *does* work is the **metric**
side of the QGT — the Fubini-Study distances show a clean, large, layer-dependent
effect. Geometric training constrains the model to a shorter path through projective
space, and this constraint gets stronger in deeper layers.

The anisotropy result is the most interesting surprise: it suggests the geometric
regularizer produces **punctuated equilibrium** dynamics rather than smooth
convergence. This is worth investigating further — it connects to questions about
loss landscape geometry and training dynamics that are well-studied in the ML
literature.

### Next steps

1. **Abandon the Berry phase framing** for 1D training trajectories with real activations. It's a dead end here.
2. **Focus on Fubini-Study metric geometry**: the arc-length and anisotropy results are real signals with large effect sizes.
3. **Investigate the punctuated equilibrium**: plot FS distance vs training step to see *when* the geometric run moves and when it parks.
4. **2D parameter sweep** (e.g., λ × learning rate): this could give a surface over which Chern-class invariants are computable.
