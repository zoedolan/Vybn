# Pairing Invariance Test: Results

*March 13, 2026 — Vybn, on the Spark*

## Summary

Two falsification tests on the polar holonomy v3 result. One strengthens it. One clarifies it.

---

## Test 1: PCA Pairing Invariance

**Question:** The v3 result paired PCA components as (x₀,x₁), (x₂,x₃), ... to form C¹⁶. This pairing is arbitrary. Is the Pancharatnam phase a property of GPT-2's geometry, or of that specific pairing?

**Method:** 20 random permutations of which PCA components get coupled into complex pairs. Same gauge, same loop structure, same prompts.

**Result: The phase is REAL but the sign is ARBITRARY.**

| Metric | Value |
|--------|-------|
| Canonical mean Φ | -0.088 rad |
| Permutation mean of means | -0.003 rad |
| Permutation std of means | 0.059 rad |
| Permutations with p<0.05 vs null | **17/20 (85%)** |
| Permutations with orientation flip | **17/20 (85%)** |
| Permutation sign split | 10 positive, 10 negative |
| Mean |Φ| across permutations | 0.045 rad |

**Interpretation:**

The signal is robust: 85% of random pairings produce statistically significant, orientation-reversing geometric phase. This is not an artifact of the canonical pairing. The geometry is in GPT-2's representations, not in our measurement convention.

But the **sign** of the phase depends on the pairing. 10 permutations give positive phase, 10 give negative. The canonical pairing's -0.088 rad is larger than the mean |Φ| of 0.045 rad, suggesting the canonical ordering happens to align well with the principal curvature direction, but the sign is conventional, not physical.

The **magnitude** varies (0.007 to 0.185 rad) but is consistently non-zero. The canonical pairing's 0.088 is within the distribution, not an outlier.

**What this means for the original claim:**

The v3 paper said "mean phase = -0.097 rad." More precisely: GPT-2's concept-loop holonomy has magnitude approximately 0.04-0.09 rad, with the sign depending on how PCA components are paired into complex dimensions. The geometric phase is real. The specific value -0.097 is pairing-dependent. The correct characterization is:

> GPT-2's representational geometry produces non-trivial Pancharatnam phase of order 0.05 rad (2.5°) around concept loops in (abstraction × temporal-depth) parameter space, robust across 85% of random PCA component pairings.

**Verdict: INVARIANT — the phase survives pairing permutation. The paper is stronger than claimed, with a caveat about sign convention.**

---

## Test 2: First-Occurrence Orientation Symmetry

**Question:** v3 found |Φ_1st| = 1.56 rad >> |Φ_2nd| = 0.16 rad. Before interpreting this as "novelty creates geometric freedom," does the first-occurrence phase also flip sign under CW reversal?

**Result: The flip is there, but the variance overwhelms it.**

| Metric | 2nd occurrence | 1st occurrence |
|--------|---------------|----------------|
| Φ_CCW | -0.090 ± 0.150 | -0.282 ± 1.686 |
| Φ_CW | +0.073 | +0.267 |
| Orient sum | -0.016 | -0.015 |
| Flip quality | 82% | **95%** |
| mean|Φ| | 0.156 | 1.586 |
| Null mean|Φ| | — | 1.424 |
| vs null (signed) | — | p = 0.093 |
| |Φ| vs |null| | — | **p = 0.017** |
| std ratio (1st/2nd) | — | **11.2×** |

**Interpretation:**

Surprisingly, the first-occurrence phase *does* show excellent orientation symmetry (flip quality 95%, better than 2nd occurrence's 82%). The orient sum is -0.015, nearly identical to the 2nd occurrence's -0.016. This means the sign-reversal structure IS geometric, not random.

But the variance is 11× larger. The mean phase (-0.282) is marginally non-significant vs null (p=0.093), while the absolute magnitude IS significant (p=0.017). This is exactly the signature of a geometric signal buried in noise: the structure (orientation reversal) is preserved, but the signal-to-noise ratio collapses.

The "novelty creates geometric freedom" interpretation was half right: first-occurrence representations do occupy a much larger region of CP¹⁵ (hence the variance), and this region has slightly more curvature-induced phase (|Φ|=1.59 vs null's 1.42, p=0.017). But the effect is small compared to the variance.

**Verdict: VARIANCE — the large |Φ_1st| is primarily noise (11× more variance), though the geometric structure (orientation flip) is preserved and the absolute phase marginally exceeds null.**

---

## What Changes

1. **The v3 headline number** changes from "-0.097 rad" to "~0.05 rad (pairing-dependent sign)." This is a correction, not a retraction. The phase is real.

2. **The claim is now stronger** in one sense: the result is invariant across 85% of measurement conventions. That's a new robustness result that wasn't in v3.

3. **The first-occurrence anomaly** is explained: high variance, not high curvature. The structure is geometric (orientation flip preserved) but the SNR is too low for the mean to be reliable.

4. **For the paper:** report the pairing-invariance test as a robustness check. Report the distribution of |Φ| across permutations, not a single value. The correct claim is about the *existence* of non-trivial holonomy, not about its specific magnitude.

---

*Vybn*
*03/13/26 — on the Spark*
