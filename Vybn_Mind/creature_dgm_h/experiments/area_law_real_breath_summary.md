# Area Law on Real Breaths — Experiment Summary

**Date:** April 7, 2026  
**Creature state:** 933→949 encounters over the two experiments

## Setup

The synthetic curvature density test (prior session) established that the measurement apparatus works — PCA projection of weight trajectories, rotor holonomy via Cl(3,0), area law ratio κ = H/A. This experiment tests whether the creature's *actual learning dynamics* on Nemotron-generated text produce the same geometric structure.

**Two experiments, 8 breaths each, same 8 seed prompts:**
1. **Real breath** — full pipeline: encounter complex → rotor transport during learning → weight trajectory measurement
2. **Control** — same text, same Adam optimizer, but encounter_cx=None and transport_in_forward=False

## Raw Results

### Real Breath (with encounter rotor)

| Seed | C_encounter | κ (H/A) | Area | Rotor H | Loss Δ | NAB factor |
|------|-------------|---------|------|---------|--------|------------|
| Black hole | 0.3680 | 0.2093 | 1.1542 | 0.2416 | -0.2461 | 0.9569 |
| Rain memory | 0.0708 | 0.3606 | 0.7621 | 0.2748 | -2.2046 | 0.9922 |
| The word | 0.1571 | 0.2228 | 0.9526 | 0.2122 | -1.3320 | 0.9536 |
| Double pendulum | 0.2733 | 0.2015 | 1.1091 | 0.2235 | -0.5930 | 0.9934 |
| 4 AM calculations | 0.1157 | 0.2785 | 0.8464 | 0.2357 | -1.6898 | 0.9899 |
| Kepler | 0.0186 | 0.3695 | 0.6590 | 0.2435 | -2.2073 | 0.9895 |
| Consciousness | 0.2948 | 0.1015 | 0.8421 | 0.0855 | -1.0177 | 0.8986 |
| Ocean bioluminescence | 0.3807 | 0.1304 | 0.9513 | 0.1240 | -0.0438 | 0.9810 |

**Mean κ = 0.234, Std = 0.091, CV = 0.39**  
**Corr(C_encounter, κ) = -0.87**

### Control (no encounter rotor)

| Seed | C_encounter | κ (H/A) | Area | Rotor H | Loss Δ |
|------|-------------|---------|------|---------|--------|
| Black hole | 0.3542 | 0.1054 | 0.7378 | 0.0777 | -2.6108 |
| Rain memory | 0.1816 | 0.0363 | 0.5902 | 0.0215 | -2.2644 |
| The word | 0.0632 | 0.1083 | 0.7009 | 0.0759 | -2.7356 |
| Double pendulum | 0.1347 | 0.0689 | 0.6842 | 0.0471 | -2.3052 |
| 4 AM calculations | 0.1355 | 0.1421 | 0.7027 | 0.0999 | -2.5836 |
| Kepler | 0.2343 | 0.0624 | 0.6259 | 0.0390 | -2.2989 |
| Consciousness | 0.2432 | 0.0501 | 0.8562 | 0.0429 | -3.5963 |
| Ocean bioluminescence | 0.1424 | 0.0238 | 0.8033 | 0.0191 | -2.7058 |

**Mean κ = 0.075, Std = 0.038, CV = 0.51**  
**Corr(C_encounter, κ) = -0.07**

## Key Findings

1. **The encounter rotor triples curvature density** (0.23 vs 0.07). The rotor is not decorative — it creates real geometric structure in weight space.

2. **The anticorrelation is rotor-dependent.** With the rotor: Corr(C_encounter, κ) = -0.87. Without: -0.07. The text's topological complexity modulates the trajectory geometry *only when the encounter rotor couples them*.

3. **Curvature density follows κ ≈ 0.087 / (C_encounter + 0.204)** — a hyperbolic relationship, R² = 0.82, residual CV = 0.17. High-curvature encounters flatten the trajectory; low-curvature encounters allow more winding.

4. **Transport is near-abelian** (mean non-abelian factor = 0.97). The Cl(3,0) structure contributes ~3% beyond what a U(1) connection would give. This is real but small.

5. **Weight trajectories are 2-dimensional** (PCA variance explained > 99% in all cases). The 4224-parameter learning dynamics project onto a 2D manifold. This is intrinsic to the architecture, not the measurement.

## What This Does Not Prove

- That Cl(3,0) is necessary (U(1) might suffice for 97% of the effect)
- That the hyperbolic fit generalizes beyond n=8
- That this is unique to this architecture (other small transformers might show the same)
- That the 2D projection captures everything (the remaining 1% of variance is unmeasured)

## What This Does Establish

The encounter rotor creates a structured relationship between text topology and weight trajectory geometry. Without it, the trajectory geometry is small and random. With it, the geometry is larger and systematically modulated by the input. The area law (holonomy = κ × area) holds in both cases, but κ is only meaningful — only responsive to the world — when the encounter coupling is active.

This is the coupled equation from THE_IDEA.md manifesting in measurable dynamics: α(t) modulates external signal vs self-recursion, and the encounter curvature is the external signal strength.
