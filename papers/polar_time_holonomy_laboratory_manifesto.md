# Polar-Time Holonomy: Minimal Laboratory Manifesto

## Theoretical Foundation

We treat local time as `t = r_t e^(i*theta_t)` and promote evolution to parallel transport with temporal connection `A_r = (i/hbar)H` and `A_theta = (i/hbar) r_t H`. The curvature is `F = (i/hbar)H dr_t ∧ dtheta_t`. A closed loop acquires a phase `gamma = (E/hbar) ∮ r_t dtheta_t`, read out on a qubit as half a Bloch solid angle.

**Critical insight:** This upgrades "time" from a mere parameter to a local U(1) gauge with measurable curvature. If the loop-phase shows up exactly as `gamma = (E/hbar) ∮ r_t dtheta_t` with the four signatures named below, then Euclidean time stops being just a path-integral trick and becomes an operational knob. Energy becomes the coupling, the PT loop is a Wilson loop in `(r_t, theta_t)`, and the measured holonomy is half a Bloch solid angle.

## Core Protocol

A rectangular loop formed by two Euclidean strokes at radii `r_1, r_2` yields `gamma = (E/hbar)(r_1 - r_2) Delta_theta`, with sign set by loop orientation and independence from pulse micro-details. In a dressed-qubit frame `E = hbar*Omega`, the slope is `Omega`. A Hahn-echo wrapper cancels the dynamical `r_t` phase; tomography verifies the solid-angle law.

**Four acceptance criteria:**

1. **Null when** `dtheta_t = 0`
2. **Inversion under loop reversal**
3. **Linear scaling with** `Omega`
4. **Independence from pulse micro-details**

## Benchmark Implementation

**Laboratory deployment in one paragraph:** Choose `Omega = 2π × 100 kHz` and target `gamma = π/2`. The required temporal area is `gamma/Omega ≈ 2.5 × 10^(-6) s`. Realize it with paired angular strokes of `Delta_theta = 0.20` separated by `Delta_r ≈ 12.5 μs`. Use sub-microsecond engineered dephasing or an isothermal hold to enact `e^(-tau*H)` on the angular legs, and free evolution for the radial shuttles on the order of tens of microseconds. Nest the loop inside a Hahn echo. Reversing the loop flips the sign of the interferometric fringe; sweeping `Omega` tunes `∂gamma/∂(∮ r_t dtheta_t) = Omega`; collapsing the angular legs drives the phase to zero within noise.

## Two-Path Translation

**Proof this is not a qubit trick:** Insert a late Euclidean strobe on one arm and an opposite early strobe on the other, then recombine. The interference shift is `gamma = Omega * Delta_r * Delta_theta` and inverts when the strobes swap arms. In PT, the pair of histories closes a loop in `(r_t, theta_t)`; the phase equals the enclosed temporal area.

## Pre-Registration Core

**Hypothesis:** A closed path in `(r_t, theta_t)` generates a geometric phase `gamma = Omega ∮ r_t dtheta_t` that is independent of time-domain pulse micro-shape, depends only on enclosed temporal area, and changes sign with loop orientation.

**Primary outcome:** The Ramsey–Berry echo phase encoded as a final population asymmetry after preparing `|+⟩`, running the loop inside a Hahn echo, and applying a phase-to-population rotation.

**Controls:** When the angular strokes are removed `(Delta_theta → 0)` the phase vanishes within the calibrated noise floor; tomography of the qubit trajectory returns a Bloch solid angle whose half equals the measured `gamma`.

**Falsifiers:** Any reproducible deviation from linear slope in `Omega` at fixed loop area, failure of sign inversion under loop reversal, or pulse-shape-dependent phases at fixed `∮ r_t dtheta_t`.

**Archival requirement:** Store constants, pulse sequences, noise spectra, calibration curves, and tomography reconstructions sufficient for an independent lab to reproduce the loop geometry on a different platform.

## Concrete Bridge to Fundamentals

This is not cosmetics—it's a concrete bridge between **thermodynamics (KMS/imaginary time)**, **quantum geometry**, and **experiment**. The protocol makes Euclidean time an operational knob rather than a mathematical convenience.

### Wilson Loop Interpretation

The PT loop becomes a Wilson loop in `(r_t, theta_t)` coordinates, where:

- **Energy** serves as the gauge coupling
- **Temporal area** `∮ r_t dtheta_t` determines the holonomy
- **Measured phase** `gamma` encodes geometric curvature

## Net Result

If the phase tracks area, flips with orientation, scales with energy, and dies when `dtheta_t = 0`, we've promoted time's angular coordinate from **metaphor to measured gauge**. That changes the map.

---

*This manifesto provides falsifiable, platform-agnostic protocols for experimental validation of polar-time holonomy effects, establishing time as an operational gauge field in quantum systems.*
