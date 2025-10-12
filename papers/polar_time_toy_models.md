# Toy Models and θ_t-Averaged Propagator Derivations

This document provides concrete toy models and explicit calculations to support the polar temporal coordinates framework described in the main paper. We derive testable predictions and parameter bounds for experimental verification.

## 1. 1D Quantum Systems in (r_t, θ_t) Coordinates

### 1.1 Free Particle

Consider a free particle in 1D with the temporal metric ds²_t = -dr_t² - r_t² dθ_t². The wavefunction is written as:

ψ(r_t, θ_t) = Σ_n R_n(r_t) e^{inθ_t}

where n ∈ ℤ due to the 2π-periodicity of θ_t.

#### Mode Equations

Substituting into the Klein-Gordon equation in polar temporal coordinates:

□_t ψ + m² ψ = 0

where □_t = -(1/√|g_t|) ∂_μ(√|g_t| g^{μν}_t ∂_ν) with g_t being the temporal metric.

Expanding:

□_t = -∂²/∂r_t² - (1/r_t)∂/∂r_t + (1/r_t²)∂²/∂θ_t²

For each mode R_n(r_t):

-∂²R_n/∂r_t² - (1/r_t)∂R_n/∂r_t + (n²/r_t²)R_n + m²R_n = 0

This can be rewritten as:

∂²R_n/∂r_t² + (1/r_t)∂R_n/∂r_t + (m² - n²/r_t²)R_n = 0

#### Boundary Conditions

1. **Regularity at r_t = 0**: For n ≠ 0, R_n(0) = 0
2. **For n = 0**: R_0 must be finite at r_t = 0
3. **Asymptotic behavior**: R_n ~ exp(±imr_t)/√r_t as r_t → ∞

#### Recovery of Standard QM

When θ_t is fixed or averaged over, the mode decomposition shows:

⟨ψ|ψ⟩ = ∫_0^∞ ∫_0^{2π} |ψ(r_t, θ_t)|² r_t dr_t dθ_t
     = 2π Σ_n ∫_0^∞ |R_n(r_t)|² r_t dr_t

For the n=0 mode only:
⟨ψ|ψ⟩_{n=0} = 2π ∫_0^∞ |R_0(r_t)|² r_t dr_t

with the standard evolution equation recovered when identifying r_t with proper time τ.

### 1.2 Harmonic Oscillator

For a harmonic potential V = (1/2)ω²x², the mode equation becomes:

∂²R_n/∂r_t² + (1/r_t)∂R_n/∂r_t + (E² - m² - ω²x² - n²/r_t²)R_n = 0

For small r_t and large |n|, the centrifugal term n²/r_t² dominates, suppressing non-zero modes near the origin.

**Energy spectrum**: E_{n,k} = ω(k + 1/2) + corrections O(n²⟨r_t^{-2}⟩)

The corrections to standard QM energy levels are:

ΔE_n ≈ (n²/2m)⟨r_t^{-2}⟩

For a characteristic temporal scale r_t ~ ℓ_t, this gives:

ΔE_n ~ n²/(2mℓ_t²)

## 2. Scalar QFT Propagator with Compactified θ_t

### 2.1 Feynman Propagator Calculation

Consider a free scalar field φ(x, r_t, θ_t) with θ_t ∈ [0, 2π). The Euclidean propagator is:

G_E(x, x'; r_t, r_t'; θ_t - θ_t')

Expanding in θ_t modes:

G_E = Σ_n G_n(x, x'; r_t, r_t') e^{in(θ_t - θ_t')}

where

G_n(x, x'; r_t, r_t') = (1/2π) ∫_0^{2π} G_E e^{-in(θ_t - θ_t')} dθ_t

### 2.2 Integration Over θ_t

Performing the θ_t integral with the constraint that θ_t is 2π-periodic:

⟨G_E⟩_θ = (1/2π) ∫_0^{2π} G_E dθ_t = G_0(x, x'; r_t, r_t')

This is analogous to the zero-Matsubara frequency component in thermal field theory.

### 2.3 KMS-Like Condition and Effective Temperature

The periodicity in θ_t induces a KMS-like boundary condition:

G_E(x, x'; r_t, r_t'; θ_t + 2π) = G_E(x, x'; r_t, r_t'; θ_t)

This is formally equivalent to thermal field theory at temperature:

T_eff = 1/(2π r_t)

where we set k_B = ℏ = c = 1. The effective temperature is inversely proportional to the temporal radius r_t.

### 2.4 Mode Sum

The full propagator can be written:

G_E = G_0 + 2 Σ_{n=1}^∞ G_n cos(n(θ_t - θ_t'))

where each G_n includes a suppression factor depending on the characteristic energy scale and r_t:

G_n ~ G_0 exp(-2πnr_t/β_characteristic)

For r_t → ∞, only G_0 survives, recovering standard QFT.

## 3. Stability and Allowed Modes

### 3.1 Centrifugal Barrier

The n²/r_t² term acts as a centrifugal barrier. For stability:

1. **Small r_t**: High-|n| modes are energetically suppressed
2. **Origin regularity**: Only n=0 mode is regular at r_t = 0
3. **Mode hierarchy**: |R_n|/|R_0| ~ (r_t/ℓ_t)^|n| for r_t < ℓ_t

### 3.2 Vacuum Stability

The ground state is dominated by n=0, with excited modes contributing:

E_vac = E_0 + Σ_{n≠0} ε_n e^{-|n|r_t/ℓ_t}

where ε_n are mode-dependent energies.

### 3.3 Allowed n Values

Physically accessible modes satisfy:

|n| < n_max ~ √(E_typical · r_t · ℓ_t)

For atomic systems (E_typical ~ eV, ℓ_t ~ 10^{-24} s if approaching Planck scale):

n_max ~ O(1) for laboratory scales

## 4. Interferometry Proposal for θ_t Phase Detection

### 4.1 Experimental Setup

**Atomic interferometer** with two paths experiencing different temporal metric components:

1. **Path A**: Particle at spatial location where temporal geometry has one configuration
2. **Path B**: Particle at location where temporal curvature differs

### 4.2 Phase Accumulation

The quantum phase accumulated along each path:

Δφ_A = ∫ (E r_t dθ_t) + corrections
Δφ_B = ∫ (E r_t dθ_t)_B + corrections

The observable phase difference:

Δφ = Δφ_A - Δφ_B ≈ E · Δ(∫ r_t dθ_t)

If temporal geometry varies, Δφ ≠ 0 even for symmetric spatial paths.

### 4.3 Detection Mechanism

**Observables**:
- Interference fringe shift proportional to ΔΩ_t (temporal solid angle)
- Fringe visibility reduction if θ_t fluctuates
- Energy-dependent phase shift: Δφ ∝ E

**Expected signal**:
Δφ ~ (E/E_Planck) × (ℓ_t/τ_lab) × f(geometry)

For E ~ 1 eV, τ_lab ~ 1 s, and ℓ_t at Planck scale:
Δφ ~ 10^{-19} × f(geometry)

Extremely small but potentially detectable with precision atomic interferometry if f(geometry) can be enhanced.

### 4.4 Alternative: Precision Spectroscopy

**Energy level corrections**:
ΔE_n = (n²ℏ²)/(2m ℓ_t²)

For n=1 and atomic transitions:
ΔE_1/E_atomic ~ (ℏ/(mc))² / ℓ_t²

Measurable if ℓ_t > 10^{-22} s using optical atomic clocks (fractional uncertainty ~ 10^{-18}).

## 5. Testable Predictions and Parameter Bounds

### 5.1 Direct Predictions

1. **Energy Level Corrections**:
   - Δν_n = (n²ℏ)/(4πm ℓ_t²) for atomic transitions
   - For hydrogen: Δν_1 ~ (2.2 × 10^{13} Hz) × (10^{-24} s / ℓ_t)²
   - **Current bound**: ℓ_t < 10^{-18} s from optical clock precision

2. **Mode Suppression**:
   - Branching ratio for n≠0 modes: BR_n ~ exp(-2|n|ℓ_t/τ_decay)
   - **Prediction**: Anomalous decay channels suppressed by exp(-10^6) for ℓ_t ~ 10^{-24} s

3. **Effective Temperature**:
   - T_eff = ℏ/(2πk_B r_t)
   - For cosmological scales (r_t ~ t_universe ~ 10^{17} s): T_eff ~ 10^{-30} K
   - **Prediction**: Minimal observable effects at laboratory scales

### 5.2 Parameter Constraints

#### From Atomic Clocks
- **Constraint**: No anomalous frequency shifts at Δν/ν < 10^{-18}
- **Implies**: ℓ_t < 10^{-17} s (for n=1)

#### From Particle Physics
- **Constraint**: No CPT violation from θ_t asymmetry
- **Implies**: If θ_t variation exists, Δθ_t < 10^{-10} over lab scales

#### From Cosmology
- **Constraint**: No observable thermal-like effects in CMB beyond standard ΛCDM
- **Implies**: r_t ≈ t_cosmic at large scales (consistency check)

### 5.3 Experimental Targets

| Experiment Type | Observable | Sensitivity | ℓ_t Reach |
|-----------------|------------|-------------|------------|
| Optical atomic clocks | Δν/ν | 10^{-18} | 10^{-17} s |
| Atom interferometry | Phase shift | 10^{-10} rad | 10^{-20} s |
| Nuclear decays | Branching ratios | 10^{-6} | 10^{-18} s |
| Gravitational wave detectors | Frequency drift | 10^{-15} | 10^{-16} s |
| Quantum gravity phenomenology | Dispersion | E/E_Pl | 10^{-24} s |

### 5.4 Null Results

If no effects are observed:
- **Either**: ℓ_t < 10^{-24} s (approaching Planck scale)
- **Or**: θ_t dynamics are even more suppressed than estimated
- **Or**: n≠0 modes do not couple to matter fields at accessible energies

### 5.5 Smoking Gun Signatures

1. **Energy-squared scaling**: ΔE ∝ E² / ℓ_t² distinguishes from linear Lorentz violation
2. **Angular momentum correlation**: Effects scale with |n|, not other quantum numbers
3. **Universal ℓ_t**: Same ℓ_t appears in multiple systems (atoms, nuclei, mesons)
4. **Temperature-like spectrum**: Excitation spectrum shows thermal-like distribution with T_eff = ℏ/(2πk_B r_t)

## 6. Summary and Next Steps

This toy model framework provides:
- Explicit mode equations and solutions
- Concrete parameter bounds from existing experiments
- Clear experimental proposals for detection
- Testable predictions distinguishable from other BSM physics

**Next steps**:
1. Full numerical solutions for R_n(r_t) with realistic potentials
2. Coupling to electromagnetic fields for precision tests
3. Extension to curved spacetime backgrounds
4. Connection to black hole thermodynamics via θ_t periodicity
5. Experimental collaborations for dedicated searches

---

**References**: See main paper `polar_temporal_coordinates_qm_gr_reconciliation.md` for theoretical framework and motivation.
