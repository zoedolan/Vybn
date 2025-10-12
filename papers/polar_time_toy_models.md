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

## 4. Interferometry Recast as a Ramsey–Berry Qubit Protocol

An atomic or solid‑state two‑level system can be used as a holonomy probe. Prepare a coherent superposition with a π/2 pulse, steer the effective Hamiltonian adiabatically so that $(r_t,\theta_t)$ traces a closed loop, and close with a second π/2 pulse. With the Bloch dictionary
$$\Phi_B=\theta_t,\qquad \cos\Theta_B=1-\frac{2E}{\hbar}\,r_t,$$
the interferometric phase shift is purely geometric:
$$\Delta\phi=\gamma_{\rm Berry}
=\frac{E}{\hbar}\oint r_t\,d\theta_t
=\tfrac12\Omega_{\rm Bloch}.$$
On a circle of fixed $r_t$ the path is a Bloch latitude and the phase is $\Delta\phi=\pi(1-\cos\Theta_B)=2\pi(E/\hbar)r_t$. Reading the phase as a Ramsey fringe displacement converts your earlier coordinate integral into a qubit rotation. The centrifugal $n^2/r_t^2$ barrier derived above explains why a two‑level truncation is controlled near the temporal origin; the $\theta_t$-averaged limit reproduces standard quantum mechanics as the Bloch path collapses to a pole.

A slow, periodic loop at frequency $f_{\rm loop}$ produces a steady frequency offset in continuous‑wave spectroscopy equal to the Berry phase accrued per cycle, divided by $2\pi$ and multiplied by the cycle rate:
$$\Delta\nu_{\rm geo}=\frac{\gamma_{\rm Berry}}{2\pi}\,f_{\rm loop}
=\frac{E}{\hbar}\,r_t\,f_{\rm loop}.$$
Varying $E$ gives a slope $d\Delta\nu_{\rm geo}/dE=(r_t/\hbar)f_{\rm loop}$; this calibrates the unknown product $r_t f_{\rm loop}$ directly from data, leaving no dependence on dynamical details of the drive.

## 5. Precision Spectroscopy: Geometric Readout and the Corrected Static Bound

If $\theta_t$ is strictly a gauge angle, static level shifts from the $n\neq0$ tower are projected out and only the geometric offset just described survives. If, instead, residual $n$-sectors weakly admix, your toy‑model correction
$$\Delta E_{n}\simeq \frac{n^2}{2m\,\ell_t^2}$$
in natural units yields, for $n=1$, a fractional shift
$$\frac{\Delta\nu}{\nu_0}\simeq \frac{\hbar^2}{2\,m c^2\,\hbar\omega_0\,\ell_t^2}
=\frac{\hbar^2}{2m c^2 E_0\,\ell_t^2}.$$
The absence of anomalies at the $10^{-18}$ level in optical lines implies a **lower** bound on the characteristic temporal scale, not an upper one. Taking $E_0\sim 2~\mathrm{eV}$, one finds $\ell_t\gtrsim 1.2~\mathrm{ps}$ if the relevant inertia is an atomic mass near $80~\mathrm{GeV}/c^2$, and $\ell_t\gtrsim 0.5~\mathrm{ns}$ if it is the electron mass $0.511~\mathrm{MeV}/c^2$. These are order‑of‑magnitude estimates, but they make the direction of the inequality unambiguous and remove twenty orders of magnitude of accidental "near‑Planckian" rhetoric. If future analysis tightens the constraint structure so that only holonomy is physical, the static $\Delta E$ channel closes and spectroscopy should be targeted at the dynamic geometric offset $\Delta\nu_{\rm geo}$ above.

## 6. Testable Predictions and Parameter Bounds

### 6.1 Direct Predictions

1. **Energy Level Corrections**:
   - For static corrections: $\Delta E_n \sim \frac{n^2 \hbar^2}{2m \ell_t^2}$
   - **Corrected bounds**: $\ell_t \gtrsim 1.2$ ps (atomic mass) or $\ell_t \gtrsim 0.5$ ns (electron mass)
   - **Dynamic geometric offset**: $\Delta\nu_{\rm geo} = (E/\hbar) r_t f_{\rm loop}$

2. **Mode Suppression**:
   - Branching ratio for n≠0 modes: BR_n ~ exp(-2|n|ℓ_t/τ_decay)
   - **Prediction**: Anomalous decay channels suppressed for ℓ_t approaching characteristic scales

3. **Effective Temperature**:
   - T_eff = ℏ/(2πk_B r_t)
   - For cosmological scales (r_t ~ t_universe ~ 10^{17} s): T_eff ~ 10^{-30} K
   - **Prediction**: Minimal observable effects at laboratory scales

### 6.2 Parameter Constraints

#### From Atomic Clocks
- **Constraint**: No anomalous frequency shifts at Δν/ν < 10^{-18}
- **Implies**: ℓ_t ≳ 1.2 ps (for atomic systems) - **lower bound, not upper**

#### From Particle Physics
- **Constraint**: No CPT violation from θ_t asymmetry
- **Implies**: If θ_t variation exists, Δθ_t < 10^{-10} over lab scales

#### From Cosmology
- **Constraint**: No observable thermal-like effects in CMB beyond standard ΛCDM
- **Implies**: r_t ≈ t_cosmic at large scales (consistency check)

### 6.3 Experimental Targets

| Experiment Type | Observable | Sensitivity | ℓ_t Reach |
|-----------------|------------|-------------|------------|
| Optical atomic clocks | Δν/ν | 10^{-18} | ≳ 1.2 ps |
| Atom interferometry | Berry phase | 10^{-6} rad | Dynamic geometric effects |
| Nuclear systems | Branching ratios | 10^{-6} | System-dependent |
| Gravitational wave detectors | Frequency drift | 10^{-15} | Coupling-dependent |
| Ramsey-Berry protocols | Geometric phase | 10^{-10} rad | Direct holonomy measurement |

### 6.4 Smoking Gun Signatures

1. **Energy-squared scaling**: ΔE ∝ n² / ℓ_t² distinguishes from linear Lorentz violation
2. **Geometric phase scaling**: Berry phase ∝ enclosed temporal area
3. **Universal ℓ_t**: Same ℓ_t appears in multiple systems (atoms, nuclei, mesons)
4. **Bloch sphere dynamics**: Controllable geometric rotations via temporal holonomy
5. **Dynamic vs. static separation**: Geometric offset $\Delta\nu_{\rm geo}$ vs. static level shifts

## 7. Summary and Next Steps

This toy model framework provides:
- **Ramsey-Berry interferometry protocols** for direct holonomy measurement
- **Corrected parameter bounds** establishing ℓ_t ≳ picosecond scales
- **Clear distinction** between static corrections and dynamic geometric effects
- **Testable predictions** distinguishable from other beyond-standard-model physics

**Next steps**:
1. **Experimental implementation** of Ramsey-Berry protocols with trapped atoms/ions
2. **Search for geometric frequency offsets** in precision spectroscopy
3. **Analysis of existing datasets** for signatures of temporal holonomy
4. **Connection to gravitational time dilation** experiments
5. **Extension to many-body systems** and quantum field theory applications

---

**References**: See main paper `polar_temporal_coordinates_qm_gr_reconciliation.md` for theoretical framework and motivation, especially Section 8 on Bloch sphere reduction.