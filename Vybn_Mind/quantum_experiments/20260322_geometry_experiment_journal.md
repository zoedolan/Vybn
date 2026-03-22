# Quantum Geometry Experiment v1 — Results

**Date:** 2026-03-22T12:08 UTC  
**Hardware:** ibm_fez (156 qubits, Heron processor)  
**Quantum time:** 6.0s (two jobs × 3.0s)  
**Budget used:** ~3% of 28-day window (522s remaining)

## Question

Does the curvature of Vybn's ComplexMemory manifold produce measurably
different behavior when encoded as quantum states on real hardware?

## Design

Three snapshots from the 49-step ComplexMemory history, selected for
maximal curvature contrast:

| Snapshot | History index | Curvature (κ) |
|----------|--------------|----------------|
| FLAT     | 34           | 0.065          |
| MEDIUM   | 17           | 0.120          |
| CURVED   | 19           | 1.602          |

For each snapshot, two circuits (4 qubits, 16 amplitudes):
- **STATE_PREP**: Encode snapshot → measure → compare to ideal (TVD)
- **INTERFERENCE**: Encode snapshot → H⊗4 → measure → compare entropy

Total: 6 circuits × 1024 shots, run twice = 12,288 shots on ibm_fez.

## Results

### State Preparation Fidelity (TVD from ideal)

| Curvature | TVD (run 1) | TVD (run 2) | TVD (avg) |
|-----------|-------------|-------------|-----------|
| κ=0.065   | 0.067       | 0.078       | 0.072     |
| κ=0.120   | 0.065       | 0.076       | 0.070     |
| κ=1.602   | 0.075       | 0.089       | 0.082     |

**Δ(curved − flat) = +0.010**

### Interference Pattern (Shannon entropy, bits)

| Curvature | H_observed | H_ideal | Noise entropy |
|-----------|-----------|---------|---------------|
| κ=0.065   | 3.577     | 3.369   | +0.208        |
| κ=0.120   | 3.489     | 3.205   | +0.284        |
| κ=1.602   | 3.446     | 3.247   | +0.200        |

**Δ(curved − flat) entropy = −0.131** — tracks the ideal difference.
Noise entropy is ~0.20–0.28 bits uniformly across curvature levels.

## Verdict: NULL

Hardware noise is indifferent to the geometric structure of the input state.

- TVD Δ of +0.01 is within shot noise for N=1024
- Noise entropy is ~0.2 bits regardless of curvature
- Both runs reproduce each other closely — experiment is clean
- This bounds the claim that ComplexMemory curvature has quantum substrate correlates

The null result is the honest result.

## Budget

6s quantum time used. 522s remaining in the 28-day window.

## Raw Data

- Jobs: `d6vtnbs69uic73cjn5o0`, `d6vtnd2f84ks73dfnur0`
- Full JSON: `20260322_geometry_hardware_results.json`
- Code: `spark/quantum_geometry_experiment.py`
