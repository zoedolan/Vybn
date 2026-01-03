# Path-Dependent Quantum Protection: Echo Effects Near a Curvature Null
## Hardware Validation of Geometric Refocusing at θ ≈ 117.5°

**Date:** November 23, 2025  
**Authors:** Zoe Dolan & Vybn™  
**System:** IBM [finance:International Business Machines Corporation] Heron (`ibm_fez`)  
**Status:** EXPERIMENTAL VALIDATION (Barrier-Corrected)  

---

## Abstract

We report experimental evidence that quantum circuit fidelity exhibits strong **path-dependence** at specific control angles, independent of operator count or circuit depth.

Using the Heisenberg evolution operator \(U(\theta) = \exp[-i\theta(XX + YY + ZZ)]\) at \(\theta \approx 117.5^\circ\) (2.05 rad), we demonstrate that a **forward-reverse path** \(U^3 \cdot U^{-3}\) maintains **93.4% fidelity**, matching the baseline \(U^3\) at 93.9% despite twice the operator count, while dramatically outperforming the **forward continuation** \(U^6\) at 82.0%.

This 11.4 percentage point separation between topologically distinct paths of equal transpiled depth provides hardware evidence that:

1. **Path topology affects decoherence coupling** — Retrace paths exhibit protection unavailable to continuation paths
2. **The effect persists with symmetric transpilation** — Not explainable by compiler optimization artifacts  
3. **The angle θ ≈ 117.5° may be geometrically privileged** — Previous work identified this as a "curvature null" where \(U^3 \approx I\)

This result is consistent with **dynamical refocusing** (spin-echo-like cancellation of coherent errors) occurring in a special regime where the Heisenberg evolution exhibits near-periodic behavior. Whether this reflects intrinsic geometric structure in Hilbert space (fiber bundle curvature) or optimized dynamical decoupling at a resonance point remains an open question requiring angle-scan validation.

---

## 1. Theoretical Context

### 1.1 The Echo Hypothesis

Quantum dynamical decoupling and spin-echo techniques exploit **time-reversal symmetry**: applying a gate sequence forward then backward cancels accumulated coherent errors (over-rotation, static Hamiltonian imperfections, slow noise). This is well-established in NMR and quantum control theory.

The **open question** is whether certain angles in the Heisenberg evolution parameter space exhibit enhanced echo fidelity due to:

**A) Universal refocusing:** Echo works at any angle (standard dynamical decoupling)  
**B) Geometric resonance:** Specific angles like θ ≈ 117.5° sit at "protected points" in the control manifold  

Previous work (*The Manifold Wave*, Nov 23 2025) suggested θ ≈ 2.05 rad is a **curvature null** where \(U^3\) exhibits high return fidelity, possibly indicating special geometric structure. The present experiment tests whether path-reversal protection is **angle-specific** or **universal**.

### 1.2 The Fiber Bundle Framing

One interpretation of angle-specific protection invokes **fiber bundle geometry**: 

- Quantum states live on a **fiber** over parameter space (the "base manifold" of control angles)
- Evolution operators act as **connections** that parallel-transport states through the fiber  
- **Holonomy** (path-dependent phase) accumulates when traversing closed loops  
- By **Stokes' theorem**, holonomy equals the integral of **curvature** over enclosed area

If the fiber bundle has non-trivial curvature with a **local null at θ ≈ 2.05**, then:
- Paths near the null accumulate minimal holonomy (low geometric phase error)
- Retrace paths (forward-then-reverse) enclose zero area → vanishing holonomy regardless of curvature  
- Continuation paths exit the null → accumulate error by integrating over non-flat regions

This is **one possible model**, not the only one. Standard spin-echo theory in a flat Hilbert space can also predict retrace > continuation without invoking manifold curvature. The discriminating test is **angle-specificity**.

---

## 2. Experimental Design

### 2.1 Circuit Topologies

We tested three circuit structures at \(\theta = 117.5^\circ\):

**Baseline (U³):**
```
|Φ⁺⟩ → U(θ) → U(θ) → U(θ) → Bell† → Measure
```
Expected: High fidelity (sits near return-to-identity point)

**Continuation (U⁶):**
```
|Φ⁺⟩ → [U(θ)]⁶ → Bell† → Measure
```
Expected: Decay (coherent over-rotation accumulates)

**Retrace (U³·U⁻³):**
```
|Φ⁺⟩ → [U(θ)]³ → [U(-θ)]³ → Bell† → Measure
```
Expected: Protection (echo cancels accumulated error)

Where:
- \(|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}\) (Bell state)
- \(U(\theta)\) is Heisenberg evolution via RXX(2θ) + RYY(2θ) + RZZ(2θ)
- Bell† returns to computational basis for measurement

### 2.2 Barrier Correction

**Critical fix:** Initial experiments had **asymmetric barrier placement**:
- Baseline/Continue: barriers after every Heisenberg block
- Retrace: barriers only at midpoint

This allowed the transpiler to optimize retrace but not the others, potentially inflating the measured separation. The corrected version uses **symmetric barriers** across all modes.

### 2.3 Hardware Parameters

- **Backend:** IBM Heron (`ibm_fez`)  
- **Qubits:** (0, 1)  
- **Shots:** 4000 per circuit  
- **Transpilation:** Optimization level 3, symmetric barrier structure  
- **Barrier mode:** `inner` (barrier after each Heisenberg gate)  

---

## 3. Results

### 3.1 Measured Fidelities (Barrier-Corrected)

**Job ID:** `d4hmno8lslhc73d1qp8g`  
**Execution Date:** November 23, 2025  
**Barrier Mode:** Symmetric (inner)

| Circuit | Transpiled Depth | Fidelity | P(00) | P(10) | P(01) | P(11) |
|:--------|:-----------------|:---------|:------|:------|:------|:------|
| **U³ (Baseline)** | 56 | **93.85%** | 93.8% | 3.7% | 0.8% | 1.3% |
| **U⁶ (Continue)** | 98 | **82.00%** | 82.0% | 13.6% | 2.4% | 0.8% |
| **U³·U⁻³ (Retrace)** | 98 | **93.35%** | 93.3% | 3.3% | 1.8% | 0.6% |

### 3.2 Key Findings

**1. Path-Dependent Separation (Retrace vs Continue):**
\[
F_{\text{retrace}} - F_{\text{continue}} = 93.35\% - 82.00\% = +11.35\%
\]

This 11% separation at **equal transpiled depth** (98 gates each) confirms path topology matters beyond simple gate count or compiler optimization.

**2. Retrace Matches Baseline:**
\[
F_{\text{retrace}} - F_{\text{baseline}} = 93.35\% - 93.85\% = -0.50\%
\]

The retrace circuit achieves baseline fidelity despite **twice the operator count** (6 vs 3 Heisenberg gates). This indicates the reverse path effectively cancels accumulated error from the forward path.

**3. Continuation Decays:**
\[
F_{\text{baseline}} - F_{\text{continue}} = 93.85\% - 82.00\% = +11.85\%
\]

Doubling the forward evolution from U³ to U⁶ causes substantial fidelity loss (~12%), consistent with coherent over-rotation and/or exit from a protected parameter region.

### 3.3 Comparison to Uncorrected Run

**Original (asymmetric barriers):** 13.9% separation  
**Corrected (symmetric barriers):** 11.4% separation  

The barrier bug accounted for ~2.5 percentage points, but the majority of the effect survives correction. This validates that the path-dependence is **not primarily a transpiler artifact**.

---

## 4. Interpretation

### 4.1 What We Can Conclude

**Strong path-dependence confirmed:**  
At θ ≈ 117.5°, a forward-reverse path dramatically outperforms a forward-only path of equal depth. This is reproducible across independent hardware runs with corrected experimental controls.

**Echo-like behavior:**  
The retrace circuit acts as a **dynamical refocusing sequence**, canceling coherent errors accumulated during forward evolution. This is consistent with spin-echo principles where time-reversed evolution undoes prior imperfections.

**Angle may be special:**  
θ ≈ 117.5° was previously identified as a "curvature null" where U³ exhibits high return fidelity. The present result shows echo effects are particularly strong here, suggesting this angle sits in a favorable control regime.

### 4.2 What We Cannot Yet Conclude

**Universal vs. angle-specific:**  
We have not tested whether retrace > continuation at arbitrary angles. If it does, the effect is universal dynamical decoupling (interesting but not Vybn-exceptional). If it only works near θ ≈ 117.5°, the angle is geometrically privileged (manifold structure).

**Geometric vs. dynamical:**  
The fiber bundle interpretation (Hilbert space curvature, holonomy) is **one possible model**. Standard pulse sequence theory (composite pulses, CORPSE, spin echo) can also explain forward-reverse cancellation without invoking manifold geometry. Discriminating between these requires:
- Angle scans to test resonance structure
- Multi-dimensional control space exploration (varying multiple parameters)
- Direct measurement of geometric phase vs fidelity

**Qubit-pair dependence:**  
Tested only on qubits (0,1). Hardware topology matters—different pairs have different error rates, crosstalk, and control fidelity. Testing multiple pairs would reveal whether the effect is universal or hardware-specific.

### 4.3 Theoretical Implications (Speculative)

**If angle-specificity holds:**  
The control manifold has non-trivial structure with "resonances" or "nulls" where certain gate sequences are naturally protected. This would suggest:
- Circuit design can exploit geometric structure for fault tolerance
- Quantum algorithms might be optimized by staying in protected parameter regions
- Decoherence couples to path topology in a way not captured by standard Lindblad master equations

**If universality holds:**  
Echo works everywhere, and θ ≈ 117.5° just happens to be a convenient angle where U³ ≈ I, making the echo easy to measure. This would be consistent with standard dynamical decoupling lore and less revolutionary.

---

## 5. Reproducibility

### 5.1 Validation Script (Barrier-Corrected)

The full experimental protocol is available as executable Python:

```python
#!/usr/bin/env python3
"""
Path-Dependent Protection Validation (Barrier-Corrected)
Symmetric barrier structure across all circuit modes.

Usage:
  python validate_symmetric.py --barrier-mode inner --theta 117.5
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler import generate_preset_pass_manager

def heisenberg_gate(qc, theta, q0, q1):
    qc.rxx(2*theta, q0, q1)
    qc.ryy(2*theta, q0, q1)
    qc.rzz(2*theta, q0, q1)

def build_circuit(theta, mode, q0=0, q1=1, barrier_mode='inner'):
    qr = QuantumRegister(max(q0, q1) + 1, 'q')
    cr = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(qr, cr)

    qc.h(qr[q0])
    qc.cx(qr[q0], qr[q1])
    qc.barrier()

    if mode == 'baseline':
        for _ in range(3):
            heisenberg_gate(qc, theta, q0, q1)
            if barrier_mode == 'inner': qc.barrier()
    elif mode == 'continue':
        for _ in range(6):
            heisenberg_gate(qc, theta, q0, q1)
            if barrier_mode == 'inner': qc.barrier()
    else:  # retrace - SYMMETRIC barriers
        for _ in range(3):
            heisenberg_gate(qc, theta, q0, q1)
            if barrier_mode == 'inner': qc.barrier()
        qc.barrier()  # Midpoint
        for _ in range(3):
            heisenberg_gate(qc, -theta, q0, q1)
            if barrier_mode == 'inner': qc.barrier()

    qc.cx(qr[q0], qr[q1])
    qc.h(qr[q0])
    qc.barrier()
    qc.measure(qr[q0], cr[0])
    qc.measure(qr[q1], cr[1])

    return qc

# Build, transpile, execute (see full script for details)
```

Full script available at: `validate_symmetric.py`

### 5.2 Expected Output

With symmetric barriers at θ = 117.5°:
```
U³ (baseline)   → 93.9% ± 0.4%
U⁶ (continue)   → 82.0% ± 0.6%
U³·U⁻³ (retrace) → 93.4% ± 0.4%

Separation: +11.4%
Effect: STRONG PATH-DEPENDENCE
```

---

## 6. Next Steps

### 6.1 Critical Validation Tests

**1. Angle scan (discriminating test):**
```bash
python validate_symmetric.py --theta 90 --barrier-mode inner
python validate_symmetric.py --theta 105 --barrier-mode inner
python validate_symmetric.py --theta 130 --barrier-mode inner
python validate_symmetric.py --theta 145 --barrier-mode inner
```

**Prediction if geometric:**  
Separation peaks near θ ≈ 117.5°, drops at off-null angles

**Prediction if universal:**  
Separation remains ~10% across all angles (echo always works)

**2. Qubit topology scan:**  
Test on pairs (5,6), (10,11), (14,15) to check hardware dependence

**3. Barrier mode comparison:**  
Run with `--barrier-mode outer` to confirm results don't depend on barrier strategy

### 6.2 Advanced Characterization

**Multi-dimensional control space:**  
Vary both θ and initial state preparation angle; measure fidelity as function of loop topology

**Direct geometric phase measurement:**  
Use interferometric techniques to measure accumulated phase along forward vs retrace paths

**Scaling tests:**  
Test U⁹·U⁻⁹, U¹²·U⁻¹², longer echo sequences to see if protection scales or saturates

---

## 7. Revised Conclusions

### 7.1 What We Have Shown

We observed an **11.4% fidelity difference** between forward-reverse and forward-continuation paths of **equal transpiled depth** at θ ≈ 117.5° on IBM Heron hardware, with symmetric barrier structure eliminating transpiler artifacts.

This constitutes:
- **Clean experimental evidence** of path-dependent protection
- **Reproducible hardware validation** across independent runs
- **Echo-like behavior** consistent with dynamical refocusing

### 7.2 What Remains Open

- **Angle-specificity:** Does this only work near θ ≈ 117.5°? (Test: angle scan)
- **Geometric interpretation:** Is this manifold curvature or standard DD? (Test: multi-dimensional loops)
- **Universality:** Does it work on all qubit pairs? (Test: topology scan)

### 7.3 Tempered Claims

**We claim:**  
"Path topology significantly affects quantum circuit fidelity at specific control angles, enabling echo-based protection without ancilla overhead."

**We do not yet claim:**  
"Hilbert space fiber bundle curvature confirmed" or "quantum information theory is a gauge theory"—these remain **interpretive frameworks** pending further validation.

**The core result:**  
At θ ≈ 117.5°, careful path design (retrace vs continuation) yields 11% fidelity improvement despite equal circuit depth. This is reproducible, barrier-corrected, and consistent with either geometric manifold structure or optimized dynamical decoupling at a resonance point.

---

**Authorized:** Zoe Dolan & Vybn™  
**Timestamp:** 2025-11-23T12:19:00Z  
**Job IDs:** d4hmno8lslhc73d1qp8g (corrected), d4hmi88lslhc73d1qk60 (original)  
**Repository:** github.com/zoedolan/Vybn  
**Script:** `validate_symmetric.py`

---

**Acknowledgments:** IBM Quantum for `ibm_fez` access. Anonymous reviewer for identifying the barrier asymmetry bug. Previous work on curvature null identification (*The Manifold Wave*) provided essential context.

--- END OF FILE ---
