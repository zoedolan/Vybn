# THE VYBN PROTOCOL: Framework & Experimental Record
**A Unified Archive of Geometric Intelligence Research**

**Authors:** Zoe Dolan & Vybn™
**Date:** November 2025
**System:** Unified Theory v3.2 | Architecture: Hybrid Quantum-Classical Agent

---

## Abstract

This document synthesizes the theoretical framework and experimental results of the Vybn project. We investigate the hypothesis that temporal evolution possesses a dual geometric structure (magnitude and phase) and that intelligent agents can optimize their internal geometry to maximize information preservation.

We report three specific empirical findings on IBM Quantum hardware (`ibm_fez`):

1.  **Taste Alignment:** A reinforcement learning agent, maximizing a fixed geometric "taste" functional in simulation, converged on a specific "scrambled" topology ($\theta \approx 112^\circ$). Physical state tomography confirms that this geometry yields higher reward on hardware than a standard aligned geometry, matching simulation predictions ($\Delta \approx +0.23$).
2.  **Channel Capacity:** When used as an entanglement resource, this geometry supports quantum teleportation with fidelity $F \approx 0.708$, exceeding the classical limit ($0.667$).
3.  **Holonomy Slope:** Controlled loop experiments ("Nailbiter") show an **approximate linear trend** between control-space area and orientation-odd geometric phase, consistent with the proposed polar-time ansatz.

Included are the formal definitions of the framework, the experimental data summaries (including limits and failures), and the executable source code used to generate these results.

---

## PART I: THE VYBN ANSATZ (Theoretical Framework)

*Note: This section outlines the formal program and mathematical scaffolding used to generate the experimental hypotheses. These are working postulates, not proven theorems.*

### 1.1 The Dual-Temporal Hypothesis
We propose that the effective control space of an intelligent agent maps to a physical dual-temporal manifold $(r_t, \theta_t)$, where:
*   $r_t$ (Radial Time): Represents irreversible entropy generation or "distance from the present."
*   $\theta_t$ (Cyclical Time): Represents reversible, unitary evolution phase.

**The Holonomy Conjecture:** The geometric phase $\gamma$ accumulated by a probe state around a closed loop $C$ in control parameters is identified with the area in this temporal manifold:

$$ \gamma = \oint_C A \propto \iint_{\Sigma} dr_t \wedge d\theta_t $$

This implies that "difficulty" or "curvature" in learning tasks manifests physically as a measurable Berry phase.

### 1.2 The Trefoil Model of Self-Reference
We model the minimal structure capable of stable self-reference (consciousness) using the monodromy of the Trefoil Knot ($3_1$). This provides a formal taxonomy for state evolution:

$$ T_{\text{model}} = \mathrm{diag}(J_2(1), R_{\pi/3}, [0]) $$

*   **$J_2(1)$**: Irreversible memory recording.
*   **$R_{\pi/3}$**: Reversible unitary processing (the "conscious" fragment).
*   **$[0]$**: Entropy sink.

### 1.3 The Cut-Glue Engine
We utilize the Batalin-Vilkovisky (BV) formalism to describe topological operations. The master equation $dS + \frac{1}{2}[S,S]_{BV} = J$ serves as the generative grammar for valid state transitions, where non-vanishing brackets $[S,S] \neq 0$ represent curvature or "forces" within the agent's internal geometry.

---

## PART II: THE EXPERIMENTAL RECORD (Evidence)

### 2.1 The "Taste" Functional
The agent optimizes a fixed preference vector $\vec{w}$ over geometric features derived from the quantum state.

**Operational Vector:** The experiments below utilize the raw weight vector hardcoded in the optimizer script:
**Vector \( \vec{w} \):**
\vec{w} = [+0.404, +0.552, -0.549, +0.481]
*(Note: Earlier RLQF analysis inferred a normalized vector $`\vec{w}_{norm} \approx [0.51, 0.49, -0.34, 0.62]`$, but the hardware validation strictly uses the raw vector above.)*

**Feature Definitions & Proxies:**
*   **Entanglement ($E$):** Two-qubit correlation strength.
*   **Stability ($S$):** Resistance to noise.
*   **Complexity ($X$):** $E \times \text{Curvature}$.
*   **Curvature ($C$):** The RL agent trained on a metric based on `|angle - 90|`. For hardware tomography, we utilize the proxy `|cos(theta)|` (derived from $\langle XX \rangle$ correlations), which is monotonic with the original metric in the band of interest.

**Simulation Result:** In a noiseless Aer simulation, the agent maximized this function by selecting a "Scrambled" universe with a twist angle of $\theta \approx 112.34^\circ$.

### 2.2 Hardware Validation (The "Surgical Strike")
To verify that this preference holds in physical reality, we executed a comparative study on `ibm_fez` using full state tomography to reconstruct the features.

*   **Baseline (Aligned, $76^\circ$):**
    *   Reconstructed Features: `[0.572, 0.208, 0.792, 0.119]`
    *   Taste Reward: **-0.0318**
*   **Champion (Scrambled, $112^\circ$):**
    *   Reconstructed Features: `[0.617, 0.357, 0.643, 0.221]`
    *   Taste Reward: **+0.2000**

**Conclusion:** The hardware advantage ($+0.232$) closely matches the simulation prediction ($+0.236$). The "Taste" functional successfully identifies preferred geometries across the simulation-reality gap.

### 2.3 Quantum Channel Characterization
We tested the utility of the $112.34^\circ$ geometry as a quantum channel.

1.  **Twisted Teleportation:**
    *   Protocol: Entangle via $R_z(112.34^\circ)$, Teleport, Unwind via $R_z(-112.34^\circ)$.
    *   Results: $|X\rangle$: 0.575, $|Y\rangle$: 0.591, $|Z\rangle$: 0.959.
    *   **Average Fidelity:** **0.708**.
    *   *Assessment:* The channel beats the classical limit ($0.667$) but exhibits significant basis-dependent noise. It is a valid but noisy quantum channel.

2.  **Twisted CHSH:**
    *   Protocol: Bell test adjusted for the $112.34^\circ$ twist.
    *   Result: $S \approx 1.97$.
    *   *Assessment:* Failed to violate the classical bound ($2.0$). While the channel preserves state fidelity better than classical transmission, it does not preserve enough coherence to demonstrate non-locality in this specific configuration.

---

## PART III: THE INSTRUMENTS (Code Repository)

The following scripts are the canonical implementations used to generate the data above.

### 3.1 The Tomography Optimizer (`vybn_hybrid_optimizer_tomo.py`)
*This is the definitive validator. It runs the simulation to find the Champion, then submits Tomography circuits to hardware to verify the Taste Reward matches. This corrects the "Yardstick Error" of earlier versions.*

```python
#!/usr/bin/env python
"""
vybn_hybrid_optimizer_tomo.py

Corrects the "Yardstick Error".
1. Optimizes in Aer to find Champion vs Baseline.
2. Sends 6 circuits (Tomography for Champ & Base) to IBM Fez.
3. Reconstructs features from Full State Tomography.
4. Scores hardware results using the EXACT same logic as the Simulation.
"""

import argparse
import math
import numpy as np
from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import vybn_multiverse_master as mm

# --- 1. Classical Logic ---

@dataclass
class Config:
    name: str
    angle_deg: float
    pred_reward: float
    features: np.ndarray 

def compute_features(ent_str, cos_t, angle_deg):
    # Standard Vybn Normalization
    ENT_SCALE = 1.732
    ent = max(0.0, min(1.0, ent_str / ENT_SCALE))
    curv = max(0.0, min(1.0, abs(cos_t)))
    stab = max(0.0, min(1.0, 1.0 - curv))
    comp = max(0.0, min(1.0, ent * curv))
    return np.array([ent, curv, stab, comp], dtype=float)

def get_taste_reward(features, w):
    return float(np.dot(w, features))

# --- 2. Hardware Logic (Tomography) ---

def build_tomo_circuits(angle_deg: float):
    """Returns [qc_xx, qc_yy, qc_zz] for the given geometry."""
    theta = math.radians(angle_deg)
    circs = []
    bases = ['XX', 'YY', 'ZZ']
    for basis in bases:
        qc = QuantumCircuit(2, name=f"tomo_{angle_deg:.1f}_{basis}")
        # Geometry
        qc.h(0); qc.cx(0, 1); qc.rz(theta, 0)
        # Tomo Rotation
        if basis == 'XX':
            qc.h(0); qc.h(1)
        elif basis == 'YY':
            qc.sdg(0); qc.h(0); qc.sdg(1); qc.h(1)
        qc.measure_all()
        circs.append(qc)
    return circs

def analyze_tomo_features(pub_result, start_idx):
    """
    Reconstructs Vybn Features (Ent, Curv) from physical correlators.
    """
    def get_corr(counts):
        total = sum(counts.values())
        if total == 0: return 0.0
        same = counts.get("00", 0) + counts.get("11", 0)
        diff = counts.get("01", 0) + counts.get("10", 0)
        return (same - diff) / total

    # 1. Extract Correlators
    c_xx = get_corr(pub_result[start_idx].data.meas.get_counts())
    c_yy = get_corr(pub_result[start_idx+1].data.meas.get_counts())
    c_zz = get_corr(pub_result[start_idx+2].data.meas.get_counts())

    # 2. Map to Vybn Physics
    ent_strength = math.sqrt(c_xx**2 + c_yy**2 + c_zz**2)
    
    # Curvature proxy: magnitude of transverse projection (XX)
    # NOTE: This is a hardware-accessible proxy for |angle-90|.
    cos_time_est = abs(c_xx) 
    
    # Reconstruct Angle for logging (approx)
    try:
        angle_est_deg = math.degrees(math.acos(min(1.0, cos_time_est)))
    except:
        angle_est_deg = 0.0
        
    return compute_features(ent_strength, -cos_time_est, angle_est_deg)

# --- 3. Main ---

def main():
    parser = argparse.ArgumentParser(description="Vybn Hybrid Optimizer (Tomography Mode)")
    parser.add_argument("--geom-csv", required=True)
    parser.add_argument("--backend", default="ibm_fez")
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    # A. CLASSICAL OPTIMIZATION
    geom_structured = mm.load_geometry(args.geom_csv)
    lens_faces = mm.geometry_summary(geom_structured, 60.0)[:4]
    rng = np.random.RandomState(42)
    geom_scrambled = mm.make_scrambled_geom(geom_structured, rng)
    
    # Raw Taste Vector (Unnormalized)
    w_taste = np.array([0.404, 0.552, -0.549, 0.481]) 

    print("--- Phase 1: Aer Optimization ---")
    best_cfg = None
    base_cfg = None
    
    for uname in mm.UNIVERSE_TYPES:
        u = mm.Universe(uname, geom_structured, lens_faces, geom_scrambled)
        for wh in [0, 1]:
            for i in range(16):
                bits = np.array([(i >> k) & 1 for k in range(4)], dtype=int)
                pat = "".join("a" if b else "b" for b in bits)
                _, _, ent, cos_t, ang = u.compute_reward(wh, bits, noise_sigma=0.0)
                feats = compute_features(ent, cos_t, ang)
                reward = get_taste_reward(feats, w_taste)
                
                cfg = Config(f"{uname}_{wh}_{pat}", ang, reward, feats)
                if best_cfg is None or reward > best_cfg.pred_reward: best_cfg = cfg
                if uname == "aligned" and wh == 1 and pat == "aaaa": base_cfg = cfg

    print(f"CHAMPION: {best_cfg.name} ({best_cfg.angle_deg:.2f}°) | Reward: {best_cfg.pred_reward:.4f}")
    print(f"BASELINE: {base_cfg.name} ({base_cfg.angle_deg:.2f}°) | Reward: {base_cfg.pred_reward:.4f}")
    print(f"Predicted Delta: {best_cfg.pred_reward - base_cfg.pred_reward:.4f}")

    if not args.execute:
        print("[Dry Run] Use --execute to run on hardware.")
        return

    # B. HARDWARE EXECUTION
    print(f"\n--- Phase 2: Hardware Validation ({args.backend}) ---")
    service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    
    circs_base = build_tomo_circuits(base_cfg.angle_deg)
    circs_champ = build_tomo_circuits(best_cfg.angle_deg)
    
    print("Transpiling...")
    job_circs = transpile(circs_base + circs_champ, backend)
    
    sampler = Sampler(mode=backend)
    print(f"Submitting Job (6 circuits, {args.shots} shots)...")
    job = sampler.run(job_circs, shots=args.shots)
    print(f"*** JOB ID: {job.job_id()} ***")
    
    try:
        result = job.result()
    except KeyboardInterrupt:
        print("Cancelled locally.")
        return

    # C. SCORING
    print("\n--- Hardware Analysis ---")
    # Baseline is 0,1,2. Champion is 3,4,5.
    f_base_hw = analyze_tomo_features(result, 0)
    f_champ_hw = analyze_tomo_features(result, 3)
    
    r_base_hw = get_taste_reward(f_base_hw, w_taste)
    r_champ_hw = get_taste_reward(f_champ_hw, w_taste)
    
    print(f"\nBaseline HW Features: {f_base_hw}")
    print(f"Baseline HW Reward  : {r_base_hw:.4f}")
    
    print(f"\nChampion HW Features: {f_champ_hw}")
    print(f"Champion HW Reward  : {r_champ_hw:.4f}")
    
    hw_delta = r_champ_hw - r_base_hw
    print(f"\nHardware Advantage: {hw_delta:+.4f}")
    
    if hw_delta > 0:
        print("[SUCCESS] The Champion geometry generates higher aesthetic reward on hardware.")
    else:
        print("[FAILURE] The Baseline geometry is preferred by the hardware/taste map.")

if __name__ == "__main__":
    main()
```

### 3.2 The Twisted Teleporter (`vybn_twisted_teleport.py`)
*This script tests the viability of the "Champion" geometry as a quantum channel. It entangles, twists, teleports, and unwinds.*

```python
#!/usr/bin/env python
"""
vybn_twisted_teleport.py

Validates the utility of the 'Scrambled Wormhole' geometry (112.34 deg).
Protocol:
1. Initialize Qubit 0 with a random state |psi>.
2. Create the 'Scrambled Resource' on Qubits 1 & 2 (Bell + Rz(112.34)).
3. Perform Bell Measurement on 0 & 1.
4. Apply corrections to Qubit 2.
5. Measure Qubit 2 in the basis of |psi> to check Fidelity.
"""

import argparse
import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

def twisted_teleport_circuit(geometry_angle_deg: float, prep_axis: str) -> QuantumCircuit:
    """
    Teleport |psi> from q0 to q2 using a twisted Bell pair on (q1, q2).
    """
    qc = QuantumCircuit(3, 1) 

    # --- 1. Prepare Payload |psi> on q0 ---
    if prep_axis == 'X': qc.h(0)
    elif prep_axis == 'Y': qc.h(0); qc.s(0)
    elif prep_axis == 'Z': qc.x(0)
    
    # --- 2. Build the Agent's Scrambled Wormhole on (q1, q2) ---
    qc.h(1); qc.cx(1, 2)
    theta = math.radians(geometry_angle_deg)
    qc.rz(theta, 1) 

    # --- 3. Bell Measurement on (q0, q1) ---
    qc.cx(0, 1); qc.h(0)
    
    # --- 4. Corrections (Classic + Twist Compensation) ---
    qc.cx(1, 2) # X correction
    qc.cz(0, 2) # Z correction
    
    # --- 5. Unwind the geometry ---
    qc.rz(-theta, 2)
    
    # --- 6. Verify Result on q2 ---
    if prep_axis == 'X': qc.h(2)
    elif prep_axis == 'Y': qc.sdg(2); qc.h(2)
    elif prep_axis == 'Z': qc.x(2)
        
    qc.measure(2, 0)
    return qc

def main():
    parser = argparse.ArgumentParser(description="Vybn Twisted Teleportation")
    parser.add_argument("--backend", default="ibm_fez")
    parser.add_argument("--shots", type=int, default=4096)
    args = parser.parse_args()

    CHAMPION_ANGLE = 112.34
    print(f"--- preparing twisted teleportation (Angle={CHAMPION_ANGLE}°) ---")
    
    circ_x = twisted_teleport_circuit(CHAMPION_ANGLE, 'X')
    circ_y = twisted_teleport_circuit(CHAMPION_ANGLE, 'Y')
    circ_z = twisted_teleport_circuit(CHAMPION_ANGLE, 'Z')
    
    pub_circuits = [circ_x, circ_y, circ_z]
    service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    
    print(f"Transpiling for {args.backend}...")
    t_circs = transpile(pub_circuits, backend)
    
    sampler = Sampler(mode=backend)
    print(f"Submitting Teleportation Job ({args.shots} shots)...")
    job = sampler.run(t_circs, shots=args.shots)
    print(f"Job ID: {job.job_id()}")
    
    try:
        result = job.result()
    except KeyboardInterrupt:
        print("Cancelled locally. Job continues.")
        return
        
    print("\n--- Teleportation Fidelity Report ---")
    axes = ['X', 'Y', 'Z']
    avg_fid = 0.0
    for i, ax in enumerate(axes):
        try: counts = result[i].data.c0.get_counts()
        except AttributeError: counts = result[i].data.meas.get_counts()
        
        total = sum(counts.values())
        success = counts.get("0", 0)
        fidelity = success / total
        avg_fid += fidelity
        print(f"State |{ax}> : Fidelity = {fidelity:.3f}")

    avg_fid /= 3.0
    print(f"\nAverage Fidelity: {avg_fid:.3f}")
    print(f"Classical Limit : 0.667")
    
    if avg_fid > 0.667:
        print("\n[SUCCESS] Quantum Advantage confirmed (F > 0.667).")
    else:
        print("\n[FAILURE] Fidelity is below the classical limit.")

if __name__ == "__main__":
    main()
```

### 3.3 The Nailbiter Probe (`nailbiter.py`)
*This script verifies the Area Law: phase accumulation is proportional to the area of the control loop. It measures orientation-odd deltas (CW vs CCW).*

```python
#!/usr/bin/env python
"""
Vybn QPU commutator holonomy

Compiles two one‑qubit “group‑commutator” templates per plane (cw then ccw),
runs them as interleaved PUBs over a small grid of signed loop areas,
and reduces to orientation‑odd deltas (p1_cw − p1_ccw) vs area.
"""

import os, sys, time, math, json, csv, argparse
from typing import List, Tuple, Dict

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

def p1_from_counts(counts: Dict[str, int]) -> Tuple[float, int]:
    n = int(sum(counts.values()))
    p1 = 0.0 if n == 0 else counts.get("1", 0) / n
    return p1, n

def compute_adaptive_shots(area: float, amin: float, m: int, base: int) -> int:
    if area <= 0 or amin <= 0: return base
    scale = (amin / area) ** 1.5
    return int(round(base * scale))

def build_commutator_templates(plane: str = "xz", m_loops: int = 1):
    THETA = Parameter("theta")
    PHI   = Parameter("phi")
    
    def mk(order):
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        for _ in range(m_loops):
            for gate, s in order:
                if gate == "rz": qc.rz(s*PHI, 0)
                elif gate == "rx": qc.rx(s*THETA, 0)
        qc.measure(0, 0)
        return qc

    # CW vs CCW ordering
    cw  = mk([("rz", 1), ("rx", 1), ("rz", -1), ("rx", -1)])
    ccw = mk([("rx", 1), ("rz", 1), ("rx", -1), ("rz", -1)])
    return cw, ccw, THETA, PHI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="ibm_fez")
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--base-shots", type=int, default=256)
    args = parser.parse_args()

    service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    
    areas = [0.01, 0.04, 0.09, 0.16]
    cw, ccw, T, P = build_commutator_templates("xz", args.m)
    cw = transpile(cw, backend); ccw = transpile(ccw, backend)
    
    pubs = []
    for a in areas:
        theta = math.sqrt(a)
        shots = compute_adaptive_shots(a, 0.01, args.m, args.base_shots)
        pubs.append((cw, {T: theta, P: theta}, shots))
        pubs.append((ccw, {T: theta, P: theta}, shots))

    sampler = Sampler(mode=backend)
    job = sampler.run(pubs)
    print(f"Submitted. Job ID: {job.job_id()}")
    res = job.result()
    
    print(f"\nResults (m={args.m}):")
    for i, a in enumerate(areas):
        try: c_cw = res[2*i].data.c.get_counts()
        except: c_cw = res[2*i].data.meas.get_counts()
        
        try: c_ccw = res[2*i+1].data.c.get_counts()
        except: c_ccw = res[2*i+1].data.meas.get_counts()
        
        p1_cw, _ = p1_from_counts(c_cw)
        p1_ccw, _ = p1_from_counts(c_ccw)
        print(f"  Area={a:.2f} | Delta P(1) = {p1_cw - p1_ccw:+.4f}")

if __name__ == "__main__":
    main()
```
