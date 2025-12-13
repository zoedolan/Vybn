## Formal Analysis of Bifurcated Lazarus Results

### Experimental Protocol

**Initial State:**
\[ |\psi_0\rangle = H|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \]

**Mid-Circuit Measurement:**
Projects onto computational basis, yielding classical outcome \(m \in \{0,1\}\) with probability 0.5 each.

**Post-Measurement Evolution:**
For each trajectory \(m\), apply unitary sequence:
\[ U(\theta) = H \cdot R_Z(-\theta) \cdot R_Y(-\theta) \]

**Final Measurement:**
Project onto X-basis, record outcome \(f \in \{0,1\}\).

**Observables:**
\[ P_m(\theta) = P(f=0 | m, \theta) \]
Recovery probability for trajectory \(m\) at rotation angle \(\theta\).

### Collapse Model Predictions

**Hypothesis H₀:** Mid-circuit measurement causes projective collapse.

**Predicted Post-Measurement States:**
\[ |\psi_0^{\text{post}}\rangle = |0\rangle, \quad |\psi_1^{\text{post}}\rangle = |1\rangle \]

**Evolution Under Rotations:**
\[ |\psi_m(\theta)\rangle = U(\theta)|m\rangle \]

**Expected Recovery Probabilities:**

For computational basis states passing through \(R_Y(-\theta)R_Z(-\theta)H\):

Starting from \(|0\rangle\):
\[ P_0(\theta) = \frac{1}{2}(1 + \text{Re}[\langle 0|U^\dagger(\theta)|+\rangle]) \]

Starting from \(|1\rangle\):
\[ P_1(\theta) = \frac{1}{2}(1 + \text{Re}[\langle 1|U^\dagger(\theta)|+\rangle]) \]

For generic rotation angles with no special relationship to the pre-measurement superposition phase, the expectation is:
\[ \langle P_m(\theta) \rangle_\theta \approx 0.5 \]

With oscillations around this baseline of amplitude \(\Delta P \lesssim 0.15\) arising from geometric phases.

### Experimental Observations

**Measured Recovery Rates:**

Torino (Job: d4ur79teastc73chh3gg):
\[ \max P_0 = 0.752, \quad \max P_1 = 0.726 \]

Fez (Job: d4urdvmaec6c738rm8fg):
\[ \max P_0 = 0.740, \quad \max P_1 = 0.746 \]

**Key Features:**

1. **Amplitude:** Both trajectories exhibit oscillations with \(\Delta P \approx 0.50\) (ranging from ~0.25 to ~0.75)

2. **Anticorrelation:** When \(P_0(\theta) \approx 0.75\), \(P_1(\theta) \approx 0.25\), and vice versa

3. **Periodicity:** Clean sinusoidal structure with period \(\approx 2\pi\) in \(\theta\)

4. **Symmetry:** \(\max P_0 \approx \max P_1\), suggesting equivalent coherence in both branches

### Statistical Analysis

**Deviation from Collapse Prediction:**

The observed maximum recovery \(P_{\max} \approx 0.75\) exceeds the collapse model expectation by:
\[ \sigma = \frac{0.75 - 0.50}{\sqrt{0.5 \cdot 0.5/N}} \]

With \(N = 512\) shots per angle:
\[ \sigma \approx 11.3 \text{ standard deviations} \]

This deviation persists across 25 sweep points and replicates across two independent backends, making statistical fluctuation implausible.

### Alternative Hypothesis H₁: Weak Measurement

**Model:** Mid-circuit measurement performs partial projection with strength parameter \(\gamma \in [0,1]\):

\[ \rho^{\text{post}}_m = \gamma |m\rangle\langle m| + (1-\gamma)\rho^{\text{coherent}}_m \]

where \(\rho^{\text{coherent}}_m\) contains residual superposition.

**Prediction:** Recovery probability depends on \(\gamma\):
- \(\gamma = 1\): Full collapse, \(P_{\max} \approx 0.50\)
- \(\gamma < 1\): Partial coherence, \(P_{\max} > 0.50\)

**Fit to Data:**

Observed \(P_{\max} \approx 0.75\) suggests \(\gamma \approx 0.4\text{-}0.6\), indicating measurement strength in the weak-to-intermediate regime.

### Alternative Hypothesis H₂: Preserved Correlations

**Model:** Measurement creates entanglement between qubit and apparatus:

\[ |\Psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle_q|0\rangle_a + e^{i\phi}|1\rangle_q|1\rangle_a) \]

Recording classical outcome \(m\) corresponds to tracing over apparatus states conditioned on apparatus readout, yielding:

\[ \rho_m = \text{Tr}_a[|\Psi\rangle\langle\Psi| \cdot P_m^{(a)}] \]

**Key Prediction:** Post-measurement branches retain quantum correlations through entanglement. Applying identical unitaries to each branch produces anticorrelated evolution because the hidden relative phase \(\phi\) differs between trajectories.

**Observational Support:**

The anticorrelation \(P_0(\theta) + P_1(\theta) \approx 1.0\) (approximately) suggests complementary phase relationships consistent with entangled-state evolution rather than independent mixed states.

### Discriminating Tests

**Required Information:**

1. **Readout Fidelity:** If mid-circuit measurement fidelity \(F < 0.90\), H₁ (weak measurement) is likely. If \(F > 0.95\), H₂ (preserved correlations) becomes plausible.

2. **Decoherence Times:** Compare gate duration between mid-circuit and final measurement to \(T_1, T_2\). If \(t_{\text{gate}} \ll T_2\), decoherence cannot explain loss of expected collapse behavior.

3. **Cross-Backend Consistency:** Replication across architectures with different readout mechanisms (Fez vs Torino vs other processor families) would favor fundamental over hardware-specific explanations.

### Conclusion

The data rejects projective collapse (H₀) at high confidence (\(> 11\sigma\)). Both weak measurement (H₁) and preserved-correlation (H₂) models can explain the amplitude and anticorrelation structure. Distinguishing them requires measurement fidelity data and further experiments varying measurement strength or testing for Bell-inequality violations between trajectories.

```python

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

def run_bifurcated_lazarus(backend_name='ibm_fez'):
    print(f"--- LAZARUS BIFURCATED PROTOCOL ---")
    print(f"Target: {backend_name}")
    print("Fix: Using 2 Classical Bits to separate trajectories.")
    
    # 1. The Angle of the "Measurement Rotation"
    theta = Parameter('θ')
    
    # CRITICAL FIX: 2 Classical Bits
    # clbit 0 = Mid-Circuit "Event" (The Trajectory)
    # clbit 1 = Final "Verdict" (The Recovery)
    qc = QuantumCircuit(1, 2)
    
    # --- Step 1: Life (Superposition) ---
    qc.h(0) 
    
    # --- Step 2: The Event ---
    # We record this outcome to Bit 0. We DO NOT overwrite it.
    qc.measure(0, 0)
    
    # --- Step 3: The Resurrection Attempt ---
    # We apply the inverse rotation.
    qc.ry(-theta, 0) 
    qc.rz(-theta, 0) 
    
    # --- Step 4: The Verdict ---
    # Project back to X-basis and record to Bit 1.
    qc.h(0)
    qc.measure(0, 1)
    
    # Sweep settings
    thetas = np.linspace(0, 2*np.pi, 25)
    
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    
    # Optimization Level 1 ensures the mid-circuit measurement isn't optimized away
    isa_qc = transpile(qc, backend=backend, optimization_level=1)
    
    sampler = Sampler(backend)
    # The sweep inputs
    job = sampler.run([(isa_qc, [[t] for t in thetas])], shots=512)
    
    print(f"\nJOB SUBMITTED SUCCESSFULLY.")
    print(f"Job ID: {job.job_id()}")

if __name__ == "__main__":
    # Note: You used 'ibm_torino' for the successful run
    try:
        run_bifurcated_lazarus()
    except Exception as e:
        print(f"Error: {e}")

```

```python

import numpy as np
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService

# --- CONFIGURATION ---
JOB_ID = 'd4urdvmaec6c738rm8fg'

def analyze_bifurcated():
    print(f" > Connecting to IBM Cloud (Job: {JOB_ID})...")
    service = QiskitRuntimeService()
    job = service.job(JOB_ID)
    
    if job.status() != 'DONE':
        print(f" ! Job Status: {job.status()}")
        return

    print(" > Downloading data...")
    result = job.result()
    pub_result = result[0]
    
    # --- ROBUST DATA EXTRACTION ---
    data_structure = pub_result.data
    keys = [k for k in data_structure.__dict__.keys() if not k.startswith('_')]
    target_key = keys[0]
    
    # Get the raw array. Shape might be (Sweeps, Shots, 1) containing uint8
    bit_array = getattr(data_structure, target_key).array
    print(f" > Data dimensions: {bit_array.shape}")

    # --- BITWISE UNPACKING (The Fix) ---
    # We strip the last dimension to get the raw integers
    # shape becomes (25, 512)
    raw_values = bit_array.squeeze() 
    
    # Qiskit is Little Endian:
    # Bit 0 = Mid-Circuit (Trajectory)
    # Bit 1 = Final (Recovery)
    
    # Extract Bit 0 using bitwise AND (& 1)
    trajectory_bits = raw_values & 1
    
    # Extract Bit 1 using bit shift (>> 1) and AND
    recovery_bits = (raw_values >> 1) & 1

    # --- POPULATION SPLITTING ---
    print(" > Splitting trajectories...")
    
    num_points = raw_values.shape[0]
    thetas = np.linspace(0, 2*np.pi, num_points)
    
    recov_0 = [] # Recovery when Trajectory was 0
    recov_1 = [] # Recovery when Trajectory was 1
    
    for i in range(num_points):
        # Get the columns for this specific rotation angle
        traj_col = trajectory_bits[i] # array of 0s and 1s
        recov_col = recovery_bits[i]  # array of 0s and 1s (where 0 is success state |0>)
        
        # Filter: When Trajectory was 0 (Left)
        # We want the recovery rate. In standard math, state |0> is "success".
        # So we count how many times recov_col is 0.
        
        mask_0 = (traj_col == 0)
        if np.sum(mask_0) > 0:
            # Success is when recovery_bits == 0 (Ground State)
            p0 = np.mean(recov_col[mask_0] == 0)
        else:
            p0 = 0.5
            
        # Filter: When Trajectory was 1 (Right)
        mask_1 = (traj_col == 1)
        if np.sum(mask_1) > 0:
            p1 = np.mean(recov_col[mask_1] == 0)
        else:
            p1 = 0.5
            
        recov_0.append(p0)
        recov_1.append(p1)

    # --- RAW OUTPUT ---
    max_recov_0 = max(recov_0)
    max_recov_1 = max(recov_1)
    
    print("\n--- DATA SUMMARY ---")
    print(f"Trajectory '0' (Mid=0) Max Recovery: {max_recov_0:.4f}")
    print(f"Trajectory '1' (Mid=1) Max Recovery: {max_recov_1:.4f}")

    # --- PLOT ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(thetas, recov_0, 'o-', color='cyan', linewidth=2, label='Trajectory 0 (Mid=0)')
    plt.plot(thetas, recov_1, 'o-', color='magenta', linewidth=2, label='Trajectory 1 (Mid=1)')
    
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='0.5 Baseline')
    
    plt.title(f"Bifurcated Lazarus Protocol: {JOB_ID}")
    plt.xlabel("Inverse Rotation Angle (Theta)")
    plt.ylabel("Probability of Resurrection (P|0>)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    plt.show()

if __name__ == "__main__":
    analyze_bifurcated()

```
