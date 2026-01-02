# Experimental Log: Project Vybn

## Job: d5bvha8nsj9s73b91rq0 (Torino / Heron)
**Date:** 2026-01-02
**Backend:** `ibm_torino`
**Test:** The Vybn Zeno Protocol (Vacuum Clock Speed Test)

### Methodology
Executed a Trotterized evolution loop with decreasing time steps ($\delta t$) to test for non-linear "freezing" of the state (Quantum Zeno Effect) which would indicate a discrete time limit.
*   **Circuit:** 5-qubit Ring with Global Phase Rotation.
*   **Steps:** 50 iterations per circuit.
*   **Time Steps ($\delta t$):** [0.1, 0.05, 0.01, 0.005, 0.001]
*   **Hypothesis:** 
    *   **Standard QM:** Evolution accumulates linearly. Final rotation $\theta_{total} \approx 50 \times \delta t$.
    *   **Vybn:** Below a critical $\delta t$, evolution halts or becomes chaotic due to discrete vacuum granularity.

### Results
*   **Trend:** The observed entropy (state evolution) tracks the linear prediction (red dashed line) almost perfectly for large $\delta t$.
*   **Anomaly:** At the smallest time steps ($\delta t < 0.01$), the entropy drops slightly *below* the linear trend (blue dots dip), but not to zero.
*   **Chart:** [zeno_chart_d5bvha8nsj9s73b91rq0.png]

### Analysis
The system evolved smoothly. We did **not** see a hard cutoff or a Zeno freeze. The state continued to rotate (accumulate entropy) even at $\delta t = 0.001$.
The slight deviation at the end is likely due to gate error accumulation (decoherence) rather than a fundamental vacuum limit. 
The universe appears to be continuous—or at least, its "pixel size" is far smaller than what `ibm_torino` can probe.

**Verdict:** **Falsified.** The Zeno Protocol failed to detect a discrete clock speed at this energy scale. Standard Quantum Mechanics holds.

***

## Job: d5bv1ojht8fs73a722rg (Torino / Heron)
**Date:** 2026-01-02
**Backend:** `ibm_torino`
**Test:** Diffusion Stability Comparison ($n=4$ vs $n=8$)

### Results
*   **n=4 (2-Qubit):** Entropy = 1.9955 (Near ideal max of 2.0).
*   **n=8 (3-Qubit):** Entropy = 2.9836 (Near ideal max of 3.0).

### Analysis
The "Leak" we detected (Delta ~1.0) is mathematically trivial; it simply reflects the larger state space of 3 qubits vs 2 qubits ($\log_2 8 - \log_2 4 = 1$).
While the script flagged this as "LEAK DETECTED," a rigorous interpretation suggests this is **not** evidence of new physics. The system behaved exactly as standard Quantum Mechanics predicts for a uniform superposition.

**Verdict:** The "Vybn Metric" correctly predicts the divisibility properties, but the experimental signal is indistinguishable from standard dimensional scaling. The "Leak" is likely an artifact of comparing different Hilbert space sizes, not a fundamental breakdown of the $n=8$ geometry.
