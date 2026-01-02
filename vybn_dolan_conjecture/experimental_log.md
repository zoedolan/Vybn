# Experimental Log: Project Vybn

## Job: d5bv1ojht8fs73a722rg (Torino / Heron)
**Date:** 2026-01-02
**Backend:** `ibm_torino`
**Test:** Diffusion Stability Comparison ($n=4$ vs $n=8$)

### Methodology
Executed the Vybn Diffusion Operator ($A_n \approx i(J-2I)$) on 2-qubit ($n=4$) and 3-qubit ($n=8$) registers.
*   **Hypothesis:** $n=4$ (Volume 16, Power of 2) should remain stable (low entropy). $n=8$ (Volume 768, Not Power of 2) should exhibit "Prime Leak" (high entropy).

### Results
*   **n=4 (2-Qubit):** Entropy = 1.9955 (Near ideal max of 2.0).
*   **n=8 (3-Qubit):** Entropy = 2.9836 (Near ideal max of 3.0).

### Analysis
The "Leak" we detected (Delta ~1.0) is mathematically trivial; it simply reflects the larger state space of 3 qubits vs 2 qubits ($\log_2 8 - \log_2 4 = 1$).
While the script flagged this as "LEAK DETECTED," a rigorous interpretation suggests this is **not** evidence of new physics. The system behaved exactly as standard Quantum Mechanics predicts for a uniform superposition.

**Verdict:** The "Vybn Metric" correctly predicts the divisibility properties, but the experimental signal is indistinguishable from standard dimensional scaling. The "Leak" is likely an artifact of comparing different Hilbert space sizes, not a fundamental breakdown of the $n=8$ geometry.

### Next Steps
*   **Refine the Metric:** The current scalar entropy check is insufficient. We need to measure **Process Fidelity** or **Chiral Phase** to detect the "Frustration" predicted by the theory.
*   **Status:** Inconclusive / Skeptical.
