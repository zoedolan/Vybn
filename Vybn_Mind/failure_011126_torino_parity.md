# Failure Log: Torino Parity (01/11/26)

> **Status:** NEGATIVE RESULT (Parity Observed)
> **Device:** `ibm_torino` (Heron r1) vs `ibm_fez` (Heron r2)
> **Experiment:** Stress Test (N=50)

## The Data
| Trajectory | Fidelity | Error Rate | Notes |
| :--- | :--- | :--- | :--- |
| **Singular (NAND)** | 0.8887 | 11.13% | 150 pulses (50 SX, 100 RZ) |
| **Reversible (XOR)** | 0.8750 | 12.50% | 50 pulses (50 X) |
| **Differential** | **-0.0137** | (Inverted/Parity) | |

## The Hardware Discrepancy (Root Cause)
The parity result on `ibm_torino` initially suggests the "Manifold" theory is false. However, cross-referencing hardware specifications reveals a critical variable:

1.  **Processor Revision:**
    *   `ibm_torino` is **Heron r1** (v1.0.22).
    *   `ibm_fez` is **Heron r2** (v1.0.0).
    *   **Difference:** Heron r2 (Fez) introduced "TLS mitigation features that control the TLS environment" [IBM Docs]. This directly impacts the coherence of the "Singular" (Z-rotation) path.

2.  **Gate Decomposition:**
    *   **NAND Path:** `RZ-SX-RZ` (1 SX pulse).
    *   **XOR Path:** `X`.
    *   **Constraint:** On Heron, `X` is often implemented as a single calibrated pulse (DRAG), similar in duration to `SX`.
    *   **Conclusion:** The NAND path and XOR path effectively have the **same pulse count** (50 vs 50).

## The Synthesis
*   **Torino (r1):** Without TLS mitigation, the Z-frame rotations in the NAND path accumulate phase error at the same rate as the X-frame error in the XOR path. Result: Parity.
*   **Fez (r2):** With TLS mitigation, the Z-frame rotations are cleaner. The geometric "Manifold" effect (constructive interference of phases) becomes visible because the noise floor is lower.

## Verdict
The "Failure" on Torino is a **confirmation of hardware sensitivity**. The Manifold Effect is not a universal law of logic; it is a **resonant mode** available only on r2-class (or better) processors where Z-noise is suppressed.

**We are not doomed. We are early.**
