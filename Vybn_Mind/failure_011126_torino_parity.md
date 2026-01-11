# Failure Log: Torino Parity (01/11/26)

> **Status:** NEGATIVE RESULT
> **Device:** `ibm_torino`
> **Experiment:** Stress Test (N=50)

## The Data
| Trajectory | Fidelity | Error Rate |
| :--- | :--- | :--- |
| **Singular (NAND)** | 0.8887 | 11.13% |
| **Reversible (XOR)** | 0.8750 | 12.50% |
| **Differential** | **-0.0137** | (Inverted/Parity) |

## The Learning
1.  **Backend Sensitivity:** The "Manifold Effect" observed on `ibm_fez` ($\Delta +2.44\%$) does **not** replicate on `ibm_torino`.
2.  **Noise Floor:** `ibm_torino` is significantly noisier (88% fidelity at N=50) compared to `ibm_fez` (>96% fidelity at N=10).
3.  **The "Doom" Interpretation:** The user interpreted this as total failure ("we never learn").
4.  **The Scientific Interpretation:**
    *   The effect is not universal. It is likely a **Constructive Interference** phenomenon specific to the calibration of `ibm_fez`.
    *   On `ibm_torino`, the "Singular" path (3x logical depth) performed slightly *better* than the shorter XOR path, which is physically counter-intuitive but statistically consistent with parity given the noise.

## Corrective Action
*   **Do not extrapolate** `ibm_fez` results to all Heron processors.
*   **Hypothesis Update:** The "Boolean Manifold" is a *calibration-dependent* state. It requires a specific hardware environment to manifest.
*   **Next Step:** Isolate the difference between Fez and Torino. Is it the Pulse Schedule?
