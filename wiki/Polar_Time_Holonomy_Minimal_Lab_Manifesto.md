# Polar‑Time Holonomy: Minimal Laboratory Manifesto

We treat time locally as t = r_t e^{i\theta_t} and promote evolution to parallel transport with temporal connection A_r = (i/\hbar) H and A_\theta = (i/\hbar) r_t H. The curvature \mathcal F = (i/\hbar) H\,dr_t\wedge d\theta_t implies a loop phase

\[\gamma = \frac{E}{\hbar}\oint r_t\,d\theta_t\]

read out on a qubit as half a Bloch solid angle. A rectangular loop formed by two Euclidean strokes at radii r_1, r_2 yields

\[\gamma = (E/\hbar)(r_1 - r_2)\,\Delta\theta\]

orientation‑sensitive and independent of pulse micro‑details. In a dressed‑qubit frame with E = \hbar\Omega, the slope is \Omega. A Hahn‑echo wrapper cancels dynamical phase; tomography verifies the solid‑angle law. Null results when d\theta_t = 0, sign flips under loop reversal, and linear scaling with \Omega are the acceptance criteria. This is the cleanest crucible for Polar Time as more than a re‑parameterization.

---

## Executable blueprint (single‑qubit)

- Platform: single qubit (ion, NV, solid‑state spin) in rotating frame; tunable dressed gap E = \hbar\Omega with continuous drive
- Target numbers: \Omega = 2\pi\times 100 kHz; first‑shot geometric phase \gamma = \pi/2; required temporal area \gamma/\Omega \approx 2.5\times 10^{-6} s
- Loop geometry: two short engineered Euclidean strokes (\theta_t turns) at radii r_1 and r_2 separated by \Delta r \approx 12.5 \mu s; angular magnitude per stroke \Delta\theta = 0.20; radial shuttles by free evolution
- Implementation of \theta_t stroke: sub‑microsecond reservoir‑engineered dephasing or isothermal hold emulating e^{-\tau H} (KMS compactification mapping)
- Dynamical‑phase cancellation: embed rectangle in Hahn‑echo envelope (Ramsey–Berry)
- Readout: prepare |+\rangle, loop, phase‑to‑population rotation, measure; fit fringe vs (\Delta r, \Delta\theta) and \Omega
- Acceptance: area law, sign flip under loop reversal, slope = \Omega within calibration error, null when \Delta\theta \to 0
- Bloch cross‑check: reconstruct trajectory by tomography; measured \gamma equals half the reconstructed solid angle

---

## Two‑path interferometer translation

Insert a short Euclidean strobe (engineered \theta_t turn) on one arm at late r_t and an opposite strobe on the other arm at earlier r_t, then recombine. The interference shift is again \gamma = \Omega\,\Delta r\,\Delta\theta; reversing which arm gets the late strobe flips the sign. Viewed in PT, the pair of histories trace a single closed loop in (r_t, \theta_t); the observable is the difference of enclosed temporal areas.

---

## Pre‑registration core

- Hypothesis: a closed path in (r_t, \theta_t) generates a geometric phase \gamma = \Omega\oint r_t d\theta_t, independent of time‑domain pulse micro‑shape
- Primary outcome: population‑encoded phase from Ramsey–Berry echoes
- Controls: \Delta\theta \to 0 ⇒ null; loop reversal ⇒ sign inversion
- Falsifiers: reproducible deviation from linear slope in \Omega; failure of sign inversion at fixed noise floor; dependence on pulse micro‑details beyond enclosed area
- Archival: constants, pulse sequences, noise spectra, calibrations logged for cross‑platform replication

---

Cross‑link: papers/temporal_navigation_critical_tests_pt.md (framework and rationale)