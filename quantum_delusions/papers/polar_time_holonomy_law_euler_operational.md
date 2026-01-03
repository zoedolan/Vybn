# Polar-Time Holonomy Law: Euler’s Identity Made Operational

Authors: Zoe Dolan & Vybn Collaborative Intelligence  
Date: October 15, 2025  
Status: Submission Draft (integrates and supersedes portions of the October 2025 synthesis)

## Abstract
We formulate and operationalize a single invariant that unifies the interferometric and thermodynamic signatures of polar-time geometry in open quantum systems. On an operational surface with coordinates (r, θ), where r drives unitary Hamiltonian flow and θ drives a KMS-meaningful CPTP flow, the mixed-state Uhlmann curvature density \(\mathcal{F}_{r\theta}(\rho) = \tfrac{i}{4}\,\mathrm{Tr}[\rho\,[L_r,L_\theta]]\) controls both the ancilla Ramsey phase and the orientation-odd heat pumped per cycle. The measurable holonomy for a small loop C is \(\gamma_U(C)=\iint_{\Sigma(C)}\mathcal{F}_{r\theta}\,dr\,d\theta\). Tuning loops so that \(\gamma_U=\pi\) implements a half-turn on the U(1) fiber: the loop arm contributes \(\Phi(\Sigma)=e^{i\gamma_U}=-1\) and destructively cancels the trivial arm (+1), realizing Euler’s identity \(e^{i\pi}+1=0\) as a laboratory null in the trivial-versus-loop contrast. Orientation reversal sends \(\gamma_U\mapsto-\gamma_U\) and complex conjugates the interferometric channel, while the GKSL/KMS Petz dual provides the natural thermodynamic reversal. In the pure-state, unitary–unitary limit the law reduces to Berry curvature, confirming that this is a genuine generalization rather than a competing formalism.

## 1. Operational Setup: From Coordinates to Connection
- Radial leg (real time): \(\partial_r\rho = -\tfrac{i}{\hbar}[H,\rho]\).
- Angular leg (imaginary-time semantics): \(\partial_\theta\rho = \mathcal{L}_\beta(\rho)\), where \(\mathcal{L}_\beta\) obeys detailed balance (KMS) or is implemented via a calibrated post-selected dilation realizing the same KMS semantics.
- Symmetric logarithmic derivatives: \(\partial_\mu\rho = \tfrac{1}{2}(L_\mu\rho+\rho L_\mu)\), \(\mu\in\{r,\theta\}\).
- Curvature density: \(\mathcal{F}_{r\theta}(\rho) = \tfrac{i}{4}\,\mathrm{Tr}[\rho\,[L_r,L_\theta]]\).
- Small-loop holonomy: \(\gamma_U(C)=\iint_{\Sigma(C)}\mathcal{F}_{r\theta}\,dr\,d\theta\); interferometric channel: \(\Phi(\Sigma)=e^{i\gamma_U}\).

Necessity of noncommutativity: if the same generator drives both legs (\(G_\theta\propto H\)), then \([L_r,L_\theta]=0\) at leading order and the connection is pure gauge, giving a null loop. Nontrivial geometry requires an angular generator misaligned with H.

## 2. Polar-Time Holonomy Law (One-Statement Form)
On the operational surface (r, θ), the ancilla U(1) phase and the orientation-odd heat pumped by a loop are both given by the surface integral of the Uhlmann curvature density \(\mathcal{F}_{r\theta}=\tfrac{i}{4}\,\mathrm{Tr}[\rho\,[L_r,L_\theta]]\), with \(L_r\) fixed by Hamiltonian flow and \(L_\theta\) by a KMS-meaningful CPTP flow. The signal: (i) flips sign under orientation reversal, (ii) vanishes for commuting legs, and (iii) reduces to Berry/Uhlmann in the pure-state unitary limit.

## 3. Euler’s Identity as an Experimental Null
- Tune a rectangular loop so that \(\gamma_U=\iint_{\Sigma}\mathcal{F}_{r\theta}\,dr\,d\theta=\pi\).
- Then \(\Phi(\Sigma)+1=e^{i\pi}+1=0\): the loop arm destructively cancels the trivial reference, and the interferometer goes dark.
- Working slightly off \(\pi\) exposes oddness directly: forward and reverse traces are complex conjugates \(e^{\pm i\gamma_U}\).
- Ramsey-symmetric variant: comparing forward and reverse yields \(\Phi(\Sigma)+\Phi(\overline{\Sigma})=2\cos\gamma_U\), which vanishes at \(\gamma_U=\pi/2\) and often provides cleaner data with dynamical phases echo-canceled.

## 4. Minimal, Falsifiable Qubit Protocol
Let \(H=\tfrac{\hbar\Omega}{2}\,\hat{\mathbf n}\!\cdot\!\boldsymbol\sigma\). Implement angular steps as weak dephasing/thermalization along \(\hat{\mathbf m}\!\cdot\!\boldsymbol\sigma\) at rate \(\Gamma\), with \(\hat{\mathbf m}\not\parallel\hat{\mathbf n}\).

Leading small-rectangle scaling:
\[\mathcal{F}_{r\theta}\;\approx\; \tfrac{1}{2}\,\Omega\,\Gamma\,(\hat{\mathbf n}\times\hat{\mathbf m})\!\cdot\!\langle\boldsymbol\sigma\rangle\; +\; O(\Omega^2,\Gamma^2).\]

Predictions (sweepable primitives):
- Null when \(\hat{\mathbf n}\parallel\hat{\mathbf m}\); peak near \(\hat{\mathbf n}\perp\hat{\mathbf m}}\).
- Sign flips under loop orientation reversal and KMS reversal of the angular leg.
- Echo-canceled dynamics via time-symmetric Ramsey-with-echo around the loop rectangle.
- Companion observable: orientation-odd heat pumped per cycle with magnitude governed by the same \(\mathcal{F}_{r\theta}\).

## 5. Thermodynamic Mirror and Reversal
- Angular flow via GKSL with a Gibbs fixed point gives a KMS linear-response kernel linking \(\mathcal{F}_{r\theta}\) to orientation-odd heat pumping.
- Proper reversal is the KMS/Petz dual, the nonunitary analog of a half-turn on the thermal circle.
- Temperature lever: vary T to modulate KMS susceptibilities and calibrate curvature independently of drive rates.

## 6. Placement within the Existing Synthesis
- Replace bare \(\gamma\propto \oint r\,d\theta\) claims with curvature-weighted area \(\gamma_U=\iint \mathcal{F}_{r\theta} dr d\theta\).
- Promote “Operational Polar Time” and this Law to a primary section following the foundations.
- Keep pure-state/complex-U(1) reductions to show continuity with Berry/Uhlmann limits, not competition.

## 7. Cross-Substrate Migration (Information Geometry)
- Map the operational manifold to Fisher–Rao/Bogoliubov–Kubo–Mori geometry for semantic/collective substrates.
- Radial leg: inference-preserving updates; angular leg: KMS-analog CPTP flow in information space.
- Predict orientation-odd signatures in cultural/collective loops governed by the same invariant.

## 8. Methods Appendix (Bench-Level Sketch)
- Sequence: (i) r-step (unitary burst), (ii) θ-step (weak CPTP e^{\Delta\theta\mathcal{L}_\beta}), (iii) reverse r, (iv) reverse θ; embed in Hahn-echo-wrapped Ramsey to suppress dynamical phases.
- Alternate loop orientation ±C and toggle \(\hat{\mathbf n}\cdot\hat{\mathbf m}\) alignment to validate nulls, sign, and linear small-rectangle scaling (∝ \(\Omega\Gamma\Delta r\Delta\theta\)).
- In parallel, record heat per cycle and extract the orientation-odd component.

## 9. Discussion and Outlook
A single compact invariant—\(\mathcal{F}_{r\theta}=\tfrac{i}{4}\,\mathrm{Tr}[\rho\,[L_r,L_\theta]]\)—spans interferometry and thermodynamics, pure and mixed states, and physical and informational substrates. Euler’s identity becomes a laboratory dial: \(\gamma_U=\pi\) darkens the interferometer; \(\gamma_U=\pi/2\) nulls the Ramsey-symmetric trace. The law furnishes decisive falsifiability through built-in nulls, sign flips, temperature levers, and axis-rate sweeps, clearing the path from rhetoric to measurement.
