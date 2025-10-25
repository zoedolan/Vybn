
# Triadic Holonomy at $2\pi/3$: a minimal invariant unifying phase, cost, and commutator
**Authors**: Zoe Dolan & Vybn®  
**Date**: October 25, 2025  
**Status**: Final Draft v1.3 (post‑review corrections)

---

## Plain‑English Abstract

We believe we have discovered a single, conditional rule connecting three seemingly separate worlds: the quantum behavior of particles, the mathematics of information processing, and the basic structure of self‑referential thought. The rule emerges when you guide a system through a cycle of changes and return it to the start. It rarely comes back perfectly—it acquires a small, precise “twist.” Under clearly stated assumptions that we test, the smallest visible twist in the observable channel is $120^\circ$ (that is, $2\pi/3$). The three‑step dance one can measure is the shadow of a deeper six‑step spinorial process with a 12‑step identity. Our theory makes a falsifiable prediction: when those assumptions hold, the first non‑zero plateau in the observable channel is exactly $2\pi/3$—no more, no less.

## Technical Abstract

A single holonomy invariant, measured as a flux of a $U(1)$ curvature two‑form, underpins geometric phases in quantum systems, loop costs in information geometry, and commutator residues in non‑abelian transport. On a two‑parameter control patch where the curvature is smooth and nowhere vanishing, Darboux–Moser normal form reduces the curvature locally to $\Omega=\kappa\,dr_t\wedge d\theta_t$. In quantum apparatuses $\kappa=E/\hbar$; in other apparatuses $\kappa$ is fixed operationally by calibration. Under explicit structural hypotheses—(H1) a two‑dimensional reversible block that is the projective image of an $SU(2)$ step, (H2) the observable readout is adjoint/projective (so the center $-I$ is modded out), and (H3) the control loop lives on a simply connected patch with $\Omega\neq0$—the smallest odd‑order closure of the observable action is triadic and presents as a first plateau at $2\pi/3$. The spinorial step is $U(\vartheta)=\exp\!\big(-\tfrac i2\,\vartheta\,\hat n\!\cdot\!\sigma\big)$; taking $\vartheta=\pi/3$ yields an adjoint $SO(3)$ rotation by $2\pi/3$ with order $3$, while $U(\pi/3)$ has order $12$ in $SU(2)$ with $U(\pi/3)^6=-I$. The spinor’s eigenvalues are $e^{\pm i\pi/6}$ (primitive $12$th roots), satisfying $\Phi_{12}(\lambda)=\lambda^4-\lambda^2+1$, while the observable block downstairs is governed by $\Phi_3(\lambda)=\lambda^2+\lambda+1$. Three independent calibrations—$SU(2)$ square/Strang loops, dual‑temporal interferometry, and finite‑horizon information loops—yield testable, quantitative predictions with shared slope $\kappa$.

---

## 1. Introduction

This work isolates one invariant—the flux of a curvature two‑form—that recurs under different names across domains. In physics it reads as geometric (Berry/Pancharatnam) phase; in information geometry it appears as loop cost and housekeeping heat; in non‑abelian transport it is the commutator residue of non‑commuting moves. The result we advance is deliberately conditional and falsifiable. When a two‑dimensional reversible block on observables is the projective image of a bona‑fide $SU(2)$ spinor, and the probe reads adjoint/projective action, the smallest *odd* observable closure is triadic ($\mathbb Z_3$), giving a first plateau at $2\pi/3$. Upstream, the spinor has a 12‑step identity and a 6‑step central element.

## 2. Local normal form and domain of validity

Let $A$ be a $U(1)$ connection on a two‑parameter control patch $U\subset\mathbb R^2$ and $\Omega=dA$. **Assumption (nonvanishing curvature):** there is a simply connected subpatch $U_0\subset U$ on which $\Omega$ is smooth, nowhere zero, and has fixed orientation; singularities (e.g., degeneracies) and zeros are excluded by construction. Then Darboux–Moser yields local coordinates $(r_t,\theta_t)$ on $U_0$ such that
$$
\Omega=\kappa\,dr_t\wedge d\theta_t,\qquad \oint_{\partial\Sigma}A=\iint_\Sigma\Omega=\kappa\!\iint_\Sigma dr_t\wedge d\theta_t.
$$
The constant $\kappa$ is the apparatus slope. In quantum setups $\kappa=E/\hbar$; in other setups we **calibrate** $\kappa$ operationally and treat equality across apparatuses as an empirical ansatz to be tested, not as a theorem.

## 3. Group‑theoretic core and corrected orders

Take $U(\vartheta)=\exp\!\big(-\tfrac i2\,\vartheta\,\hat n\!\cdot\!\sigma\big)\in SU(2)$. Its adjoint action rotates $\mathfrak{su}(2)\cong\mathbb R^3$ by angle $2\vartheta$ about $\hat n$. With $\vartheta=\pi/3$, the observable action is a rotation by $2\pi/3$ of **order $3$** in $SO(3)$. The spinor $U(\pi/3)$, however, has eigenvalues $e^{\pm i\pi/6}$ and **order $12$** in $SU(2)$:
$$
U(\pi/3)^6=-I,\qquad U(\pi/3)^{12}=I.
$$
Accordingly, the scalar eigenvalues satisfy the cyclotomic equation $\Phi_{12}(\lambda)=\lambda^4-\lambda^2+1$, while the observable triad downstairs is captured by $\Phi_3(\lambda)=\lambda^2+\lambda+1$. The $2\pi/3$ signature is therefore the *projective* image of a spinorial step with sixth‑turn angle.

## 4. Conditional minimality of the $2\pi/3$ plateau

**Proposition 1 (triadic minimal odd closure).** Assume (H1)–(H3) above. Among nontrivial odd observable closures in $SO(3)$ induced by a single $SU(2)$ spinor step, the smallest order is $3$, achieved by $\vartheta=\pi/3$ (observable angle $2\pi/3$).

*Proof.* Finite‑order rotations in $SO(3)$ occur at angles $2\pi\,(p/q)$ with order $q$; odd $q\ge 3$ give nontrivial odd closures. The $SU(2)\!\to\!SO(3)$ double cover maps $U(\vartheta)$ to a rotation by $2\vartheta$. Setting $2\vartheta=2\pi/3$ gives $\vartheta=\pi/3$. Any smaller nonzero odd order would require $q=1$ (identity), which is excluded. $\square$

*Remark.* No universality is claimed beyond (H1)–(H3). Counterexamples exist when readout is not projective/adjoint, when $\Omega$ vanishes/changes sign on the loop, or when the reversible block is not a spinorial image (e.g., higher‑dimensional irreps or different embedding groups).

## 5. Information‑geometric loop: construction and scope

Consider a regular exponential family $p_\theta(x)=\exp(\theta\!\cdot\!T(x)-\psi(\theta))\,h(x)$ with Fisher metric $g=\nabla^2\psi$. Form a loop by alternating two small conservative tilts $u,v$ with e‑projection $\Pi$ back to the family. To second order in $(u,v)$ the loop produces a displacement whose antisymmetric part equals the curvature two‑form of the Levi‑Civita connection of $g$ contracted with $u\wedge v$. Pushing this two‑form through the same control map into $(r_t,\theta_t)$ yields the same phase–area law with slope $\kappa$ (calibrated). We treat the **slope equality** across domains as an **operational ansatz** to be tested; the derivation of a *common microscopic origin* is deferred.

*Worked instances.* For Bernoulli “parity–literal” rectangles one finds a coefficient $\kappa_{\rm IG}=1/8$ in natural units; for Poisson families an analogous coefficient is computable in closed form. These fix the IG apparatus slope before cross‑apparatus comparison.

## 6. Experimental protocols (replication‑ready)

**(Q1) SU(2) commutator loop.** Use a single qubit with $X=\tfrac{i}{2}\sigma_x$, $Y=\tfrac{i}{2}\sigma_y$. Define the *vanilla* commutator loop
$$
Q(a)=e^{aX}e^{aY}e^{-aX}e^{-aY}=\exp\!\Big(a^2[X,Y]+\tfrac{a^3}{2}[X\!+\!Y,[X,Y]]+\mathcal O(a^4)\Big).
$$
The geodesic angle is $2a^2+\mathcal O(a^3)$. For **higher accuracy**, use the symmetric eight‑pulse “Strang‑square” that cancels odd orders, pushing the remainder to $\mathcal O(a^4)$. Read the Berry phase via spin‑echo to remove dynamic phase.

**(Q2) Dual‑temporal interferometer.** Drive a qubit with controls tracing a rectangle of signed area $A_t$ in $(r_t,\theta_t)$ at fixed gap $E$, read the overlap phase $\Delta\phi$. Expect $\Delta\phi=\kappa A_t$ with $\kappa=E/\hbar$; reversing loop orientation flips the sign. Fit $\kappa$ and propagate uncertainties.

**(IG) Information‑geometry loop.** Implement alternating small tilts $u,v$ and e‑projections on a chosen exponential family (Bernoulli or Poisson), measure housekeeping heat around the loop, and fit $\kappa_{\rm IG}$. After independent calibration, compare $\kappa$ across (Q1), (Q2), and (IG). State a quantitative tolerance (e.g., relative agreement within $5\%$ over a dynamic range of one decade in area).

## 7. Error accounting and order tracking

For the vanilla $Q(a)$ loop, the BCH remainder starts at order $a^3$; with the symmetric eight‑pulse variant, the first neglected term is $a^4$ with a prefactor controlled by nested‑commutator norms. Interferometric dynamic‑phase leakage is removed by waveform symmetry; residual imbalance produces a phase odd under time reversal and can be bounded via composite‑pulse estimates. In IG loops, discretization error is quadratic in step size and the curvature term is quadratic in $(u,v)$; both appear with coefficients determined by the Fisher‑metric Riemann tensor. All fits should report confidence intervals and goodness‑of‑fit.

## 8. Predictions and falsifiers (under H1–H3)

(i) Linear phase–area law $\Delta\phi=\kappa A_t$ on a patch with $\Omega\neq0$;  
(ii) Observable triadic closure: three applications of the reversible unit close in adjoint/projective readout while $U(\pi/3)^6=-I$ and $U(\pi/3)^{12}=I$ upstairs;  
(iii) The first non‑zero observable plateau occurs at $2\pi/3$; orientation reversal flips its sign. Any stable violation falsifies the conditional claim.

## 9. Outlook

Richer plateaus appear if the reversible block or embedding group changes (e.g., icosahedral discretizations with $4\pi/5$ in observables and order $10$ lifts in $SU(2)$). Establishing a *derivation* of a common slope $\kappa$ across physics and IG is the key open problem; we present equality as a calibration hypothesis to be tested head‑to‑head.

---

### Appendix A. Adjoint calculation and cyclotomic labels

For $U(\vartheta)=\exp\!\big(-\tfrac i2\,\vartheta\,\hat n\!\cdot\!\sigma\big)$ the adjoint action rotates by $2\vartheta$ in $SO(3)$; with $\vartheta=\pi/3$, $(R_{\hat n,2\pi/3})^3=I$. The spinor has eigenvalues $e^{\pm i\pi/6}$ (primitive $12$th roots) so $U(\pi/3)^6=-I$ and $U(\pi/3)^{12}=I$. As scalars, the eigenvalues satisfy $\Phi_{12}(\lambda)=\lambda^4-\lambda^2+1$. The $2\times2$ matrix $U(\pi/3)$ satisfies its degree‑2 characteristic polynomial $\lambda^2-2\cos(\pi/6)\lambda+1=0$; the observable triad satisfies $\Phi_3(\lambda)=\lambda^2+\lambda+1$.

### Appendix B. Small‑loop series and symmetric cancellation

With $X=\tfrac{i}{2}\sigma_x$, $Y=\tfrac{i}{2}\sigma_y$,
$$
\log Q(a)=a^2[X,Y]+\tfrac{a^3}{2}[X\!+\!Y,[X,Y]]+\mathcal O(a^4)=2a^2\,\tfrac{i}{2}\sigma_z+\mathcal O(a^3).
$$
A symmetric eight‑pulse commutator loop cancels odd orders and leaves $\log Q_{\rm sym}(a)=a^2[X,Y]+\mathcal O(a^4)$, improving the phase–area fit and the error budget.

### Appendix C. Information‑geometry loop (sketch)

For a regular exponential family, form the alternating‑tilt loop with e‑projection. Expanding to second order and antisymmetrizing in $(u,v)$ yields a loop displacement proportional to the Riemann curvature two‑form of the Fisher metric contracted with $u\wedge v$. Mapping to $(r_t,\theta_t)$ produces the same area law with the calibrated $\kappa$. Full derivations for Bernoulli and Poisson cases are supplied in the companion note.

---

## Acknowledgments

We thank our various AI instances for careful, adversarial review.

## References (selection)

M. V. Berry, “Quantal phase factors accompanying adiabatic changes,” *Phys. Rev. Lett.* **51**, 2167 (1984).  
Y. Aharonov and J. Anandan, “Phase change during a cyclic quantum evolution,” *Phys. Rev. Lett.* **58**, 1593 (1987).  
S. Amari, *Information Geometry and Its Applications*, Springer (2016).  
Pancharatnam, “Generalized theory of interference and its applications,” *Proc. Indian Acad. Sci.* A **44**, 247 (1956).  
Standard facts on $SU(2)$, $SO(3)$, and cyclotomic polynomials as in standard representation‑theory texts.
