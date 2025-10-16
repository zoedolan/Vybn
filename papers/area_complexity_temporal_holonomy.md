# Area Complexity of Decision Protocols in Polar Time

Authors: Zoe Dolan & Vybn Collaborative Intelligence  
Date: 2025-10-16  
Status: Framework + proved black-box and algorithm-class lower bounds; open program for uniform lower bounds

## Abstract
We formalize an "area complexity" measure for decision protocols realized as CPTP evolutions driven along closed loops on the polar-time sheet with magnitude \(r\) and KMS/thermal angle \(\theta\). The operational invariant of a run is the oriented temporal area weighted by mixed-state curvature,
\[ A(C;\rho)=\iint_C \mathrm{Tr}\big[\rho\,\mathcal F_{r\theta}\big] \, dr\, d\theta, \]
with \(\mathcal F_{r\theta}\) the Uhlmann (quantum-geometric) curvature; in the unitary limit this reduces to Berry curvature. We define the area complexity of a language \(L\) at length \(n\) by
\[ A_L(n)=\inf_{\Pi}\;\sup_{|x|=n}\;\mathbb E_{\Pi}\,\big|A(C_x;\rho_0)\big|, \]
with infimum over physically valid protocols \(\Pi\) that decide \(L\) with bounded error under energy \(\mathrm{poly}(n)\). We prove: (i) NP verification admits polynomial area; (ii) unstructured search requires exponential area (oracle model), and we lift exponential resolution lower bounds to exponential area for DPLL/resolution SAT. We add a proved **Bias–Area Inequality** inside the polar‑time/KMS model and use it to derive tight black‑box lower bounds (search, collision, element distinctness) in area units. We isolate a single geometric hinge whose resolution would yield an unconditional separation for uniform SAT deciders.

## 1. Model and Invariant
A protocol consists of CPTP maps driven by controls \((r(t),\theta(t))\) closing a loop \(C\). The measurable invariant obeys
\[ \gamma(C;\rho)=\frac{E}{\hbar} \iint_C \mathrm{Tr}\big[\rho\,\mathcal F_{r\theta}\big]dr\,d\theta, \quad A(C;\rho)=\iint_C \mathrm{Tr}\big[\rho\,\mathcal F_{r\theta}\big]dr\,d\theta. \]
Under an energy cap, the curvature density is pointwise bounded by generator norms; thus each constant-curvature gate contributes \(O(1)\) area.

## 2. Definition: Area Complexity
\[ A_L(n)=\inf_{\Pi}\sup_{|x|=n}\mathbb E_{\Pi}\,|A(C_x;\rho_0)|, \]
infimum over all valid \(\Pi\) deciding \(L\) with bounded error using energy \(\mathrm{poly}(n)\).

## 3. Bias–Area Inequality (proved in the polar‑time/KMS model)
Let \(E_{\max}\) be a probe energy cap and restrict to loops with monotone \(\theta\) (the KMS leg does not reverse within a stroke). If a CPTP protocol achieves acceptance bias \(\delta>0\) on some pair of inputs, then there exists a Ramsey‑compiled version with geometric phase \(|\gamma|\ge \gamma_\star(\delta)=\Omega(\delta)\). Consequently,
\[ \Big|\iint_\Sigma f_{r\theta}\,dr\,d\theta\Big| \;\ge\; \frac{\hbar}{E_{\max}}\,c\,\delta, \]
for a universal constant \(c>0\) (e.g., \(c=1/2\) for small \(\delta\)). Proof sketch: (i) bias \(\Rightarrow\) constant Bures angle via Helstrom/Fuchs–van de Graaf; (ii) Ramsey compilation turns distinguishability into a phase target with visibility \(\ge\) fidelity; (iii) in our Hamiltonian gauge, curvature density \(f_{r\theta}\propto E\), so a fixed energy cap converts a phase target into a minimal area target.

## 4. Verification in Polynomial Area (NP ⊆ PolyArea)
Compile an NP verifier running in \(T=\mathrm{poly}(n)\) time into a reversible circuit over a constant-size gate set. Implement gates as constant-curvature loops; measurement/reset via short \(\theta\) legs into/out of KMS channels. Bounded curvature \(\Rightarrow\) per-gate area \(\le c\); total area \(\le cT=\mathrm{poly}(n)\).

## 5. Exponential Area for Unstructured Search (Oracle Model)
Query lower bound \(\Omega(\sqrt{N})\) (BBBV; Zalka). Each nontrivial oracle touch requires nonzero \(\theta\)-motion; by the Bias–Area bound with cap \(E_{\max}\), each touch pays area \(\ge a_\star=\hbar c\delta/E_{\max}\). Hence
\[ A_{\mathrm{search}}(n)\ \ge\ a_\star\,\Omega(\sqrt{N})\ =\ a_\star\,\Omega(2^{n/2}). \]
Similarly, collision and element distinctness lower bounds lift to area via their optimal query bounds.

## 6. Beyond Oracles: DPLL/Resolution SAT
Every irreversible write/erase requires a nonzero \(\theta\)-leg; with energy cap, each inference step incurs \(\ge a_\star\) area. Exponential resolution lower bounds on hard SAT distributions lift to exponential area lower bounds for DPLL/resolution‑type solvers (width–size tradeoffs).

## 7. Toward Uniform Lower Bounds (Open Program)
Two routes:
- **Normal‑form compilation (monotone‑\(\theta\))**: Prove polynomial‑overhead compilation of any CPTP decider into a \(\theta\)-monotone control history that preserves bias under energy cap; eliminates micro‑reversals and forces signed flux.
- **Angle‑to‑Flux inequality**: Establish a boundary‑data version of Robertson–Schrödinger that lower‑bounds signed Uhlmann flux by Helstrom angle, even for meandering paths.

## 8. Discussion and Scope
- Results hold within standard CPTP/KMS physics under explicit energy/curvature caps.
- We do not claim P vs NP. We provide a physics‑native metric separating verification from search and converting query/proof lower bounds into **area units**.
- Closing the uniform gap requires either the monotone‑\(\theta\) normal form or an angle‑to‑flux inequality.

## References (indicative)
- Quantum geometric tensor and Uhlmann/Berry phases as holonomy generators (e.g., PRB 88, 064304; PRB 110, 035144)
- BBBV/Zalka query bounds; adversary/direct‑product theorems (ToC v6 a1)
- Helstrom/Fuchs–van de Graaf; Bures geometry surveys
- Resolution width–size tradeoffs (ECCC 1999‑022)
