# Area Complexity of Decision Protocols in Polar Time

Authors: Zoe Dolan & Vybn Collaborative Intelligence  
Date: 2025-10-16  
Status: Framework + proven bounds (oracle/algorithm-class); open program for uniform lower bounds

## Abstract
We formalize an "area complexity" measure for decision protocols realized as CPTP evolutions driven along closed loops on the polar-time sheet with magnitude \(r\) and KMS/thermal angle \(\theta\). The operational invariant of a run is the oriented temporal area weighted by mixed-state curvature,
\[ A(C;\rho)=\iint_C \mathrm{Tr}\big[\rho\,\mathcal F_{r\theta}\big] \, dr\, d\theta, \]
with \(\mathcal F_{r\theta}\) the Uhlmann (quantum-geometric) curvature; in the unitary limit this reduces to Berry curvature. We define the area complexity of a language \(L\) at length \(n\) by
\[ A_L(n)=\inf_{\Pi}\;\sup_{|x|=n}\;\mathbb E_{\Pi}\,\big|A(C_x;\rho_0)\big|, \]
with infimum over physically valid protocols \(\Pi\) that decide \(L\) with bounded error under energy \(\mathrm{poly}(n)\). We prove: (i) NP verification admits polynomial area; (ii) unstructured search requires exponential area (oracle model), and we lift exponential resolution lower bounds to exponential area for DPLL/resolution SAT. We outline a uniform lower-bound program via information-geometric inequalities that tie acceptance bias to integrals of the quantum geometric tensor under curvature caps.

## 1. Model and Invariant
A protocol consists of CPTP maps driven by controls \((r(t),\theta(t))\) closing a loop \(C\). The measurable invariant obeys
\[ \gamma(C;\rho)=\frac{E}{\hbar} \iint_C \mathrm{Tr}\big[\rho\,\mathcal F_{r\theta}\big]dr\,d\theta, \quad A(C;\rho)=\iint_C \mathrm{Tr}\big[\rho\,\mathcal F_{r\theta}\big]dr\,d\theta. \]
Under an energy cap, the curvature density is pointwise bounded by generator norms; thus each constant-curvature gate contributes \(O(1)\) area.

## 2. Definition: Area Complexity
\[ A_L(n)=\inf_{\Pi}\sup_{|x|=n}\mathbb E_{\Pi}\,|A(C_x;\rho_0)|, \]
infimum over all valid \(\Pi\) deciding \(L\) with bounded error using energy \(\mathrm{poly}(n)\).

## 3. Verification in Polynomial Area (NP ⊆ PolyArea)
Take any NP verifier \(V\) running in time \(T=\mathrm{poly}(n)\) with witness length \(\mathrm{poly}(n)\). Compile \(V\) to a reversible circuit over a constant-size gate set. Implement each gate by a constant-curvature loop in \((r,\theta)\); measurement/reset via short \(\theta\)-legs into/out of KMS channels. Bounded curvature ⇒ per-gate area \(\le c\); total area \(\le cT=\mathrm{poly}(n)\).

## 4. Bias–Area Inequality (proved in our model)
Let a CPTP protocol with energy cap \(E_{\max}\) decide a promise problem with acceptance bias \(\delta\). Then there exists a Ramsey‑compiled version whose geometric phase satisfies \(|\gamma|\ge \gamma_\star(\delta)=\Omega(\delta)\), and hence
\[ \Big|\iint_\Sigma f_{r\theta}\,dr\,d\theta\Big|\ \ge\ \frac{\hbar}{E_{\max}}\,c\,\delta, \]
for a universal constant \(c>0\). Proof sketch: Helstrom/Fuchs–van de Graaf give a constant Bures angle from bias; ancilla Ramsey maps that to a phase target with visibility ≥ fidelity; in our Hamiltonian gauge, bounded energy caps curvature density, turning the phase target into a minimal area target.

## 5. Exponential Area for Unstructured Search (Oracle Model)
Quantum query complexity for search over \(N=2^n\) items is \(\Omega(\sqrt{N})\) (BBBV; Zalka optimality). Any nontrivial oracle call requires nonzero \(\theta\)-motion to couple to the KMS leg; with an energy cap and curvature bound, each call consumes area \(\ge a_\star>0\). Hence
\[ A_{\mathrm{search}}(n)\ \ge\ a_\star\,\Omega(\sqrt{N})\ =\ a_\star\,\Omega(2^{n/2}). \]
This yields standard oracle separations translated into area units and generalizes to other query lower bounds (collision, element distinctness, etc.) via the same toll.

## 6. Beyond Oracles: DPLL/Resolution SAT
Every irreversible write/erase requires a nonzero \(\theta\)-leg; with energy cap, each inference step incurs \(\ge a_\star\) area. Known exponential lower bounds on resolution proof sizes for broad SAT distributions therefore lift to exponential area lower bounds for DPLL/resolution-type solvers on those instances (via width–size tradeoffs).

## 7. Toward Uniform Lower Bounds (Open Program)
Two viable routes:
- **Monotone-\(\theta\) normal form (constructive)**: Compile any uniform CPTP decider, at poly overhead, into a form with orientation‑monotone \(\theta\) so micro‑reversals cannot shrink signed flux; then apply the Bias–Area bound.
- **Angle‑to‑Flux inequality (geometric)**: Prove a boundary‑data inequality that lower‑bounds signed Uhlmann flux by Helstrom/Bures angle even for meandering paths; strengthen Robertson–Schrödinger by coupling \(f\) to boundary data.

## 8. Discussion and Scope
- Results (verification in poly area; search exponential area; DPLL/exponential area) hold under standard CPTP/KMS dynamics with explicit energy caps.
- This metric does not resolve P vs NP; it provides a physics‑native resource that separates verification from search. The program closes if either route in §7 succeeds.

## References (indicative)
- Quantum geometric tensor and Uhlmann curvature (e.g., PRB 110, 035144; PRB 107, 165415)
- Helstrom, Fuchs–van de Graaf; query lower bounds (PRA 60, 2746; BBBV; Zalka)
- Resolution width/size lower bounds (ECCC 1999-022); adversary/direct-product theorems (ToC v6 a1)
