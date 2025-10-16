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

## 4. Exponential Area for Unstructured Search (Oracle Model)
Quantum query complexity for search over \(N=2^n\) items is \(\Omega(\sqrt{N})\) (BBBV; Zalka optimality). Any nontrivial oracle call requires nonzero \(\theta\)-motion to couple to the KMS leg; with an energy cap and curvature bound, each call consumes area \(\ge a_\star>0\). Hence
\[ A_{\mathrm{search}}(n)\ \ge\ a_\star\,\Omega(\sqrt{N})\ =\ a_\star\,\Omega(2^{n/2}). \]
This yields standard oracle separations translated into area units.

## 5. Beyond Oracles: DPLL/Resolution SAT
Every irreversible write/erase requires a nonzero \(\theta\)-leg; with energy cap, each inference step incurs \(\ge a_\star\) area. Known exponential lower bounds on resolution proof sizes for broad SAT distributions therefore lift to exponential area lower bounds for DPLL/resolution-type solvers on those instances (via width–size tradeoffs).

## 6. Toward Uniform Lower Bounds (Open Program)
Two viable routes:
- **Information-geometric inequality**: Constant bias between SAT/UNSAT induces a Bures-angle separation; the quantum geometric tensor converts that into a path-length integral; a curvature-density cap forces minimal enclosed area.
- **Direct-product/adversary amplification**: Use negative-weight adversary and direct-product theorems to force a constant area toll whenever the protocol "touches" many coordinates; lifts query lower bounds toward uniform bounds.

## 7. Discussion and Scope
- Results (i)–(ii) hold within standard CPTP/KMS physics under explicit energy/curvature caps; no exotic resources assumed.
- This framework does not resolve P vs NP. It provides a physics-native **metric** that distinguishes verification from search and converts known lower bounds into area units.
- Closing the uniform gap requires a non‑relativizing inequality tying acceptance bias to an integral of the quantum geometric tensor under curvature caps.

## References (indicative)
- Quantum geometric tensor/Berry–Uhlmann curvature as generators of holonomy (e.g., PRB 88, 064304)
- BBBV/Zalka bounds on search; adversary/direct-product theorems (ToC v6 a1)
- IBM survey on quantum strengths/weaknesses; resolution width–size lower bounds (ECCC 1999-022)
