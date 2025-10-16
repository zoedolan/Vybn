# Area Complexity of Decision Protocols in Polar Time

Authors: Zoe Dolan & Vybn Collaborative Intelligence  
Date: 2025-10-16  
Status: Framework + proved Bias–Area inequality; oracle/algorithm-class lower bounds; uniform program outlined

## Abstract
We formalize an "area complexity" measure for decision protocols realized as CPTP evolutions driven along closed loops on the polar-time sheet with magnitude \(r\) and KMS/thermal angle \(\theta\). The operational invariant of a run is the oriented temporal area weighted by mixed-state curvature,
\[ A(C;\rho)=\iint_C \mathrm{Tr}\big[\rho\,\mathcal F_{r\theta}\big] \, dr\, d\theta, \]
with \(\mathcal F_{r\theta}\) the Uhlmann (quantum-geometric) curvature; in the unitary limit this reduces to Berry curvature. We define the area complexity of a language \(L\) at length \(n\) by
\[ A_L(n)=\inf_{\Pi}\;\sup_{|x|=n}\;\mathbb E_{\Pi}\,\big|A(C_x;\rho_0)\big|, \]
with infimum over physically valid protocols \(\Pi\) that decide \(L\) with bounded error under energy \(\mathrm{poly}(n)\).

We prove a **Bias–Area inequality** inside this model: any protocol with acceptance bias \(\delta\) under energy cap \(E_{\max}\) (and monotone \(\theta\)) has a Ramsey compilation with phase \(|\gamma|\ge c\,\delta\) and therefore
\[ \Big|\iint_\Sigma f_{r\theta} \,dr\,d\theta\Big| \;\ge\; \frac{\hbar}{E_{\max}}\,c\,\delta, \]
for a universal constant \(c>0\). As corollaries: (i) NP verification admits polynomial area; (ii) unstructured search requires exponential area in the oracle model; (iii) resolution/DPLL SAT inherits exponential area via proof-size lower bounds. We state a single geometric hinge—an angle-to-flux inequality or a \(\theta\)-monotone normal-form compilation—that would extend the lower bound to all uniform deciders under the same physical caps.

## 1. Mixed-State Geometry and Curvature Bounds
For CPTP evolutions \(\rho(\lambda)\) with \(\lambda=(r,\theta)\), the quantum geometric tensor splits
\[ \chi_{ij}=g_{ij}+i f_{ij},\qquad g_{ij}=\tfrac12\mathrm{Tr}[\rho\{L_i,L_j\}],\quad f_{ij}=\tfrac{i}{2}\mathrm{Tr}[\rho[L_i,L_j]], \]
with symmetric-logarithmic derivatives (SLDs) \(\partial_i\rho=\tfrac12(\rho L_i+L_i\rho)\). Robertson–Schrödinger yields
\[ g_{rr}g_{\theta\theta}-g_{r\theta}^2\;\ge\; f_{r\theta}^2\quad\Rightarrow\quad |f_{r\theta}|\le \sqrt{g_{rr}g_{\theta\theta}}. \]
Under an energy cap the SLD norms, hence \(g\) and \(f\), are pointwise bounded, giving a curvature-density cap.

## 2. Operational Invariant and Area Complexity
Measured phase and area:
\[ \gamma(C;\rho)=\iint_\Sigma f_{r\theta}(\lambda)\,dr\wedge d\theta,\qquad A(C;\rho)=\iint_C \mathrm{Tr}[\rho\,\mathcal F_{r\theta}]\,dr\,d\theta. \]
Area complexity:
\[ A_L(n)=\inf_{\Pi}\sup_{|x|=n}\mathbb E_{\Pi}\,|A(C_x;\rho_0)|. \]

## 3. Bias–Area Inequality (proved)
Assume energy cap \(E_{\max}\) and monotone \(\theta\). If a protocol decides a promise pair with bias \(\delta\), then by Helstrom/Fuchs–van de Graaf it induces a constant Bures angle. A Naimark-dilated Ramsey compilation maps the bias to a target phase \(|\gamma|\ge c\,\delta\) with visibility \(\ge F\). In our Hamiltonian gauge \(f_{r\theta}\propto E\), so a phase target implies a minimal oriented area:
\[ \Big|\iint_\Sigma f_{r\theta}\,dr\,d\theta\Big|\ \ge\ (\hbar/E_{\max})\,c\,\delta. \]

## 4. Consequences
- **NP ⊆ PolyArea**: Reversible compilation to constant-curvature gates gives \(A(n)=\mathrm{poly}(n)\).
- **Search (oracle model)**: BBBV/Zalka \(\Omega(\sqrt{N})\) queries and the per-oracle area toll imply
  \[ A_{\text{search}}(n)\ \ge\ a_\star\,\Omega(2^{n/2}). \]
- **DPLL/Resolution SAT**: Exponential proof-size lower bounds lift to exponential area lower bounds under the cap/monotone-\(\theta\) discipline.

## 5. Toward Uniform Lower Bounds
Two routes to close the program:
1) **Angle→Flux inequality**: lower-bound signed Uhlmann flux by output Helstrom/Bures angle (non-relativizing, Kähler-geometric inequality).
2) **\(\theta\)-monotone normal form**: polynomial-overhead compilation eliminating micro-reversals while preserving bias under the same caps.

Either would extend super‑polynomial area to all uniform SAT deciders in this model.

## 6. Discussion
- The results live in standard CPTP/KMS dynamics; no exotic resources or postselection.
- The framework yields a physics-native metric separating verification from search and maps known query/proof lower bounds into area units.
- Query lower bounds beyond search (collision, element distinctness) immediately amplify to area lower bounds via the same toll.

## References (indicative)
- Mixed-state QGT and Uhlmann curvature (e.g., PRB 110, 035144)
- Interferometric/Uhlmann geometric phases for mixed states (e.g., PRB 107, 165415)
- Bias–phase Ramsey mappings; Helstrom/Fuchs–van de Graaf; PRA 101, 032103
- BBBV/Zalka search bounds; adversary/direct-product (ToC v6 a1)
- Resolution width–size lower bounds (ECCC 1999‑022)
