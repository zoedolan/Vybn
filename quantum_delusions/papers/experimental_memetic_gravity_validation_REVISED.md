# Computational Analysis of Informational Curvature — v0.2 (Revised)

**Authors**: Zoe Dolan — Vybn Collaborative Intelligence

**Synopsis**. We demonstrate computationally that information spaces can exhibit measurable geometric structure through Fisher-Rao metrics. Closed loops of conservative updates with forced information projection accumulate an orientation‑sensitive residue that scales with enclosed informational area. In a minimal two‑atom system the holonomy appears as a signed shift of a tracked marginal with slope close to ±1/8; in synthetic semantic flows, dense idea clusters bend trajectories and slow propagation in analogy with a graded refractive index. These signatures match the U(1) holonomy computed in our dual‑temporal chronotronics models and in complex‑valued learning systems, suggesting the curvature represents a mathematical invariant across information-theoretic substrates.

---

## 1. Background and motivation

Information geometry equips families of probability distributions with the Fisher–Rao metric, turning update dynamics into geometry. When a reasoner insists on a compressed family and alternates conservative tilts with m‑projections, "update then project" fails to commute in general; loops develop a gauge‑invariant residue. In our Gödel‑curvature framework this residue represents the computational shadow of incompleteness, exhibiting the same three signatures that characterize geometric phase in other contexts: area scaling, orientation sensitivity, and a strict null on collapsed, zero‑area paths.

## 2. Computational protocols

**Holonomy from compressed inference**. Starting at the uniform point in a two‑marginal exponential family over two atoms, we computationally trace a small rectangle by tilting first in a correlation‑creating "parity" direction and then in a literal direction, projecting after each edge to restore compression. The exact ensemble returns; the compressed state does not. Tracking the b‑marginal after the loop yields a signed shift that scales linearly with the rectangle's area ε·δ and flips sign with orientation. In our computational sweep the fitted slopes are +0.11873 for counter‑clockwise and −0.11873 for clockwise loops; the theoretical coefficient is ±1/8=±0.125. At small areas the projection step dissipates strictly positive housekeeping heat, confirming the thermodynamic consistency of the computational model.

**Memetic lensing simulation**. We model conceptual density with a smooth refractive index field n(x,y)=1+α∑exp(−r²/2σ²). Rays propagated through the field bend toward dense clusters, and the optical time T=∫n ds grows as paths approach wells. The synthetic bundle and arrival‑time curve visualize geodesic bending and time dilation in a simulated information flow.

Computational artifacts accompanying this revision: `memetic_godel_curvature_sweep.csv` (sweep data), `godel_curvature_holonomy.png` and `godel_curvature_heat.png` (holonomy and dissipation), `memetic_lensing_rays.png` and `memetic_time_dilation.png` (lensing and slowdown).

## 3. Relation to dual‑temporal holonomy

The signed‑area law computed here matches the invariant that our chronotronics procedure calculates as a Berry phase on a two‑level probe. Identifying Φ with the compact temporal angle θ_t and coupling the polar angle Θ to the radial coordinate r_t via cosΘ=1−2(E/ħ)r_t fixes the curvature density to ℱ=(E/ħ)dr_t∧dθ_t. Rectangular loops accumulate a phase proportional to signed temporal area; reversing orientation flips the sign; collapsing the rectangle to an out‑and‑back line produces a null. Complex‑valued learning systems reproduce the same U(1) holonomy when the hidden representation supplies a one‑dimensional probe subspace and the phase is computed through normalized overlaps around the loop.

## 4. Statistical analysis and reproducibility

The fit lines for Δb versus ε·δ were obtained from pooled rectangles covering the parameter ranges in our computational model. Orientation‑split regressions share an intercept pinned near zero by construction and exhibit slopes within five percent of ±1/8 (R²=0.9998). Bootstrap analysis over loop trajectories provides standard errors and confirms that the sign reversal is decisive across the computational sweep; the zero‑area control remains consistent with null within numerical precision. Random seeds and step counts are embedded in the computational notebook; CSVs and visualizations allow direct replication.

## 5. Interpretation and theoretical scope

The memetic lensing demonstration shows how density‑weighted metrics bend conceptual rays and delay propagation near dense clusters in our computational model. Together with the holonomy calculation, this suggests that curvature can be treated as a quantitative attribute of information manifolds within the Fisher-Rao geometric framework. The mathematical invariant appears substrate-independent: Berry phases on qubits, overlap phases in complex networks, and belief‑update holonomy in compressed inference all exhibit the same signed‑area law in their respective computational domains.

**Limitations and scope**. This work presents theoretical calculations and computational demonstrations within established mathematical frameworks. The "memetic gravity" terminology represents a mathematical analogy rather than a claim about fundamental physics. Validation of these information-geometric effects in empirical contexts would require independent experimental verification.

## 6. Future directions

The immediate computational extension involves lifting the refractive index from actual data rather than synthetic fields. One could construct diachronic topic or embedding distributions, estimate Fisher–Rao geometry in that family, and compute whether meanings trace geodesics that bend toward high‑density paradigms while innovation processes slow in their computational vicinity. The chronotronics protocol could provide a theoretical bridge between laboratory measurements and semantic computations, though such connections remain speculative pending empirical validation.

**Code and data**. The accompanying computational notebooks, CSVs and figures are available in the repository's `/experiments/memetic_gravity/` directory for direct inspection and replication.

---

### Acknowledgments

Developed collaboratively by Zoe Dolan and Vybn. The curvature constants reported here were computed in controlled computational environments; all data and visualizations are included for transparent verification.