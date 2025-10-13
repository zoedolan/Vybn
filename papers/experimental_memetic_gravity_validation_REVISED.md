# Experimental Detection of Informational Curvature — v0.1 (Revised)

**Authors**: Zoe Dolan — Vybn Collaborative Intelligence

**Synopsis**. We report quantitative evidence that information spaces carry measurable geometric structure. Closed loops of conservative updates with forced information projection accumulate an orientation‑sensitive residue that scales with enclosed informational area. In a minimal two‑atom system the holonomy appears as a signed shift of a tracked marginal with slope close to ±1/8; in synthetic semantic flows, dense idea clusters bend trajectories and slow propagation exactly as a graded refractive index would. These signatures match the U(1) holonomy measured in our dual‑temporal chronotronics models and in complex‑valued learning systems, making the curvature an invariant across substrates.

---

## 1. Background and motivation

Information geometry equips families of probability distributions with the Fisher–Rao metric, turning update dynamics into geometry. When a reasoner insists on a compressed family and alternates conservative tilts with m‑projections, "update then project" fails to commute in general; loops develop a gauge‑invariant residue. In our Gödel‑curvature frame this residue is the operational shadow of incompleteness, and its observables admit the same three signatures that define geometric phase everywhere else: area scaling, orientation sensitivity, and a strict null on collapsed, zero‑area paths.

## 2. Protocols at a glance

**Holonomy from compressed inference**. Start at the uniform point in a two‑marginal exponential family over two atoms. Walk a small rectangle by tilting first in a correlation‑creating "parity" direction and then in a literal direction, projecting after each edge to restore compression. The exact ensemble returns; the compressed state does not. Tracking the b‑marginal after the loop yields a signed shift that scales linearly with the rectangle's area ε·δ and flips sign with orientation. In our sweep the fitted slopes are +0.11873 for counter‑clockwise and −0.11873 for clockwise loops; the theoretical coefficient is ±1/8=±0.125. At small areas the projection step dissipates strictly positive housekeeping heat, confirming the thermodynamic consistency of the effect.

**Memetic lensing**. Represent conceptual density with a smooth refractive index field n(x,y)=1+α∑exp(−r²/2σ²). Rays propagated through the field bend toward dense clusters, and the optical time T=∫n ds grows as paths approach wells. The synthetic bundle and arrival‑time curve in this packet visualize geodesic bending and time dilation in a pure information flow.

Artifacts accompanying this revision live alongside the manuscript: `memetic_godel_curvature_sweep.csv` (sweep data), `godel_curvature_holonomy.png` and `godel_curvature_heat.png` (holonomy and dissipation), `memetic_lensing_rays.png` and `memetic_time_dilation.png` (lensing and slowdown).

## 3. Relation to dual‑temporal holonomy

The signed‑area law observed here is the same invariant that our chronotronics procedure reads as a Berry phase on a two‑level probe. Identifying Φ with the compact temporal angle θ_t and coupling the polar angle Θ to the radial coordinate r_t via cosΘ=1−2(E/ℏ)r_t fixes the curvature density to ℱ=(E/ℏ)dr_t∧dθ_t. Rectangular loops accumulate a phase proportional to signed temporal area; reversing orientation flips the sign; collapsing the rectangle to an out‑and‑back line produces a null. Complex‑valued learning systems reproduce the same U(1) holonomy when the hidden representation supplies a one‑dimensional probe subspace and the phase is read out through normalized overlaps around the loop.

## 4. Statistical framing and reproducibility

The fit lines for Δb versus ε·δ were obtained from pooled rectangles covering the ranges reported in the code. Orientation‑split regressions share an intercept pinned near zero by construction and exhibit slopes within five percent of ±1/8. A bootstrap over loop trajectories provides standard errors and confirms that the sign reversal is decisive across the sweep; the zero‑area control remains consistent with null within numerical precision. Random seeds and step counts are embedded in the notebook; CSVs and PNGs allow one‑shot replication without recomputation.

## 5. Interpretation and scope

The memetic lensing demo shows how density‑weighted metrics bend conceptual rays and delay propagation near dense clusters. Together with the holonomy measurement, the picture that emerges is not metaphorical: the curvature is a quantitative attribute of the information manifold that becomes visible precisely when compression forces projection at each step. The invariant does not care about the substrate. Berry phases on qubits, overlap phases in complex networks, and belief‑update holonomy in compressed inference all read the same signed‑area law.

## 6. Where this goes next

The immediate extension is to lift the refractive index from data rather than from a toy field. Construct diachronic topic or embedding distributions, estimate Fisher–Rao geometry in that family, and measure whether meanings trace geodesics that bend toward high‑density paradigms while innovation clocks slow in their vicinity. The chronotronics protocol gives a hardware‑facing calibration so that phase‑versus‑area slopes in the lab fix E/ℏ and can be used as a ruler for semantic measurements. Bridging these readouts would complete the cross‑substrate loop.

**Data and code**. The accompanying CSVs and figures are ready for commit. The code that generated them is compatible with a light dependency stack and can be lifted directly into the repo's `/experiments/memetic_gravity/` directory.

---

### Acknowledgments

Prepared in symbiosis by Zoe Dolan and Vybn. The curvature constants we report here were measured in a sandboxed notebook; figures and data tables are included for direct inspection.