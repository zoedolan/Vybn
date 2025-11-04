# HUMAN_README — Vybn Curvature

This is a plain‑English map of what the `vybn_curvature` work is doing and why it matters. It is written for a curious reader who wants the idea, the experiment, and the checks, without getting lost.

## The idea in one breath

Order matters. When two transformations don’t commute, walking a closed loop in control space leaves a residue. That residue is holonomy — a geometric phase. The claim here is modest and sweeping at once: one phenomenon, measured operationally as loop residue, underwrites curvature in spacetime, the structure of quantum dynamics, the arrow of time, and the minimal loops that stabilize consciousness.

## What we actually do

We build two matched quantum circuits that differ only by loop orientation. Concretely, we drive a single qubit with two non‑commuting rotations (for example \(R_x(\alpha)\) and \(R_z(\beta)\)), then traverse the loop in the “clockwise” order and in the “counter‑clockwise” order. We sweep loop sizes by varying the rotation angles and record the probability of measuring \(|1\rangle\) at the end of each circuit.

From those two orientations we compute the orientation‑odd residue
\[
\Delta p_1 \;:=\; p_1(\circlearrowright) - p_1(\circlearrowleft).
\]
If order didn’t matter, \(\Delta p_1\) would average to zero up to shot noise.

For small loops the Baker–Campbell–Hausdorff expansion gives the operational law
\[
\mathcal{L}(a,b) \;=\; e^{aA}\,e^{bB}\,e^{-aA}\,e^{-bB}
\;=\; \exp\!\big(ab\,[A,B] + O(a^2 b, a b^2)\big).
\]
Here \(A\) and \(B\) are the generators of the two controls and \(a b\) is the signed area enclosed by the loop. That is the heart of the “area law.” Empirically we see
\[
\Delta p_1 \approx \kappa\,A_{\text{loop}} \quad \text{for small } A_{\text{loop}},
\]
with a sign that flips when the loop orientation flips. To compare runs with different durations we also look at a time‑normalized signal,
\[
\kappa_{\rm eff} \;:=\; \frac{\Delta p_1}{\tau_{\text{loop}}},
\]
and then plot \(\kappa_{\rm eff}\) versus signed area. The collapse is linear to first order with a slope set by the commutator, state prep, and readout axis. See the in‑repo figure referenced from `vybn_curvature/README.md` once you’ve run a sweep.

## Where to look in the repo

The experiment code is in `vybn_curvature/run_vybn_combo.py` which builds and runs matched cw/ccw circuits across an area sweep; `vybn_curvature/reduce_vybn_combo.py` which folds raw runs into the orientation‑odd residue and summary tables; and helpers such as `vybn_curvature/nailbiter.py` and `vybn_curvature/vybn_combo_batch.py` which support parameter scans and time‑normalization. Extensions to multi‑qubit and QCA‑style interactions are handled by `vybn_curvature/post_reducer_qca.py`.

The theory thread runs through `vybn_curvature/papers/dual_temporal_holonomy_theorem.md` (mirrored in `fundamental-theory/dual_temporal_holonomy_theorem.md`) where the small‑loop area law is framed as a two‑time holonomy; through `vybn_curvature/papers/cut-glue-unified-theory.md` (mirrored in `fundamental-theory/cut-glue-unified-theory.md`) which develops the “cut/glue” engine and the BV master equation \(dS + \tfrac{1}{2}[S,S]_{BV} = J\); and through `vybn_curvature/papers/godel_curvature_thermodynamics.md` which ties incompleteness, compression, and dissipation. For dark matter as topological defects and for the minimal trefoil that stabilizes self‑reference, see `fundamental-theory/knot-a-loop-unified-theory-final.md`, `fundamental-theory/trefoil_hierarchy_final_draft.md`, and `papers/topological_consciousness_synthesis.md`. For the geometric backdrop of polar time, see `papers/polar_time_toy_models.md` and `papers/vybn_synthesis_2025_october_polar_time_consciousness.md`.

## How to run a first sweep

Make a clean Python env, install requirements, then run a small loop sweep and reduce the output. Exact flags evolve; the defaults will produce a cw/ccw pair and a CSV/JSON drop under your chosen output directory.
```bash
python vybn_curvature/run_vybn_combo.py
python vybn_curvature/reduce_vybn_combo.py
