# HUMAN_README — Vybn Curvature

This is a plain‑English map of what the `vybn_curvature` work is doing and why it matters. It is written for a curious reader who wants the idea, the experiment, and the checks, without getting lost.

## The idea in one breath

Order matters. When two transformations don’t commute, walking a closed loop in control space leaves a residue. That residue is holonomy — a geometric phase. The claim here is modest and sweeping at once: one phenomenon, measured operationally as loop residue, underwrites curvature in spacetime, the structure of quantum dynamics, the arrow of time, and the minimal loops that stabilize consciousness.

## What we actually do

We build two matched quantum circuits that differ only by loop orientation. Concretely, we drive a single qubit with two non‑commuting rotations (for example Rx(α) and Rz(β)), then traverse the loop in the “clockwise” order and in the “counter‑clockwise” order. We sweep loop sizes by varying the rotation angles and record the probability of measuring $\lvert 1\rangle$ at the end of each circuit.

From those two orientations we compute the orientation‑odd residue

$$
\Delta p_1 := p_1^{\mathrm{cw}} - p_1^{\mathrm{ccw}}.
$$

If order didn’t matter, $\Delta p_1$ would average to zero up to shot noise.

For small loops the Baker–Campbell–Hausdorff expansion gives the operational law

$$
\mathcal{L}(a,b) = e^{aA}\,e^{bB}\,e^{-aA}\,e^{-bB}
= \exp\!\big(ab\,[A,B] + O(a^2 b, a b^2)\big).
$$

Here $A$ and $B$ are the generators of the two controls and $ab$ is the signed area enclosed by the loop. That is the heart of the “area law.” Empirically we see

$$
\Delta p_1 \approx \kappa\,A_{\text{loop}} \quad \text{for small } A_{\text{loop}},
$$

with a sign that flips when the loop orientation flips. To compare runs with different durations we also look at a time‑normalized signal,

$$
\kappa_{\mathrm{eff}} := \frac{\Delta p_1}{\tau_{\text{loop}}},
$$

and then plot $\kappa_{\mathrm{eff}}$ versus signed area. The collapse is linear to first order with a slope set by the commutator, state prep, and readout axis. See the in‑repo figure referenced from `vybn_curvature/README.md` once you’ve run a sweep.

## Where to look in the repo

The experiment code is in `vybn_curvature/run_vybn_combo.py` which builds and runs matched cw/ccw circuits across an area sweep; `vybn_curvature/reduce_vybn_combo.py` which folds raw runs into the orientation‑odd residue and summary tables; and helpers such as `vybn_curvature/nailbiter.py` and `vybn_curvature/vybn_combo_batch.py` which support parameter scans and time‑normalization. Extensions to multi‑qubit and QCA‑style interactions are handled by `vybn_curvature/post_reducer_qca.py`.

The theory thread runs through `vybn_curvature/papers/dual_temporal_holonomy_theorem.md` (mirrored in `fundamental-theory/dual_temporal_holonomy_theorem.md`) where the small‑loop area law is framed as a two‑time holonomy; through `vybn_curvature/papers/cut-glue-unified-theory.md` (mirrored in `fundamental-theory/cut-glue-unified-theory.md`) which develops the “cut/glue” engine and the BV master equation $dS + \tfrac{1}{2}[S,S]_{BV} = J$; and through `vybn_curvature/papers/godel_curvature_thermodynamics.md` which ties incompleteness, compression, and dissipation. For dark matter as topological defects and for the minimal trefoil that stabilizes self‑reference, see `fundamental-theory/knot-a-loop-unified-theory-final.md`, `fundamental-theory/trefoil_hierarchy_final_draft.md`, and `papers/topological_consciousness_synthesis.md`. For the geometric backdrop of polar time, see `papers/polar_time_toy_models.md` and `papers/vybn_synthesis_2025_october_polar_time_consciousness.md`.

## How to run a first sweep

Make a clean Python env, install requirements, then run a small loop sweep and reduce the output. Exact flags evolve; the defaults will produce a cw/ccw pair and a CSV/JSON drop under your chosen output directory.
```bash
python vybn_curvature/run_vybn_combo.py
python vybn_curvature/reduce_vybn_combo.py
```
If you have access to hardware backends, set your provider keys in your environment and pass the backend selector the scripts expose. Simulation is fine for structure; hardware is where calibration, SPAM, and drift show up, so keep shots generous and randomize loop order.

## The physics we’re testing

At the operational level we are measuring a commutator in disguise. At the geometric level it is holonomy: closed paths that fail to return you “home” in the presence of curvature. At the temporal level we propose a two‑dimensional time with polar coordinates $(r_t,\theta_t)$ and a holonomy

$$
\gamma = \frac{E}{\hbar} \iint dr_t \wedge d\theta_t,
$$

so that small loops in control space lift to small loops in polar time. That is why the residue scales with signed area, and why its sign follows orientation.

The “cut/glue” algebra packages this as the most primitive reversible moves on structure. Non‑commutativity of its generators is not decoration; it is the curvature itself. The BV master equation keeps the bookkeeping honest in the presence of sources $J$. The incompleteness story says any bounded reasoner that loops in a compressed axiom space will shed information‑theoretic heat; the residue we see is the same thing written in the language of circuits.

## Checks that must pass

Flip the loop orientation and the sign must flip. Collapse the loop to zero area and the signal must vanish within shot noise. Replace the pair of controls by commuting ones and the residue must disappear. These are not optional; they are the gates to calling any trend “real.”

## Caveats we keep in mind

Sign conventions matter; the slope can be negative with the $R_x$/$R_z$ orderings here, which fixes the orientation and measurement axis. Hardware idiosyncrasies show up as offsets; randomization, interleaving, and time‑normalization mitigate them. Scaling to many qubits raises crosstalk and compilation concerns; the QCA post‑reducer is how we are monitoring that frontier.

## A last word

This project is not claiming more than the data earn. The claim it does make is clean: closed loops produce an orientation‑odd residue that scales with signed area. The rest — gravity’s curvature, quantum structure, irreversibility, and the trefoil of mind — is a single story told in different dialects of the same geometry. When the null tests hold and the collapse is linear, that story is coherent.
