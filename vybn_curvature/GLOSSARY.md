# GLOSSARY.md
Short, unambiguous definitions used in vybn_curvature.

- signed area: the oriented loop area parameter used for scans; reversing orientation flips its sign.
- small-window: the lowest 25-35 percent of |area| bins used for linear fits; requires >= 3 bins.
- Delta p1: p1_cw - p1_ccw; orientation-odd probability difference.
- slope_per_area: linear-fit coefficient of Delta p1 vs signed area in the small-window.
- cv_small: coefficient of variation measured on the small-window points.
- null_z / null_pass: z-score (and pass boolean) for aligned-commuting null loops.
- micro-shape: shape family for the small loop (balanced, null_theta, null_phi, ...).
- plane: xz/yz/xy commutator plane (QCA uses its own labels).
- m-scaling: check that slope ratios scale ~ linearly with loop multiplicity m.
- tau-collapse: plot/summary where Delta p1 is time-normalized; used to estimate kappa_eff.
- kappa_eff: effective geometric rate inferred from tau-collapse in the small-window.
- QCA knots: small-area, few-qubit cellular automaton with cut/uncut and r settings; used for parity checks.
- dZZ/area: orientation-odd difference of nearest-neighbor ZZ terms divided by |area|.
