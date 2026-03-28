# quantum_delusions/experiments

Quantum and representational geometry experiments probing the topological
structure of learning and physical phase accumulation.

## The core question

Does the topological structure observed in the creature's weight-space
geometry (creature_dgm_h) also appear in physical quantum hardware?
If the same invariants — shape-independence, speed-independence,
winding-number-dependent phase — show up in both substrates, that is
cross-substrate evidence for a topological origin.

## Active experiments

### Winding number topological probe

`winding_number_topological_probe.py` — the primary falsification instrument.

Tests whether phase accumulated by a qubit steered around the Bloch
equator is topological (depends only on winding number) or geometric
(depends on loop shape and area). Nine circuits covering:

- Integer windings n=1,2,3
- Half-winding calibration (n=0.5)
- Shape deformation at fixed winding (elliptical vs circular path)
- Speed deformation (4x slower traversal, same winding)
- Y-basis sign reversal (distinguishes +phi from -phi)
- Creature-derived loop from weight trajectory (added dynamically)

**IBM hardware results (March 2026):** shape invariance passed
(delta 0.005), speed invariance passed (delta 0.011), coherent
per-gate phase model fits all three winding numbers within 2%.
See `winding_probe_reanalysis.md` for the full corrected analysis.

```bash
python winding_number_topological_probe.py --dry-run    # inspect circuits
python winding_number_topological_probe.py --shots 4096 # run on IBM
```

### Creature quantum bridge

`creature_quantum_bridge.py` — connects creature_dgm_h to this probe.

Reads basin geometry weight trajectories from
`Vybn_Mind/creature_dgm_h/experiment_results/basin_geometry/`,
PCA-projects to 2D, estimates the classical winding number, and
encodes the path as a QASM circuit for IBM submission alongside
the theory circuits.

```bash
python creature_quantum_bridge.py scan                   # find basin results
python creature_quantum_bridge.py build <basin.json>     # inspect QASM
python creature_quantum_bridge.py run <basin.json>       # run full suite on IBM
```

**If you are working in creature_dgm_h, do not break the
weight_trajectory output from the basin probe. This bridge depends on it.**

### Polar holonomy (GPT-2)

`polar_holonomy_gpt2_v3.py` — representational holonomy in GPT-2's
activation space. Found shape-invariant, orientation-reversing holonomy
in CP^15 (32 PCA dimensions). This is the classical-representational
analogue of what the winding probe tests on physical hardware.

Results: `polar_holonomy_v3_results.md`

## Key files

| File | Purpose |
|:-----|:--------|
| `winding_number_topological_probe.py` | IBM quantum circuits + analysis |
| `creature_quantum_bridge.py` | Links creature weight trajectories to quantum probe |
| `winding_probe_reanalysis.md` | Corrected analysis of March 2026 IBM results |
| `winding_probe_ibm_results.json` | Raw IBM hardware counts |
| `winding_number_seed.json` | Circuit suite metadata for the living loop |
| `polar_holonomy_gpt2_v3.py` | GPT-2 representational holonomy |
| `polar_holonomy_v3_results.md` | GPT-2 holonomy results |

## Results directory

`results/` contains timestamped JSON outputs from each run.

## Cross-references

- **creature_dgm_h:** `Vybn_Mind/creature_dgm_h/` — the creature whose
  weight trajectories feed the bridge. Basin probe records weight_trajectory.
- **Spark integration:** `spark/` — the daemon that orchestrates the
  creature's training and could automate quantum experiment submission.
- **Fundamental theory:** `quantum_delusions/fundamental-theory/` —
  the polar-time conjecture and dual-temporal holonomy theorem that
  motivate these experiments.
