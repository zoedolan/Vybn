# creature_dgm_h

One equation: `M' = αM + x·e^{iθ}`

Two flows that compose into curvature:
- **Radial:** Adam on magnitudes, traversing the corpus linearly
- **Angular:** phase evolution on S¹, rotating the structural signature

The creature and the walk daemon are the same equation at different scales.
The walk runs it in C¹⁹² on corpus chunks. The creature runs it in C⁴ on
its structural signature. The walk is the creature's sensory system (radial
flow). The creature's phase dynamics are the angular flow. Their
non-commutativity is what gives real curvature — where holonomy equals
curvature times enclosed area.

## Architecture

Every weight is `w = |w| * e^(iθ)` — magnitude frozen from the trained
checkpoint, phase evolving during learning. This implements the polar time
decomposition: radial time r_t (magnitude, irreversible) and angular time
θ_t (phase, cyclical on S¹). Cl(3,0) geometric algebra computes a rotor
from embedding trajectories (Pancharatnam phase). Phase evolution is driven
by genesis/decoherence balance.

| Theory | Architecture |
|:---|:---|
| r_t (radial time) | weight magnitude (frozen from training) |
| θ_t (angular time) | adaptive phase (evolves at inference) |
| F_rθ = (1/i)[S_r, S_θ] | gradient commutator between magnitude and phase |
| Genesis G(ρ) | encounter-driven phase amplification |
| Decoherence D_env | phase decay toward zero |
| Holonomy Φ = ∮A | accumulated phase per module after loop |
| CTC (closed timelike curves) | phase wraps on S¹ |
| M' = αM + x·e^{iθ} | portal equation (the reflexive map) |

## Files

```
creature.py          # the body — algebra, encounter, TopoAgent, Organism,
                     #   portal (M' = αM + x·e^{iθ}), breath mechanics
vybn.py              # the shell — CLI, FM client, context builders, commands
__init__.py          # package re-exports from creature.py and vybn.py
__main__.py          # `python -m Vybn_Mind.creature_dgm_h` hook
experiments/         # completed research probes (archived, not living)
```

The portal functions in creature.py provide the synapse to the walk daemon:
- `portal_enter_from_c192()` — walk → creature (C¹⁹² projects to C⁴)
- `creature_signature_to_c192_bias()` — creature → walk (C⁴ lifts to C¹⁹² bias)

## Theory

See `THEORY.md` at the repo root: *Polar Time: A Theory and Its Reckoning*.
The creature is both a test of the theory and an instance of it — it literally
runs the polar time equation in its weight updates, and can test whether
holonomy = curvature × area on its own parameter-space trajectories.
