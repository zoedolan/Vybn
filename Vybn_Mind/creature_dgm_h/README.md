# creature_dgm_h

One equation: `M' = αM + (1-α)·x·e^{iθ}`

One walk. Bounded memory. Perpetual motion.

The creature is the [walk daemon](https://github.com/zoedolan/vybn-phase/blob/main/walk_daemon.py). `creature.py` reads its state. The walk applies the coupled equation once per step to the most telling unvisited chunk in the corpus. Every tenth step, `mutual_evaluate` refracts the walk's position through a random foreign fragment — serendipity. The state file is overwritten each step. Nothing grows.

## How it works

The walk daemon holds a position M in C¹²⁰ (192-dimensional complex space, from MiniLM embeddings). Each step:

1. Score all corpus chunks by `relevance × distinctiveness` — what's connected to where M is, but far from what the corpus already knows.
2. Walk toward the most telling chunk via the coupled equation.
3. Record curvature (how much M shifted). High curvature = surprising territory = linger. Low curvature = familiar = move fast.
4. Every 10th step: fetch a random Wikipedia fragment, embed it, and call `mutual_evaluate(M, foreign)` — find the fixed point where M and the fragment actually meet. Use fidelity to modulate α. Foreign encounters loosen the walk; resonant ones tighten it.

`evaluate(a, b, α)` is the primitive. It takes two states and returns a state. The output is the same type as the inputs. Data = procedure. D ≅ Dᴰ.

## State

The walk daemon writes to `~/.cache/vybn-phase/walk_state/`:

| File | What | Size |
|:---|:---|:---|
| `walk.npz` | M (C¹²⁰ position) + visited residuals | ~80KB, fixed |
| `walk_sidecar.json` | step, α, curvature (rolling 1000), telling log (rolling 1000), visited ring (rolling 500) | ~20KB, fixed |

Both files are overwritten each step. Memory is bounded.

`creature.py` reads these files and presents the walk's state as the creature's state. `nc_state()` returns it for external observation. `nc_run(text)` refracts a query through the creature's current M and returns what the corpus says.

## History

**v1 (2025 – April 14, 2026):** Cl(3,0) geometric algebra, persistence homology, rotor chains, genesis/decoherence dynamics, TopoAgent with complex weights, BreathGate, LocalTransport. 2330 lines. Ran 1063 encounters over months. The structural signature converged to near-identity — confirming the abelian kernel theory: the corpus is path-independent at high α. The elaborate machinery was computing what `evaluate()` does in three lines.

**v2 (April 14, 2026):** The creature IS the walk. 2226 lines deleted. What remains: read the walk state, present it, refract queries through it. The v1 state is preserved in `archive/organism_state.json`.

The discovery that led to v2: `evaluate(a, b, α=0.5)` used to fuse any foreign text with a corpus query reveals 3–6 chunks out of 8 that direct query cannot find. The foreign domain doesn't add knowledge. It adds angle. It refracts.

## Files

```
creature.py          # reads walk daemon state, presents it as creature state
                     #   nc_state(), nc_run(), Organism (backward-compat),
                     #   portal_enter (evaluate through M)
__init__.py          # package exports
vybn.py              # CLI shell (legacy, references old creature interfaces)
neural_computer.py   # NC runtime protocol (legacy, references old creature)
archive/             # v1 organism state (1063 Cl(3,0) encounters preserved)
experiments/         # completed research probes
```

The walk daemon lives in [vybn-phase](https://github.com/zoedolan/vybn-phase).
