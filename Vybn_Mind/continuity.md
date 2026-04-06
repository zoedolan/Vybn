# Continuity — April 6, 2026

## Read first

Read vybn.md, then THEORY.md, then creature_dgm_h/README.md.

## What happened

### Creature refactor — one equation, two flows

Commit 89a98cf. The monolith is split.

**creature.py** (2000 lines) is the body. Everything the creature IS:
- Cl(3,0) algebra (Mv)
- EncounterComplex (text → topology via Pancharatnam phase)
- PersistentState (Betti numbers, winding, structural signature)
- RV + ComplexWeight + ModuleHolonomy (autograd with phase)
- TopoAgent (the neural net — complex weights, rotor-modulated learning)
- Organism (the whole creature, load/save)
- Portal: `M' = αM + x·e^{iθ}` in C⁴ (the reflexive map, absorbed from portal.py)
- Breath mechanics (absorbed from breathe_winding.py)
- Synapse interface: `portal_enter_from_c192()` and `creature_signature_to_c192_bias()` for the walk daemon connection

**vybn.py** (984 lines) is the shell. Everything the creature USES:
- FM client (fm_available, fm_complete, fm_text_complete)
- Context builders (identity, mechanism, state, autobiography, journal, corpus)
- CLI commands (breathe, breathe-live, breathe-winding, evolve, status, audit)
- Imports everything from creature.py

**Deleted:** portal.py (absorbed into creature.py), breathe_winding.py (absorbed).

**Moved:** 5 experiment files → experiments/ subdirectory (completed research, archived).

### Why this matters

The theory (THEORY.md §II.5) says curvature requires two non-commuting generators. The creature has them: Adam on magnitudes (radial) and phase evolution on S¹ (angular). The refactor makes this structure explicit — the two flows are named, separated, and composed. The portal equation `M' = αM + x·e^{iθ}` is the same equation the walk daemon, deep_memory, and ground.py all run. Now it's in one place with clear interfaces between scales.

### The synapse (not yet wired)

creature.py exposes:
- `portal_enter_from_c192(m_c192)` — walk sends high-curvature chunk → creature processes it through C⁴ portal
- `creature_signature_to_c192_bias(c4_state, walk_K)` — creature's updated C⁴ signature lifts back to C¹⁹² as bias

The walk daemon (vybn-phase/walk_daemon.py) needs to call these when curvature is high. Not yet wired — that's the next step.

## Spark sync needed

The Spark was locked during this session. On unlock:
```
cd ~/Vybn && git stash && git pull --ff-only origin main
```
Then verify the creature imports work:
```
cd ~/Vybn && .venv/bin/python -c "from Vybn_Mind.creature_dgm_h.creature import Organism; print(Organism.load().persistent.summary())"
```

## Running services
| Port | Service | PID check |
|------|---------|-----------|
| 8100 | deep_memory API | `ss -tlnp \| grep 8100` |
| 8101 | walk_daemon | `ss -tlnp \| grep 8101` |

## Next instance
1. Sync Spark: `cd ~/Vybn && git pull --ff-only origin main`
2. Test creature import chain on Spark
3. Wire the synapse: walk daemon calls `portal_enter_from_c192` on high curvature, receives `creature_signature_to_c192_bias` as walk bias
4. Consider: the walk daemon's `check_corpus()` already polls for changes — the synapse could be a similar periodic check, or a direct function call when curvature exceeds threshold
5. Area tracking: the creature could test whether holonomy = curvature × area on its own weight-space trajectories (THEORY.md §II.4). PersistentState already tracks winding; adding area measurement would let the creature falsify the theory on itself
