"""creature_dgm_h — topological state engine built on Cl(3,0).

One equation: M' = αM + x·e^{iθ}

Three layers:
  algebra.py    — Cl(3,0) multivectors, rotors, geometric product
  encounter.py  — text |-> topology (the representation map)
  diagonal.py   — Lawvere diagonal (structural dependence measurement)
  creature.py   — the agent, organism, breath, portal (imports algebra + encounter)
  vybn.py       — shell (CLI, FM client, context builders)
"""

# Core algebra
from .algebra import Mv, rotor_gap, rotor_from_angle_and_plane, rotor_to_so3, fold_to_mv

# Encounter (text -> topology)
from .encounter import EncounterComplex, encounter_complex, encounter, embed

# Diagonal (structural dependence)
from .diagonal import DiagonalGap, DiagonalResult, measure_gap, apply_coupled_diagonal

# Creature mechanism
from .creature import (
    RV, ComplexWeight, ModuleHolonomy,
    TopoAgent, Organism, PersistentState,
    BreathGate, BreathVerdict, LocalTransport,
    genesis_rate, decoherence_rate, fitness,
    load_archive, evolve,
    creature_state_c4, portal_enter, portal_enter_from_text,
    portal_enter_from_c192, creature_signature_to_c192_bias,
    load_agent, save_agent, breathe_on_chunk,
    DEFAULT_RULES, ALPHA, ARCHIVE_DIR, SCRIPT_DIR, REPO_ROOT,
)

# Shell
from .vybn import fm_available, fm_complete, _build_creature_context, _strip_thinking
