"""Stored-walk compatibility API; historical v1 state is explicit, not live.

See README.md for provenance, archive recovery, and limits.
"""
from .creature import (
    # The creature's state
    CreatureState,
    Organism,

    # Public API
    nc_state,
    nc_run,

    # Portal (evaluate through the creature's position)
    portal_enter,
    portal_enter_from_text,
    portal_enter_from_c192,
    creature_state_c4,
    creature_signature_to_c192_bias,

    # Backward-compatible stubs
    Mv, EncounterComplex, encounter_complex, encounter,
    DiagonalGap, measure_gap, apply_coupled_diagonal,
    genesis_rate, decoherence_rate,
    BreathGate, BreathVerdict,
    TopoAgent, PersistentState,
    embed, rotor_gap, rotor_from_angle_and_plane, rotor_to_so3, fold_to_mv,
    breathe_on_chunk, load_agent, save_agent,
)
