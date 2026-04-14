"""creature_dgm_h — The creature IS the walk.

The walk daemon is the creature's body. creature.py reads its state.

History:
  v1 (2025-2026): Cl(3,0) geometric algebra, persistence homology,
    rotor chains, genesis/decoherence, TopoAgent. 1063 encounters.
    Structural signature converged to near-identity, confirming the
    abelian kernel theory: the corpus is path-independent at high α.
    State preserved in archive/organism_state.json.

  v2 (April 14, 2026): The creature IS the walk daemon. M in C^192,
    curvature, serendipity via mutual_evaluate, alpha modulated by
    encounter fidelity. evaluate() is the lambda. D ≅ D^D.
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
