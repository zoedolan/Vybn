"""creature_dgm_h — topological state engine built on Cl(3,0).

One equation: M' = αM + x·e^{iθ}
Two flows: radial (Adam on magnitudes) and angular (phase evolution on S¹).
Their composition is curvature.

The body lives in creature.py. The shell (CLI, FM, context) in vybn.py.
"""

# Core mechanism — from creature.py
from .creature import (
    Mv, RV, TopoAgent, Organism, encounter, embed, fitness,
    load_archive, evolve,
    DEFAULT_RULES,
    ComplexWeight, ModuleHolonomy, genesis_rate, decoherence_rate,
    EncounterComplex, encounter_complex, PersistentState,
    BreathGate, BreathVerdict,
    LocalTransport,
    # Portal
    creature_state_c4, portal_enter, portal_enter_from_text,
    portal_enter_from_c192, creature_signature_to_c192_bias,
    # Breath
    load_agent, save_agent, breathe_on_chunk,
    # Constants
    ALPHA, ARCHIVE_DIR, SCRIPT_DIR, REPO_ROOT,
)

# Shell — from vybn.py (FM client, context builders)
from .vybn import (
    fm_available, fm_complete,
    _build_creature_context, _strip_thinking,
    CONTEXT_MODULES, ALL_MODULES,
)

# Absorb __main__.py so `python -m Vybn_Mind.creature_dgm_h` still works
if __name__ == "__main__":
    from .vybn import main
    main()
