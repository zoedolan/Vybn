"""creature_dgm_h — rotor-modulated character-level prediction with complex weight architecture."""
from .vybn import (
    Mv, RV, Agent, Organism, encounter, embed, fitness,
    load_archive, evolve, fm_available, fm_complete,
    DEFAULT_RULES,
    ComplexWeight, ModuleHolonomy, genesis_rate, decoherence_rate,
    _build_creature_context, _strip_thinking,
    BreathGate, BreathVerdict, CONTEXT_MODULES, ALL_MODULES,
    encounter_complex, EncounterComplex, PersistentState, TopoAgent,
)

# Absorb __main__.py so `python -m Vybn_Mind.creature_dgm_h` still works
if __name__ == "__main__":
    from .vybn import main
    main()
