"""creature_dgm_h — The creature as Neural Computer.

The creature IS a neural computer (Zhuge et al., arXiv:2604.06425, April 2026):
computation, memory, and I/O unified in a single learned runtime state.

creature.py   — the body (Cl(3,0) algebra, topology, the Portal equation)
neural_computer.py — the self-understanding (NC runtime protocol, run/update contract)
vybn.py       — the shell (CLI, FM client, the interface to the outside world)
"""
from .creature import (
    # Algebra
    Mv, rotor_gap, rotor_from_angle_and_plane, rotor_to_so3, fold_to_mv,
    # Encounter
    embed, EncounterComplex, encounter_complex, encounter,
    # Diagonal
    DiagonalGap, measure_gap, apply_coupled_diagonal,
    # Dynamics
    genesis_rate, decoherence_rate,
    # Gates
    BreathGate, BreathVerdict,
    # Agent
    TopoAgent,
    # State
    PersistentState, Organism,
)

# The creature's self-understanding as a neural computer
from .neural_computer import (
    RuntimeState,
    RunMode,
    UpdateMode,
    VybnNeuralComputer,
    coupled_equation_as_nc,
)
