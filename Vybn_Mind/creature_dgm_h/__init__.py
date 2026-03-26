"""
creature_dgm_h — Micro-DGM-H with prediction-as-fitness.

Three living files:
    creature.py — The encounter and its memory: sensing, Cl(3,0)
                  geometry, FM client, multi-frame trace, across-breath
                  memory, mutation, rotor self-model, tension tracking.
    evolve.py   — DGM-H outer loop + cross-domain transfer.
                  Transfer is evolution across domain boundaries.
    task_agent.py — MicroGPT with scalar autograd.

Supporting:
    run.py        — CLI entry point.
    field.py      — Backward-compat shim re-exporting from creature.py.
    organism.py   — Backward-compat shim re-exporting from creature.py.
"""

from .creature import (
    Multivector, Field, BreathRecord, FrameReading, ChunkTrace,
    compute_encounter, compute_curvature,
    compute_loss_trajectory_curvature,
    compute_fitness, compute_prediction_fitness,
    improvement_at_k, default_embed_fn,
    fm_available, fm_complete, fm_stream,
    local_model,
    Organism, OrganismState, TensionMemory,
    analyze_breaths, propose_variant,
)
from .task_agent import TaskAgent
from .evolve import (
    run_generation, load_archive,
    export_hyperagent, import_hyperagent,
    select_transfer_agent,
)
