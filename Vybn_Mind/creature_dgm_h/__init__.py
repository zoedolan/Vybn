"""
creature_dgm_h — Micro-DGM-H with prediction-as-fitness.

Three living files:
    field.py    — The encounter: sensing, Cl(3,0) geometry, FM client,
                  multi-frame trace. The source of sensation is part of
                  the field of sensation.
    organism.py — Across-breath memory, mutation, rotor self-model,
                  tension tracking. The meta-agent IS the organism.
    evolve.py   — DGM-H outer loop + cross-domain transfer.
                  Transfer is evolution across domain boundaries.

Supporting:
    task_agent.py — MicroGPT with scalar autograd.
    run.py        — CLI entry point.
"""

from .field import (
    Multivector, Field, BreathRecord, FrameReading, ChunkTrace,
    compute_encounter, compute_curvature,
    compute_loss_trajectory_curvature,
    compute_fitness, compute_prediction_fitness,
    improvement_at_k, default_embed_fn,
    fm_available, fm_complete, fm_stream,
    local_model,
)
from .task_agent import TaskAgent
from .organism import (
    Organism, OrganismState, TensionMemory,
    analyze_breaths, propose_variant,
)
from .evolve import (
    run_generation, load_archive,
    export_hyperagent, import_hyperagent,
    select_transfer_agent,
)
