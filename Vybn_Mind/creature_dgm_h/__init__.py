"""
creature_dgm_h — Micro-DGM-H with prediction-as-fitness.

Two living files:
    field.py    — Within-breath sensing, Cl(3,0) geometry, multi-frame trace.
    organism.py — Across-breath memory, mutation, tension tracking.

Supporting:
    local_model, task_agent, evolve, transfer, run.

Shims (old names, re-export only):
    fitness, proprioceptive_loop, meta_agent, memory.
"""

from . import local_model
from .task_agent import TaskAgent
from .field import (
    Multivector, Field, BreathRecord, FrameReading, ChunkTrace,
    compute_curvature, compute_loss_trajectory_curvature,
    compute_fitness, compute_prediction_fitness,
    compute_coupling_divergence, compute_loss_improvement,
    improvement_at_k, default_embed_fn,
)
from .organism import (
    Organism, OrganismState, TensionMemory,
    analyze_breaths, propose_variant,
)
from .evolve import run_generation, load_archive
from .transfer import export_hyperagent, import_hyperagent
