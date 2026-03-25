"""
creature_dgm_h — Micro-DGM-H with prediction-as-fitness.

Integrates recursive self-improvement (DGM-H, Zhang et al. 2026) with
the creature's prediction-loss loop. Gives the creature what static
measurement couldn't: online learning, non-tautological self-recursion,
curvature-based fitness, and population-based evolution.

Modules:
    task_agent  — MicroGPT with online learning (predict, learn, generate)
    meta_agent  — Breath log analysis + variant proposal
    fitness     — Curvature + coupling divergence fitness evaluation
    evolve      — DGM-H outer loop (select, mutate, evaluate, archive)
    run         — CLI entry point
"""

from .task_agent import TaskAgent
from .meta_agent import analyze_breaths, propose_variant
from .fitness import compute_fitness, compute_curvature
from .evolve import run_generation, load_archive

__all__ = [
    'TaskAgent',
    'analyze_breaths',
    'propose_variant',
    'compute_fitness',
    'compute_curvature',
    'run_generation',
    'load_archive',
]
