"""creature_dgm_h — the whole creature lives in vybn.py."""

from .vybn import (
    Mv as Multivector, Mv, V, Agent, Agent as TaskAgent,
    Organism, encounter, encounter as compute_encounter,
    encounter as compute_curvature,
    fitness as compute_fitness, embed as default_embed_fn,
    load_archive, evolve as run_generation,
    fm_available, fm_complete,
    DEFAULT_RULES, DEFAULT_CONFIG,
)
