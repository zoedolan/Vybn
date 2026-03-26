"""Shim. Real code lives in field.py."""
from .field import Field, default_embed_fn  # noqa: F401


def run_proprioceptive_breath(prompt, task_agent, chunk_size=50,
                              max_chunks=8, system_prompt=None,
                              embed_fn=None, temperature=0.7,
                              on_chunk=None):
    """Backward-compatible wrapper around Field.breathe()."""
    f = Field(task_agent=task_agent, embed_fn=embed_fn or default_embed_fn)
    record = f.breathe(prompt, chunk_size=chunk_size, max_chunks=max_chunks,
                       system_prompt=system_prompt, temperature=temperature,
                       on_chunk=on_chunk)
    if record is None:
        return None
    return {
        'full_text': record.full_text,
        'chunks': [{'chunk_num': c.chunk_num, 'text': c.text,
                     'mean_surprise': c.mean_surprise, 'contour': c.contour,
                     'curvature': c.curvature} for c in record.chunks],
        'trajectory': record.trajectory,
        'curvature': record.curvature,
        'curvature_angle': record.curvature_angle,
        'loss_trajectory_curvature': record.loss_trajectory_curvature,
        'injections': [c.annotation for c in record.chunks],
        'n_chunks': len(record.chunks),
    }


def run_ab_experiment(prompt, task_agent, n=5, chunk_size=50,
                      max_chunks=8, system_prompt=None, embed_fn=None,
                      temperature=0.7):
    """Backward-compatible wrapper around Field.compare_conditions()."""
    f = Field(task_agent=task_agent, embed_fn=embed_fn or default_embed_fn)
    return f.compare_conditions(prompt, n=n, chunk_size=chunk_size,
                                max_chunks=max_chunks,
                                system_prompt=system_prompt,
                                temperature=temperature)
