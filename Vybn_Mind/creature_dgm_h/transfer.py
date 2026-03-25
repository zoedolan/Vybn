"""
transfer.py — Export/import evolved hyperagent state for cross-domain transfer.

DGM-H (Zhang et al. 2026, Section 5.2-5.3) shows that the meta-level
innovations (performance tracking, persistent memory, evolved rules)
transfer across domains. A hyperagent trained on one task can bootstrap
improvement on a new task because the *improvement mechanism itself*
generalizes, even when the task-level parameters don't.

Export bundles:
  - meta-agent rules (the evolved rulebook)
  - performance history (what worked and what didn't)
  - persistent memory (synthesized insights and causal hypotheses)

Import seeds the new domain's meta-agent with these, then lets
evolution continue from there.

Transfer agent selection uses the lineage-discounted criterion
from Section D.4: select based on the maximum performance gain
achieved by descendants, discounted by lineage depth.
"""

import json
import math
from pathlib import Path


def _compute_descendant_gains(archive):
    """Compute the maximum descendant fitness gain for each variant.

    For each variant, find all its descendants (children, grandchildren, etc.)
    and compute the maximum fitness improvement relative to the variant itself,
    discounted by lineage depth.

    The discount factor is 0.9^depth — descendants further away contribute
    less, because the improvement may be due to intermediate mutations
    rather than the original variant's meta-agent rules.

    Args:
        archive: list of variant dicts

    Returns:
        dict mapping variant_id -> discounted_gain (float)
    """
    # Build parent -> children index
    children_of = {}
    by_id = {}
    for v in archive:
        by_id[v['id']] = v
        pid = v.get('parent_id')
        if pid:
            children_of.setdefault(pid, []).append(v['id'])

    discount = 0.9
    gains = {}

    for v in archive:
        base_fitness = v.get('fitness', 0.0)
        max_gain = 0.0

        # BFS through descendants
        queue = [(vid, 1) for vid in children_of.get(v['id'], [])]
        visited = {v['id']}

        while queue:
            desc_id, depth = queue.pop(0)
            if desc_id in visited:
                continue
            visited.add(desc_id)

            desc = by_id.get(desc_id)
            if desc is None:
                continue

            desc_fitness = desc.get('fitness', 0.0)
            discounted_gain = (desc_fitness - base_fitness) * (discount ** depth)
            if discounted_gain > max_gain:
                max_gain = discounted_gain

            for child_id in children_of.get(desc_id, []):
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        gains[v['id']] = max_gain

    return gains


def select_transfer_agent(archive):
    """Select the best variant for cross-domain transfer.

    Uses the lineage-discounted criterion from Section D.4:
    select based on the maximum performance gain achieved by its
    descendants, measured relative to the agent itself and discounted
    by lineage depth.

    This selects variants that *enabled* the most improvement in their
    lineage, not just variants with the highest absolute fitness.
    A variant that spawned a long chain of improvements is more
    transferable than a lucky one-off.

    Args:
        archive: list of variant dicts

    Returns:
        dict: the selected variant, or None if archive is empty
    """
    if not archive:
        return None

    gains = _compute_descendant_gains(archive)

    # Select the variant with the highest descendant gain
    # Tie-break with own fitness
    best_id = max(
        gains.keys(),
        key=lambda vid: (gains[vid],
                         next((v.get('fitness', 0.0) for v in archive
                               if v['id'] == vid), 0.0)))

    return next((v for v in archive if v['id'] == best_id), None)


def export_hyperagent(archive_path, output_path, meta_agent=None,
                      performance_tracker=None, memory=None):
    """Export the best hyperagent for transfer to a new domain.

    Bundles:
      - meta-agent rules (from the transfer-selected variant or current MetaAgent)
      - performance history (if PerformanceTracker provided)
      - persistent memory (if PersistentMemory provided)
      - the selected variant's config and fitness (as a reference point)

    Args:
        archive_path: path to archive directory
        output_path: path to write the export JSON
        meta_agent: MetaAgent instance (optional; if None, uses rules
            from the selected variant's archived rules)
        performance_tracker: PerformanceTracker instance (optional)
        memory: PersistentMemory instance (optional)

    Returns:
        dict: the exported bundle
    """
    from .evolve import load_archive

    archive = load_archive(archive_path)
    selected = select_transfer_agent(archive)

    bundle = {
        'format_version': 1,
        'source_archive_size': len(archive),
    }

    if selected:
        bundle['selected_variant'] = {
            'id': selected['id'],
            'fitness': selected.get('fitness', 0.0),
            'config': selected.get('config', {}),
            'generation': selected.get('generation', 0),
        }
        # Use archived rules from the selected variant if available
        if 'meta_agent_rules' in selected and meta_agent is None:
            bundle['meta_agent_rules'] = selected['meta_agent_rules']

    # Use current meta-agent rules if provided
    if meta_agent is not None:
        bundle['meta_agent_rules'] = meta_agent.get_rules()
        bundle['mutation_log'] = meta_agent.mutation_log

    # Performance statistics (not full history — just summary)
    if performance_tracker is not None:
        bundle['performance_stats'] = performance_tracker.get_statistics()
        best_config = performance_tracker.get_best_config()
        if best_config:
            bundle['best_config'] = best_config

    # Persistent memory entries
    if memory is not None:
        bundle['memory'] = memory.recall()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(bundle, indent=2, default=str))

    return bundle


def import_hyperagent(input_path, target_archive_path=None):
    """Import a transferred hyperagent as the seed for a new domain.

    Loads the exported bundle and returns components that can be used
    to initialize a new evolutionary run.

    Args:
        input_path: path to the exported JSON bundle
        target_archive_path: optional path to the new archive directory
            (for setting up the seed variant)

    Returns:
        dict with:
            - rules: list of rule dicts for MetaAgent
            - seed_config: dict of hyperparameters to start with
            - memory_entries: dict of persistent memory to pre-load
            - performance_stats: dict of source domain stats (for context)
            - mutation_log: list of prior rule mutations
    """
    bundle = json.loads(Path(input_path).read_text())

    result = {
        'rules': bundle.get('meta_agent_rules'),
        'seed_config': (bundle.get('best_config')
                        or bundle.get('selected_variant', {}).get('config')
                        or {}),
        'memory_entries': bundle.get('memory', {}),
        'performance_stats': bundle.get('performance_stats', {}),
        'mutation_log': bundle.get('mutation_log', []),
    }

    # If a target archive path is given, write the seed variant
    if target_archive_path and result['seed_config']:
        from .evolve import archive_variant
        archive_variant(
            config=result['seed_config'],
            fitness_result={
                'fitness': bundle.get('selected_variant', {}).get('fitness', 0.0),
                'breakdown': {},
                'curvature': 0.0,
                'divergence': 0.0,
                'loss_improvement': 0.0,
            },
            generation=0,
            parent_id='transferred',
            archive_path=target_archive_path,
            meta_agent_rules=result['rules'])

    return result
