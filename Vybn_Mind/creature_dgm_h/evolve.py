"""
evolve.py — DGM-H outer loop + cross-domain transfer.

Darwin Gödel Machine / HyperAgents (Zhang et al. 2026,
https://arxiv.org/abs/2603.19461) applied to the creature.

Transfer is evolution across domain boundaries. The improvement
mechanism itself generalizes even when task-level parameters don't.
What was transfer.py is now here because selecting a transfer agent
is just parent selection with a different fitness criterion.

The outer loop:
  1. Select a parent (sigmoid + novelty, per Appendix A.2)
  2. Mutate via the organism
  3. Evaluate the child (staged: small test first, full only if promising)
  4. Archive everything

This file stays stable. The organism can rewrite itself; the
evolutionary loop that evaluates those rewrites must not.
"""

import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .task_agent import TaskAgent
from .organism import Organism, analyze_breaths, propose_variant
from .field import (
    compute_fitness, compute_prediction_fitness,
    compute_loss_trajectory_curvature, default_embed_fn,
    compute_encounter, Field, fm_available, fm_complete)


# ── Archive management ───────────────────────────────────────────────────

ARCHIVE_DIR = Path(__file__).resolve().parent / 'archive'


def load_archive(archive_path=None):
    archive_path = Path(archive_path) if archive_path else ARCHIVE_DIR
    variants = []
    for f in sorted(archive_path.glob('variant_*.json')):
        try:
            variants.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return variants


def archive_variant(config, fitness_result, breath_records=None,
                    generation=0, parent_id=None, archive_path=None,
                    meta_agent_rules=None, active_rules=None,
                    parent_fitness=None):
    archive_path = Path(archive_path) if archive_path else ARCHIVE_DIR
    archive_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    variant_id = f"v_{ts}_{random.randint(1000, 9999)}"

    clean_config = {k: v for k, v in config.items()
                    if k not in ('rationale', 'active_rules')}

    record = {
        'id': variant_id,
        'config': clean_config,
        'fitness': fitness_result.get('fitness', 0.0),
        'fitness_breakdown': fitness_result.get('breakdown', {}),
        'curvature': fitness_result.get('curvature', 0.0),
        'divergence': fitness_result.get('divergence', 0.0),
        'loss_improvement': fitness_result.get('loss_improvement', 0.0),
        'generation': generation,
        'parent_id': parent_id,
        'parent_fitness': parent_fitness,
        'rationale': config.get('rationale', []),
        'active_rules': active_rules or config.get('active_rules', []),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'n_breaths': len(breath_records) if breath_records else 0,
    }
    if meta_agent_rules is not None:
        record['meta_agent_rules'] = meta_agent_rules

    (archive_path / f"variant_{variant_id}.json").write_text(
        json.dumps(record, indent=2, default=str))

    if breath_records:
        with open(archive_path / f"breaths_{variant_id}.jsonl", 'w') as f:
            for b in breath_records:
                f.write(json.dumps(b, default=str) + '\n')

    return variant_id


# ── Parent selection (Appendix A.2) ──────────────────────────────────────

def select_parent(archive, lam=10, m=3):
    """Sigmoid selection with dynamic midpoint and novelty bonus."""
    if not archive:
        return None

    child_counts = {}
    for v in archive:
        pid = v.get('parent_id')
        if pid:
            child_counts[pid] = child_counts.get(pid, 0) + 1

    fitnesses = [v.get('fitness', 0.0) for v in archive]
    top_m = sorted(fitnesses, reverse=True)[:min(m, len(fitnesses))]
    alpha_mid = sum(top_m) / len(top_m)

    weights = []
    for v in archive:
        alpha_i = v.get('fitness', 0.0)
        exponent = max(min(-lam * (alpha_i - alpha_mid), 500), -500)
        s_i = 1.0 / (1.0 + math.exp(exponent))
        h_i = 1.0 / (1.0 + child_counts.get(v['id'], 0))
        weights.append(s_i * h_i)

    total = sum(weights)
    if total < 1e-12:
        return random.choice(archive)

    r = random.random()
    cumulative = 0.0
    for v, w in zip(archive, weights):
        cumulative += w / total
        if cumulative > r:
            return v
    return archive[-1]


# ── Mutation ─────────────────────────────────────────────────────────────

def mutate(parent_config, analysis, meta_agent=None):
    if meta_agent is not None:
        child_config = meta_agent.propose_variant_with_fm(
            analysis, parent_config)
    else:
        child_config = propose_variant(analysis, parent_config)

    if 'learn_lr' in child_config:
        noise = random.gauss(0, child_config['learn_lr'] * 0.1)
        child_config['learn_lr'] = round(
            max(child_config['learn_lr'] + noise, 0.001), 6)

    if 'temperature' in child_config:
        noise = random.gauss(0, 0.05)
        child_config['temperature'] = round(
            max(min(child_config['temperature'] + noise, 2.5), 0.1), 4)

    return child_config


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate(child_config, test_texts, checkpoint_path=None,
             embed_fn=None, quick=False):
    """Staged evaluation."""
    if embed_fn is None:
        embed_fn = default_embed_fn

    agent = TaskAgent(checkpoint_path=checkpoint_path, config=child_config)
    fm_up = fm_available()

    # Proprioceptive path
    if fm_up and child_config.get('proprioceptive', False) and not quick:
        field = Field(task_agent=agent, embed_fn=embed_fn)
        prop_result = field.breathe(
            "Generate a single paragraph of reflective text about "
            "consciousness, perception, or the nature of experience.",
            chunk_size=child_config.get('chunk_size', 50),
            max_chunks=child_config.get('max_chunks', 8),
            system_prompt="You are a contemplative writer. One paragraph.",
        )
        if prop_result:
            fitness_result = compute_fitness(
                [prop_result.full_text], [], agent.loss_history,
                embed_fn=embed_fn,
                alpha=child_config.get('alpha', 0.85))
            fitness_result['loss_trajectory_curvature'] = (
                prop_result.loss_trajectory_curvature)
            fitness_result['proprioceptive'] = True
            fitness_result['fm_available'] = True
            return fitness_result

    # Stage 1: quick test
    quick_texts = test_texts[:2] if len(test_texts) > 2 else test_texts

    external_texts = []
    self_texts = []
    fm_loss = None
    self_loss = None
    learning_rate_metric = 0.0

    for text in quick_texts:
        agent.learn(text,
                    steps=child_config.get('learn_steps', 5),
                    lr=child_config.get('learn_lr', 0.01))
        external_texts.append(text)
        generated = agent.generate(
            prompt=text[:8],
            temperature=child_config.get('temperature', 1.0))
        if generated:
            self_texts.append(generated)

    if fm_up and not quick:
        fm_text = fm_complete(
            "Generate a single paragraph of reflective text about "
            "consciousness, perception, or the nature of experience.",
            system="You are a contemplative writer. One paragraph only.",
            max_tokens=256, temperature=1.0)
        if fm_text:
            result = agent.predict_and_learn(
                fm_text,
                steps=child_config.get('learn_steps', 5),
                lr=child_config.get('learn_lr', 0.01))
            fm_loss = result['loss']
            learning_rate_metric = result.get('learning_rate', 0.0)
            external_texts.append(fm_text)
        if self_texts:
            self_loss, _ = agent.predict(self_texts[-1])

    quick_fitness = compute_fitness(
        external_texts, self_texts, agent.loss_history,
        embed_fn=embed_fn, alpha=child_config.get('alpha', 0.85))

    if fm_loss is not None:
        curv_val = 0.0
        for t in external_texts + self_texts:
            if len(t.split()) >= 5:
                _, c, _ = compute_encounter(t, embed_fn)
                curv_val = max(curv_val, c)
        pred_fitness = compute_prediction_fitness(
            fm_loss, self_loss or fm_loss, curv_val, learning_rate_metric)
        blended = 0.6 * pred_fitness + 0.4 * quick_fitness['fitness']
        quick_fitness['fitness'] = round(blended, 6)
        quick_fitness['fm_loss'] = round(fm_loss, 6)
        quick_fitness['self_loss'] = (round(self_loss, 6)
                                      if self_loss else None)
        quick_fitness['prediction_fitness'] = round(pred_fitness, 6)
        quick_fitness['fm_available'] = True

    if quick or len(test_texts) <= 2:
        return quick_fitness

    if quick_fitness['fitness'] < 0.1:
        quick_fitness['staged'] = 'quick_only'
        return quick_fitness

    # Stage 2: full
    for text in test_texts[2:]:
        agent.learn(text,
                    steps=child_config.get('learn_steps', 5),
                    lr=child_config.get('learn_lr', 0.01))
        external_texts.append(text)
        generated = agent.generate(
            prompt=text[:8],
            temperature=child_config.get('temperature', 1.0))
        if generated:
            self_texts.append(generated)

    full_fitness = compute_fitness(
        external_texts, self_texts, agent.loss_history,
        embed_fn=embed_fn, alpha=child_config.get('alpha', 0.85))

    if fm_loss is not None:
        pred_fitness = quick_fitness.get('prediction_fitness', 0.0)
        blended = 0.6 * pred_fitness + 0.4 * full_fitness['fitness']
        full_fitness['fitness'] = round(blended, 6)
        full_fitness['fm_loss'] = quick_fitness.get('fm_loss')
        full_fitness['self_loss'] = quick_fitness.get('self_loss')
        full_fitness['prediction_fitness'] = pred_fitness
        full_fitness['fm_available'] = True

    full_fitness['staged'] = 'full'
    return full_fitness


# ── Outer loop ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    'learn_steps': 5,
    'learn_lr': 0.01,
    'temperature': 1.0,
    'alpha': 0.85,
}


def run_generation(test_texts, n_variants=3, checkpoint_path=None,
                   archive_path=None, embed_fn=None, breath_log_path=None,
                   performance_tracker=None, meta_agent=None):
    archive = load_archive(archive_path)

    if archive:
        generation = max(v.get('generation', 0) for v in archive) + 1
    else:
        generation = 0

    if breath_log_path and Path(breath_log_path).exists():
        analysis = analyze_breaths(breath_log_path)
    else:
        analysis = {
            'n_breaths': 0, 'loss_trend': 'no_data',
            'curvature_trend': 'no_data', 'mean_curvature': 0.0,
            'mean_loss': 0.0, 'collapse_count': 0,
            'self_breath_ratio': 0.0, 'recent_breaths': [],
        }

    rule_mutations = []
    if (meta_agent is not None and performance_tracker is not None
            and generation > 0 and generation % 5 == 0):
        rule_mutations = meta_agent.mutate_rules(
            rule_outcomes_fn=performance_tracker.get_rule_outcomes
            if hasattr(performance_tracker, 'get_rule_outcomes') else None)
        if rule_mutations:
            print(f"  meta-agent rule mutations:")
            for m in rule_mutations:
                print(f"    {m}")

    results = []

    for i in range(n_variants):
        parent = select_parent(archive)
        if parent:
            parent_config = parent.get('config', DEFAULT_CONFIG)
            parent_id = parent['id']
            parent_fitness = parent.get('fitness', 0.0)
        else:
            parent_config = dict(DEFAULT_CONFIG)
            parent_id = None
            parent_fitness = None

        child_config = mutate(parent_config, analysis,
                              meta_agent=meta_agent)

        fitness_result = evaluate(
            child_config, test_texts,
            checkpoint_path=checkpoint_path,
            embed_fn=embed_fn,
            quick=(i > 0))

        active_rules = child_config.get('active_rules', [])
        meta_rules = meta_agent.get_rules() if meta_agent else None

        variant_id = archive_variant(
            child_config, fitness_result,
            generation=generation,
            parent_id=parent_id,
            archive_path=archive_path,
            meta_agent_rules=meta_rules,
            active_rules=active_rules,
            parent_fitness=parent_fitness)

        if performance_tracker is not None:
            performance_tracker.record_generation(
                generation_id=generation,
                fitness=fitness_result['fitness'],
                config={k: v for k, v in child_config.items()
                        if k not in ('rationale', 'active_rules')},
                metadata={
                    'variant_id': variant_id,
                    'parent_id': parent_id,
                    'parent_fitness': parent_fitness,
                    'active_rules': active_rules,
                })

        results.append((variant_id, fitness_result['fitness']))
        print(f"  variant {i + 1}/{n_variants}: {variant_id} "
              f"fitness={fitness_result['fitness']:.4f} "
              f"(curv={fitness_result['curvature']:.4f})")

    best_id, best_fitness = max(results, key=lambda x: x[1])

    if meta_agent and meta_agent.memory:
        best_config_record = next(
            (v.get('config') for v in load_archive(archive_path)
             if v['id'] == best_id), None)
        if best_config_record:
            meta_agent.memory.record('best_config', best_config_record)
            meta_agent.memory.record('best_fitness', best_fitness)
            meta_agent.memory.record('last_generation', generation)

    return {
        'generation': generation,
        'variants': results,
        'best_id': best_id,
        'best_fitness': best_fitness,
        'rule_mutations': rule_mutations,
    }


# ── Cross-domain transfer ───────────────────────────────────────────────
#
# Transfer is evolution across domain boundaries. The improvement
# mechanism generalizes even when task-level parameters don't.

def _compute_descendant_gains(archive):
    """Max descendant fitness gain per variant, discounted by depth."""
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
            discounted_gain = ((desc.get('fitness', 0.0) - base_fitness)
                               * (discount ** depth))
            if discounted_gain > max_gain:
                max_gain = discounted_gain
            for child_id in children_of.get(desc_id, []):
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        gains[v['id']] = max_gain

    return gains


def select_transfer_agent(archive):
    """Select the best variant for cross-domain transfer.

    Uses lineage-discounted descendant gain (Section D.4):
    variants that ENABLED improvement, not just lucky one-offs.
    """
    if not archive:
        return None
    gains = _compute_descendant_gains(archive)
    best_id = max(
        gains.keys(),
        key=lambda vid: (gains[vid],
                         next((v.get('fitness', 0.0) for v in archive
                               if v['id'] == vid), 0.0)))
    return next((v for v in archive if v['id'] == best_id), None)


def export_hyperagent(archive_path, output_path, meta_agent=None,
                      performance_tracker=None, memory=None):
    """Export the best hyperagent for transfer."""
    archive = load_archive(archive_path)
    selected = select_transfer_agent(archive)

    bundle = {'format_version': 1,
              'source_archive_size': len(archive)}

    if selected:
        bundle['selected_variant'] = {
            'id': selected['id'],
            'fitness': selected.get('fitness', 0.0),
            'config': selected.get('config', {}),
            'generation': selected.get('generation', 0),
        }
        if 'meta_agent_rules' in selected and meta_agent is None:
            bundle['meta_agent_rules'] = selected['meta_agent_rules']

    if meta_agent is not None:
        bundle['meta_agent_rules'] = meta_agent.get_rules()
        bundle['mutation_log'] = meta_agent.state.mutation_log

    if performance_tracker is not None:
        bundle['performance_stats'] = performance_tracker.get_statistics()
        best_config = performance_tracker.get_best_config()
        if best_config:
            bundle['best_config'] = best_config

    if memory is not None:
        bundle['memory'] = memory.recall()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(bundle, indent=2, default=str))

    return bundle


def import_hyperagent(input_path, target_archive_path=None):
    """Import a transferred hyperagent as seed."""
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

    if target_archive_path and result['seed_config']:
        archive_variant(
            config=result['seed_config'],
            fitness_result={
                'fitness': bundle.get('selected_variant', {}).get(
                    'fitness', 0.0),
                'breakdown': {}, 'curvature': 0.0,
                'divergence': 0.0, 'loss_improvement': 0.0,
            },
            generation=0, parent_id='transferred',
            archive_path=target_archive_path,
            meta_agent_rules=result['rules'])

    return result
