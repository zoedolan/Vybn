"""
evolve.py — DGM-H outer loop.

Darwin Gödel Machine / HyperAgents (Zhang et al. 2026,
https://arxiv.org/abs/2603.19461) applied to the creature.

The outer loop:
  1. Select a parent variant from the archive (fitness-weighted, diversity-incentivized)
  2. Mutate via the meta-agent (propose parameter changes based on breath analysis)
  3. Evaluate the child (staged: small test first, full test only if promising)
  4. Archive the result (config + fitness + breath log + lineage)

This file is the one piece we keep fixed initially. The meta-agent and
task agent can be modified by future generations; the evolutionary loop
itself stays stable. Per DGM-H: the mechanism that selects and evaluates
changes must be more stable than the changes it evaluates.
"""

import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from .task_agent import TaskAgent
from .meta_agent import analyze_breaths, propose_variant
from .fitness import compute_fitness, default_embed_fn


# ── Archive management ───────────────────────────────────────────────────

ARCHIVE_DIR = Path(__file__).resolve().parent / 'archive'


def load_archive(archive_path=None):
    """Load all variants from the archive directory.

    Returns:
        list of dicts, each a variant record with at least:
            - id: str
            - config: dict
            - fitness: float
            - generation: int
            - parent_id: str or None
            - timestamp: str
    """
    archive_path = Path(archive_path) if archive_path else ARCHIVE_DIR

    variants = []
    for f in sorted(archive_path.glob('variant_*.json')):
        try:
            variant = json.loads(f.read_text())
            variants.append(variant)
        except (json.JSONDecodeError, OSError):
            continue

    return variants


def archive_variant(config, fitness_result, breath_records=None,
                    generation=0, parent_id=None, archive_path=None):
    """Save a variant to the archive.

    Every variant gets saved with its fitness score, the config that
    produced it, and the breath log. Population-based search, not
    hill-climbing.

    Args:
        config: dict of hyperparameters
        fitness_result: dict from compute_fitness()
        breath_records: list of breath dicts (optional)
        generation: int, which generation this belongs to
        parent_id: str, ID of the parent variant (None for seed)
        archive_path: override archive directory

    Returns:
        str: variant ID
    """
    archive_path = Path(archive_path) if archive_path else ARCHIVE_DIR
    archive_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    variant_id = f"v_{ts}_{random.randint(1000, 9999)}"

    # Strip non-serializable keys from config
    clean_config = {k: v for k, v in config.items() if k != 'rationale'}
    rationale = config.get('rationale', [])

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
        'rationale': rationale,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'n_breaths': len(breath_records) if breath_records else 0,
    }

    out_path = archive_path / f"variant_{variant_id}.json"
    out_path.write_text(json.dumps(record, indent=2, default=str))

    # Also save breath log if provided
    if breath_records:
        log_path = archive_path / f"breaths_{variant_id}.jsonl"
        with open(log_path, 'w') as f:
            for b in breath_records:
                f.write(json.dumps(b, default=str) + '\n')

    return variant_id


# ── Parent selection ─────────────────────────────────────────────────────

def select_parent(archive):
    """Probabilistic selection weighted by fitness, with diversity incentive.

    Per DGM-H: downweight parents with many children to maintain population
    diversity. A variant that has already spawned many children gets less
    selection probability, even if it has high fitness.

    Args:
        archive: list of variant dicts

    Returns:
        dict: selected parent variant, or None if archive is empty
    """
    if not archive:
        return None

    # Count children per parent
    child_counts = {}
    for v in archive:
        pid = v.get('parent_id')
        if pid:
            child_counts[pid] = child_counts.get(pid, 0) + 1

    # Compute selection weights: fitness / (1 + n_children)
    weights = []
    for v in archive:
        fitness = max(v.get('fitness', 0.0), 0.001)
        n_children = child_counts.get(v['id'], 0)
        weight = fitness / (1.0 + n_children)
        weights.append(weight)

    # Normalize
    total = sum(weights)
    if total < 1e-12:
        return random.choice(archive)

    probs = [w / total for w in weights]

    # Weighted sample
    r = random.random()
    cumulative = 0.0
    for v, p in zip(archive, probs):
        cumulative += p
        if cumulative > r:
            return v

    return archive[-1]


# ── Mutation ─────────────────────────────────────────────────────────────

def mutate(parent_config, analysis):
    """Apply the meta-agent's proposed changes to the parent config.

    This is a thin wrapper around meta_agent.propose_variant().
    The mutation is deterministic given the analysis — no randomness
    beyond what the meta-agent's heuristics produce. If we want
    stochastic mutation later, add it here.

    Args:
        parent_config: dict of parent's hyperparameters
        analysis: dict from analyze_breaths()

    Returns:
        dict: child config with changes + rationale
    """
    child_config = propose_variant(analysis, parent_config)

    # Add small random perturbation to break ties and explore
    # This is the only stochastic component — everything else is
    # deterministic from the analysis.
    if 'learn_lr' in child_config:
        noise = random.gauss(0, child_config['learn_lr'] * 0.1)
        child_config['learn_lr'] = max(child_config['learn_lr'] + noise, 0.001)
        child_config['learn_lr'] = round(child_config['learn_lr'], 6)

    if 'temperature' in child_config:
        noise = random.gauss(0, 0.05)
        child_config['temperature'] = max(
            min(child_config['temperature'] + noise, 2.5), 0.1)
        child_config['temperature'] = round(child_config['temperature'], 4)

    return child_config


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate(child_config, test_texts, checkpoint_path=None,
             embed_fn=None, quick=False):
    """Staged evaluation of a variant config.

    Per DGM-H's staged protocol: run a small test first. If the variant
    looks unpromising (fitness below threshold), skip the full test to
    save compute. This is especially important here because each
    forward/backward pass through the scalar autograd is slow.

    Args:
        child_config: dict of hyperparameters
        test_texts: list of strings to evaluate on
        checkpoint_path: path to base checkpoint
        embed_fn: embedding function (optional)
        quick: if True, use minimal test set

    Returns:
        dict: fitness result from compute_fitness()
    """
    if embed_fn is None:
        embed_fn = default_embed_fn

    agent = TaskAgent(checkpoint_path=checkpoint_path, config=child_config)

    # Stage 1: quick test (first 2 texts)
    quick_texts = test_texts[:2] if len(test_texts) > 2 else test_texts

    external_texts = []
    self_texts = []

    for text in quick_texts:
        # Learn on the text (online fine-tuning)
        agent.learn(text,
                    steps=child_config.get('learn_steps', 5),
                    lr=child_config.get('learn_lr', 0.01))
        external_texts.append(text)

        # Generate from the model (self-recursion that isn't tautological)
        generated = agent.generate(
            prompt=text[:8],
            temperature=child_config.get('temperature', 1.0))
        if generated:
            self_texts.append(generated)

    quick_fitness = compute_fitness(
        external_texts, self_texts, agent.loss_history,
        embed_fn=embed_fn,
        alpha=child_config.get('alpha', 0.85))

    if quick or len(test_texts) <= 2:
        return quick_fitness

    # Stage 2: if quick fitness is above threshold, run full test
    if quick_fitness['fitness'] < 0.1:
        # Not promising enough for full evaluation
        quick_fitness['staged'] = 'quick_only'
        return quick_fitness

    # Full evaluation on remaining texts
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
        embed_fn=embed_fn,
        alpha=child_config.get('alpha', 0.85))
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
                   archive_path=None, embed_fn=None, breath_log_path=None):
    """One generation of the evolutionary loop.

    Select parents -> mutate -> evaluate -> archive -> repeat.

    Args:
        test_texts: list of strings as test corpus
        n_variants: number of child variants to produce
        checkpoint_path: path to base model checkpoint
        archive_path: path to archive directory
        embed_fn: embedding function (optional)
        breath_log_path: path to existing breath log for analysis

    Returns:
        dict with:
            - generation: int
            - variants: list of (variant_id, fitness) tuples
            - best_id: str, ID of best variant this generation
            - best_fitness: float
    """
    archive = load_archive(archive_path)

    # Determine generation number
    if archive:
        generation = max(v.get('generation', 0) for v in archive) + 1
    else:
        generation = 0

    # Analyze breath logs if available
    if breath_log_path and Path(breath_log_path).exists():
        analysis = analyze_breaths(breath_log_path)
    else:
        analysis = {
            'n_breaths': 0,
            'loss_trend': 'no_data',
            'curvature_trend': 'no_data',
            'mean_curvature': 0.0,
            'mean_loss': 0.0,
            'collapse_count': 0,
            'self_breath_ratio': 0.0,
            'recent_breaths': [],
        }

    results = []

    for i in range(n_variants):
        # Select parent
        parent = select_parent(archive)
        if parent:
            parent_config = parent.get('config', DEFAULT_CONFIG)
            parent_id = parent['id']
        else:
            parent_config = dict(DEFAULT_CONFIG)
            parent_id = None

        # Mutate
        child_config = mutate(parent_config, analysis)

        # Evaluate
        fitness_result = evaluate(
            child_config, test_texts,
            checkpoint_path=checkpoint_path,
            embed_fn=embed_fn,
            quick=(i > 0))  # Full eval only for first variant

        # Archive
        variant_id = archive_variant(
            child_config, fitness_result,
            generation=generation,
            parent_id=parent_id,
            archive_path=archive_path)

        results.append((variant_id, fitness_result['fitness']))
        print(f"  variant {i + 1}/{n_variants}: {variant_id} "
              f"fitness={fitness_result['fitness']:.4f} "
              f"(curv={fitness_result['curvature']:.4f})")

    # Find best
    best_id, best_fitness = max(results, key=lambda x: x[1])

    return {
        'generation': generation,
        'variants': results,
        'best_id': best_id,
        'best_fitness': best_fitness,
    }
