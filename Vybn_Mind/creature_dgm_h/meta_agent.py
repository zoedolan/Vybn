"""
meta_agent.py — Breath log analysis and variant proposal.

Reads the creature's breath logs, computes trends in loss, curvature,
and collapse warnings, then proposes configuration modifications.

Following DGM-H (Zhang et al. 2026): the improvement mechanism is itself
a program that can be modified. Initially we keep evolve.py fixed and let
the meta-agent edit measurement/breathing parameters.

The heuristics here are simple on purpose. A heuristic that says
"increase learning steps when loss is trending up" is honest about
what it does. We don't claim these heuristics are optimal — we claim
they're legible and their effects are measurable.
"""

import json
import math
from pathlib import Path


def analyze_breaths(breath_log_path):
    """Read a JSONL breath log and compute trends.

    Args:
        breath_log_path: Path to a JSONL file where each line is a breath record.

    Returns:
        dict with:
            - n_breaths: total number of breaths
            - loss_trend: 'improving' | 'degrading' | 'stable' | 'no_data'
            - curvature_trend: 'improving' | 'degrading' | 'stable' | 'no_data'
            - mean_curvature: float
            - mean_loss: float (mean surprise, which is our loss proxy)
            - collapse_count: number of collapse warnings
            - self_breath_ratio: fraction of breaths that were self-recursion
            - recent_breaths: last 5 breath records (for debugging)
    """
    path = Path(breath_log_path)
    if not path.exists():
        return {
            'n_breaths': 0,
            'loss_trend': 'no_data',
            'curvature_trend': 'no_data',
            'mean_curvature': 0.0,
            'mean_loss': 0.0,
            'collapse_count': 0,
            'self_breath_ratio': 0.0,
            'recent_breaths': [],
        }

    breaths = []
    for line in path.read_text().strip().split('\n'):
        line = line.strip()
        if line:
            try:
                breaths.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not breaths:
        return {
            'n_breaths': 0,
            'loss_trend': 'no_data',
            'curvature_trend': 'no_data',
            'mean_curvature': 0.0,
            'mean_loss': 0.0,
            'collapse_count': 0,
            'self_breath_ratio': 0.0,
            'recent_breaths': [],
        }

    # Extract series
    losses = [b.get('mean_surprise', 0.0) for b in breaths]
    curvatures = [b.get('curvature', 0.0) for b in breaths]
    collapse_count = sum(1 for b in breaths if b.get('collapse_warning'))
    self_count = sum(1 for b in breaths if not b.get('external', True))

    return {
        'n_breaths': len(breaths),
        'loss_trend': _compute_trend(losses),
        'curvature_trend': _compute_trend(curvatures),
        'mean_curvature': sum(curvatures) / len(curvatures) if curvatures else 0.0,
        'mean_loss': sum(losses) / len(losses) if losses else 0.0,
        'collapse_count': collapse_count,
        'self_breath_ratio': self_count / len(breaths) if breaths else 0.0,
        'recent_breaths': breaths[-5:],
    }


def _compute_trend(values, window=5):
    """Classify a series as improving/degrading/stable.

    For loss: lower is better, so a negative slope is 'improving'.
    For curvature: higher is better, so a positive slope is 'improving'.
    We return the raw direction and let callers interpret.
    """
    if len(values) < 2:
        return 'no_data'

    recent = values[-window:]
    if len(recent) < 2:
        return 'no_data'

    # Simple linear regression slope
    n = len(recent)
    x_mean = (n - 1) / 2.0
    y_mean = sum(recent) / n
    num = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))

    if den < 1e-12:
        return 'stable'

    slope = num / den

    if abs(slope) < 0.001:
        return 'stable'
    elif slope > 0:
        return 'increasing'
    else:
        return 'decreasing'


def propose_variant(analysis, current_config):
    """Given breath analysis, propose a modified configuration.

    Uses simple heuristics — no pretense of intelligence here.
    Each rule has a clear rationale documented inline.

    Args:
        analysis: dict from analyze_breaths()
        current_config: dict with keys like learn_steps, learn_lr, temperature, alpha

    Returns:
        dict with modified config + 'rationale' explaining each change
    """
    config = dict(current_config)
    rationale = []

    # Default values if not in config
    config.setdefault('learn_steps', 5)
    config.setdefault('learn_lr', 0.01)
    config.setdefault('temperature', 1.0)
    config.setdefault('alpha', 0.85)

    # Rule 1: If loss (mean_surprise) is trending up, the model is forgetting
    # or seeing increasingly unfamiliar text. More learning steps might help
    # memorize the recent distribution. This is memorization, not generalization.
    if analysis.get('loss_trend') == 'increasing':
        old = config['learn_steps']
        config['learn_steps'] = min(old + 2, 20)
        if config['learn_steps'] != old:
            rationale.append(
                f"learn_steps {old} -> {config['learn_steps']}: "
                "loss trending up, more steps to memorize recent input"
            )

    # Rule 2: If curvature is dropping, the text trajectory is becoming
    # flatter — less conceptual turning. Adjusting alpha (memory decay)
    # makes the coupled equation more responsive to new input.
    if analysis.get('curvature_trend') == 'decreasing':
        old = config['alpha']
        config['alpha'] = max(old - 0.05, 0.5)
        if config['alpha'] != old:
            rationale.append(
                f"alpha {old:.2f} -> {config['alpha']:.2f}: "
                "curvature dropping, lower alpha makes memory more responsive"
            )

    # Rule 3: If self-recursion ratio is high and curvature is low,
    # the creature is talking to itself and flatlining. Higher temperature
    # increases generation diversity, which might break the loop.
    if (analysis.get('self_breath_ratio', 0) > 0.5
            and analysis.get('mean_curvature', 0) < 0.05):
        old = config['temperature']
        config['temperature'] = min(old + 0.2, 2.0)
        if config['temperature'] != old:
            rationale.append(
                f"temperature {old:.1f} -> {config['temperature']:.1f}: "
                "high self-recursion + low curvature, increase diversity"
            )

    # Rule 4: If there are collapse warnings, reduce learning rate to
    # prevent wild parameter swings.
    if analysis.get('collapse_count', 0) > 2:
        old = config['learn_lr']
        config['learn_lr'] = max(old * 0.5, 0.001)
        if config['learn_lr'] != old:
            rationale.append(
                f"learn_lr {old:.4f} -> {config['learn_lr']:.4f}: "
                f"{analysis['collapse_count']} collapse warnings, stabilize"
            )

    if not rationale:
        rationale.append("no changes proposed — metrics within normal range")

    config['rationale'] = rationale
    return config


def evaluate_variant(task_agent, variant_config, test_texts):
    """Run a variant configuration on test texts and return mean loss.

    This is a simple evaluation: apply the config, predict each text,
    return average loss. The fitness module handles the full multi-signal
    evaluation; this is just the loss component.

    Args:
        task_agent: TaskAgent instance (will be modified by config)
        variant_config: dict with learn_steps, learn_lr, temperature, etc.
        test_texts: list of strings to evaluate on

    Returns:
        float: mean prediction loss across test texts
    """
    if not test_texts:
        return 0.0

    # Apply config to task agent
    for key in ('learn_steps', 'learn_lr', 'temperature'):
        if key in variant_config:
            task_agent.config[key] = variant_config[key]

    total_loss = 0.0
    count = 0
    for text in test_texts:
        loss, contour = task_agent.predict(text)
        if contour:
            total_loss += loss
            count += 1

    return total_loss / max(count, 1)
