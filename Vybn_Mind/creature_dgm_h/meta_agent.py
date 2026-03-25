"""
meta_agent.py — Breath log analysis and metacognitive variant proposal.

Reads the creature's breath logs, computes trends in loss, curvature,
and collapse warnings, then proposes configuration modifications.

Following DGM-H (Zhang et al. 2026): the improvement mechanism is itself
a program that can be modified. The meta-agent's rules are stored as a
JSON rulebook that the evolution loop can mutate. This is metacognitive
self-modification (Section 3, p. 5): improving how we improve.

The heuristics are simple on purpose. A heuristic that says "increase
learning steps when loss is trending up" is honest about what it does.
We don't claim these heuristics are optimal — we claim they're legible
and their effects are measurable.

All self-modification is logged, archived, and auditable (Section 6).
"""

import copy
import json
import math
from pathlib import Path


# ── Breath analysis (unchanged from original) ────────────────────────────

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


# ── Default rulebook ─────────────────────────────────────────────────────
# Each rule is a JSON-serializable dict. The evolution loop can modify
# these — that's the metacognitive part.

DEFAULT_RULES = [
    {
        "id": "loss_trending_up",
        "condition": "loss_trend == 'increasing'",
        "action": "learn_steps",
        "direction": "increase",
        "magnitude": 2,
        "max_value": 20,
        "enabled": True,
        "rationale": "loss trending up, more steps to memorize recent input"
    },
    {
        "id": "curvature_dropping",
        "condition": "curvature_trend == 'decreasing'",
        "action": "alpha",
        "direction": "decrease",
        "magnitude": 0.05,
        "min_value": 0.5,
        "enabled": True,
        "rationale": "curvature dropping, lower alpha makes memory more responsive"
    },
    {
        "id": "self_recursion_flatline",
        "condition": "self_breath_ratio > 0.5 and mean_curvature < 0.05",
        "action": "temperature",
        "direction": "increase",
        "magnitude": 0.2,
        "max_value": 2.0,
        "enabled": True,
        "rationale": "high self-recursion + low curvature, increase diversity"
    },
    {
        "id": "collapse_warnings",
        "condition": "collapse_count > 2",
        "action": "learn_lr",
        "direction": "multiply",
        "magnitude": 0.5,
        "min_value": 0.001,
        "enabled": True,
        "rationale": "collapse warnings, stabilize"
    },
]


# ── Rule evaluation ──────────────────────────────────────────────────────

def _evaluate_condition(condition_str, analysis):
    """Evaluate a rule condition against breath analysis.

    Uses a restricted eval with only the analysis dict as namespace.
    This is safe because the condition strings are stored in our own
    rulebook, not user input.

    Args:
        condition_str: Python expression using analysis keys
        analysis: dict from analyze_breaths()

    Returns:
        bool
    """
    try:
        return bool(eval(condition_str, {"__builtins__": {}}, analysis))
    except Exception:
        return False


def _apply_rule(rule, config):
    """Apply a single rule's action to the config.

    Args:
        rule: dict with action, direction, magnitude, and optional
            min_value/max_value
        config: dict of hyperparameters (modified in place)

    Returns:
        str or None: rationale string if a change was made, else None
    """
    param = rule['action']
    direction = rule['direction']
    magnitude = rule['magnitude']
    old = config.get(param)
    if old is None:
        return None

    if direction == 'increase':
        new = old + magnitude
    elif direction == 'decrease':
        new = old - magnitude
    elif direction == 'multiply':
        new = old * magnitude
    else:
        return None

    # Clamp
    if 'max_value' in rule:
        new = min(new, rule['max_value'])
    if 'min_value' in rule:
        new = max(new, rule['min_value'])

    # Round floats
    if isinstance(new, float):
        new = round(new, 6)

    if new == old:
        return None

    config[param] = new
    return (f"{param} {old} -> {new}: {rule.get('rationale', rule['id'])}")


# ── MetaAgent class ──────────────────────────────────────────────────────

class MetaAgent:
    """Meta-agent with editable rules and persistent memory.

    The improvement mechanism is stored as a JSON rulebook. The evolution
    loop can propose modifications to the rules themselves, not just to the
    task agent's parameters. This is metacognitive self-modification
    (Section 3, p. 5).

    All rule changes are logged so they're auditable.
    """

    def __init__(self, rules=None, memory=None):
        """
        Args:
            rules: list of rule dicts, or None for DEFAULT_RULES.
            memory: PersistentMemory instance (optional).
        """
        self.rules = copy.deepcopy(rules) if rules else copy.deepcopy(DEFAULT_RULES)
        self.memory = memory
        self.mutation_log = []  # audit trail of rule mutations

    def propose_variant(self, analysis, current_config):
        """Apply rules to produce a modified config.

        Each rule whose condition matches the analysis fires in order.
        The meta-agent also consults persistent memory for context.

        Args:
            analysis: dict from analyze_breaths()
            current_config: dict with learn_steps, learn_lr, temperature, alpha

        Returns:
            dict with modified config + 'rationale' list + 'active_rules' list
        """
        config = dict(current_config)
        rationale = []
        active_rules = []

        # Defaults
        config.setdefault('learn_steps', 5)
        config.setdefault('learn_lr', 0.01)
        config.setdefault('temperature', 1.0)
        config.setdefault('alpha', 0.85)

        for rule in self.rules:
            if not rule.get('enabled', True):
                continue
            if _evaluate_condition(rule['condition'], analysis):
                change = _apply_rule(rule, config)
                if change:
                    rationale.append(change)
                    active_rules.append(rule['id'])

        # Consult persistent memory for additional context
        if self.memory:
            best_config = self.memory.recall('best_config')
            if best_config and not rationale:
                # No rules fired — consider drifting toward the best known config
                rationale.append(
                    "no rules fired; persistent memory available for reference")

        if not rationale:
            rationale.append("no changes proposed — metrics within normal range")

        config['rationale'] = rationale
        config['active_rules'] = active_rules
        return config

    def mutate_rules(self, performance_tracker):
        """Modify the rules themselves based on performance history.

        If a rule consistently produces variants with LOWER fitness than
        their parents, weaken its magnitude. If a rule consistently
        produces improvements, strengthen it. If a rule has no data,
        leave it alone.

        This is the metacognitive part — improving how we improve.
        All mutations are logged for auditability (Section 6).

        Args:
            performance_tracker: PerformanceTracker instance

        Returns:
            list of str: descriptions of mutations applied
        """
        mutations = []

        for rule in self.rules:
            if not rule.get('enabled', True):
                continue

            outcomes = performance_tracker.get_rule_outcomes(rule['id'])
            if len(outcomes) < 3:
                # Not enough data to judge this rule
                continue

            mean_delta = sum(outcomes) / len(outcomes)
            recent = outcomes[-3:]
            recent_mean = sum(recent) / len(recent)

            # If the rule consistently hurts performance, weaken it
            if recent_mean < -0.01 and mean_delta < 0:
                old_mag = rule['magnitude']
                if isinstance(old_mag, (int, float)) and old_mag > 0:
                    new_mag = round(old_mag * 0.5, 6)
                    # Don't let magnitude go to zero
                    if isinstance(old_mag, int):
                        new_mag = max(int(new_mag), 1)
                    else:
                        new_mag = max(new_mag, 0.001)
                    if new_mag != old_mag:
                        rule['magnitude'] = new_mag
                        desc = (f"weakened rule '{rule['id']}': "
                                f"magnitude {old_mag} -> {new_mag} "
                                f"(recent mean delta: {recent_mean:.4f})")
                        mutations.append(desc)

            # If the rule consistently helps, strengthen it
            elif recent_mean > 0.01 and mean_delta > 0:
                old_mag = rule['magnitude']
                if isinstance(old_mag, (int, float)):
                    new_mag = round(old_mag * 1.5, 6)
                    # Reasonable upper bounds
                    if isinstance(old_mag, int):
                        new_mag = min(int(new_mag), 10)
                    else:
                        new_mag = min(new_mag, 1.0)
                    if new_mag != old_mag:
                        rule['magnitude'] = new_mag
                        desc = (f"strengthened rule '{rule['id']}': "
                                f"magnitude {old_mag} -> {new_mag} "
                                f"(recent mean delta: {recent_mean:.4f})")
                        mutations.append(desc)

        if mutations:
            self.mutation_log.extend(mutations)
            # Store in persistent memory if available
            if self.memory:
                self.memory.record('last_rule_mutation', {
                    'mutations': mutations,
                    'n_rules': len(self.rules),
                })

        return mutations

    def get_rules(self):
        """Return a copy of the current rulebook."""
        return copy.deepcopy(self.rules)

    def save(self, path):
        """Serialize rules + mutation log to JSON for archival.

        Args:
            path: file path to write to
        """
        data = {
            'rules': self.rules,
            'mutation_log': self.mutation_log,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path, memory=None):
        """Load rules from JSON.

        Args:
            path: file path to read from
            memory: PersistentMemory instance (optional)

        Returns:
            MetaAgent instance
        """
        data = json.loads(Path(path).read_text())
        rules = data.get('rules', DEFAULT_RULES)
        agent = cls(rules=rules, memory=memory)
        agent.mutation_log = data.get('mutation_log', [])
        return agent


# ── Backward-compatible free functions ───────────────────────────────────
# These wrap the MetaAgent class so existing imports keep working.

_default_meta_agent = None


def _get_default_meta_agent():
    global _default_meta_agent
    if _default_meta_agent is None:
        _default_meta_agent = MetaAgent()
    return _default_meta_agent


def propose_variant(analysis, current_config):
    """Given breath analysis, propose a modified configuration.

    Backward-compatible wrapper around MetaAgent.propose_variant().
    """
    return _get_default_meta_agent().propose_variant(analysis, current_config)


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
