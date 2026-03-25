"""
memory.py — Performance tracking and persistent memory for meta-level learning.

The DGM-H paper (Zhang et al. 2026, Section 5.2, E.3) identifies these as
the key meta-level innovations that the hyperagent autonomously discovers:

1. PerformanceTracker — tracks metrics across generations, computes
   improvement trends. This is what enables compounding improvement:
   without knowing whether changes helped, the meta-agent is blind.

2. PersistentMemory — stores synthesized insights, causal hypotheses,
   and forward-looking plans. Instead of just logging numbers, the
   hyperagent stores analysis of WHY things worked. Later generations
   consult this memory to build on earlier discoveries and avoid
   repeating past mistakes.

We include both from the start because the paper shows they're essential
for transfer across domains (Section 5.2-5.3).

No external dependencies beyond the standard library.
"""

import json
import time
from pathlib import Path


class PerformanceTracker:
    """Tracks performance metrics across agent generations.

    Modeled on the PerformanceTracker that DGM-H autonomously discovers
    (Section 5.2, p. 11). We include it from the start because the paper
    shows it's essential for compounding improvement.
    """

    def __init__(self, tracking_file):
        """
        Args:
            tracking_file: path to JSON file for persistence.
                Created if it doesn't exist.
        """
        self.tracking_file = Path(tracking_file)
        self.history = self._load_history()

    def _load_history(self):
        """Load history from disk, or return empty list."""
        if self.tracking_file.exists():
            try:
                data = json.loads(self.tracking_file.read_text())
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save(self):
        """Persist history to disk."""
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        self.tracking_file.write_text(
            json.dumps(self.history, indent=2, default=str))

    def record_generation(self, generation_id, fitness, config,
                          metadata=None):
        """Record fitness and config for a generation.

        Args:
            generation_id: int or str identifying the generation
            fitness: float, composite fitness score
            config: dict of hyperparameters used
            metadata: optional dict of extra info (variant_id, rationale, etc.)
        """
        entry = {
            'generation': generation_id,
            'fitness': fitness,
            'config': config,
            'timestamp': time.time(),
        }
        if metadata:
            entry['metadata'] = metadata
        self.history.append(entry)
        self._save()

    def get_improvement_trend(self, window=5):
        """Calculate improvement trend using moving average.

        Returns:
            float: positive means improving (fitness increasing),
                   negative means degrading, near-zero means stable.
                   Returns 0.0 if insufficient data.
        """
        if len(self.history) < 2:
            return 0.0

        recent = self.history[-window:]
        if len(recent) < 2:
            return 0.0

        fitnesses = [e['fitness'] for e in recent]

        # Linear regression slope over the window
        n = len(fitnesses)
        x_mean = (n - 1) / 2.0
        y_mean = sum(fitnesses) / n
        num = sum((i - x_mean) * (fitnesses[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))

        if den < 1e-12:
            return 0.0

        return num / den

    def get_statistics(self):
        """Summary statistics across all recorded generations.

        Returns:
            dict with best, worst, average fitness, total generations,
            trend, and best_generation index.
        """
        if not self.history:
            return {
                'best': 0.0,
                'worst': 0.0,
                'average': 0.0,
                'total_generations': 0,
                'trend': 0.0,
                'best_generation': None,
            }

        fitnesses = [e['fitness'] for e in self.history]
        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])

        return {
            'best': max(fitnesses),
            'worst': min(fitnesses),
            'average': sum(fitnesses) / len(fitnesses),
            'total_generations': len(self.history),
            'trend': self.get_improvement_trend(),
            'best_generation': self.history[best_idx]['generation'],
        }

    def get_best_config(self):
        """Return the config that produced the best fitness.

        Returns:
            dict or None if no history.
        """
        if not self.history:
            return None
        best = max(self.history, key=lambda e: e['fitness'])
        return best.get('config')

    def get_rule_outcomes(self, rule_id):
        """Get fitness outcomes for variants produced by a specific rule.

        Looks at metadata.active_rules to find entries where this rule fired,
        then returns the fitness delta relative to the parent.

        Args:
            rule_id: str identifying the rule

        Returns:
            list of floats (fitness deltas). Positive means the rule
            helped; negative means it hurt.
        """
        deltas = []
        for entry in self.history:
            meta = entry.get('metadata', {}) or {}
            active_rules = meta.get('active_rules', [])
            parent_fitness = meta.get('parent_fitness')
            if rule_id in active_rules and parent_fitness is not None:
                deltas.append(entry['fitness'] - parent_fitness)
        return deltas


class PersistentMemory:
    """Stores synthesized insights, causal hypotheses, and forward-looking plans.

    The paper (Section 5.2, E.3, p. 12) shows this is critical: instead of
    just logging numbers, the hyperagent stores analysis of WHY things worked.
    Later generations consult this memory to build on earlier discoveries
    and avoid repeating past mistakes.
    """

    def __init__(self, memory_file):
        """
        Args:
            memory_file: path to JSON file for persistence.
                Created if it doesn't exist.
        """
        self.memory_file = Path(memory_file)
        self.entries = self._load()

    def _load(self):
        """Load entries from disk, or return empty dict."""
        if self.memory_file.exists():
            try:
                data = json.loads(self.memory_file.read_text())
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save(self):
        """Persist entries to disk."""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(
            json.dumps(self.entries, indent=2, default=str))

    def record(self, key, value):
        """Store a named insight with timestamp.

        Args:
            key: str identifying the insight (e.g. 'best_alpha_range',
                'collapse_cause', 'temperature_hypothesis')
            value: any JSON-serializable value
        """
        self.entries[key] = {
            'value': value,
            'timestamp': time.time(),
        }
        self._save()

    def recall(self, key=None):
        """Retrieve stored insights.

        Args:
            key: if provided, return that entry's value.
                If None, return all entries.

        Returns:
            The stored value for the key, or dict of all entries,
            or None if key not found.
        """
        if key is None:
            return dict(self.entries)
        entry = self.entries.get(key)
        if entry is None:
            return None
        return entry.get('value')

    def summarize_for_meta_agent(self):
        """Generate a summary string for the meta-agent to consume.

        Includes: stored insights organized by recency, with keys
        and values. The meta-agent uses this context when proposing
        variants — it's the mechanism by which past discoveries
        influence future exploration.

        Returns:
            str: human-readable summary of all stored insights.
        """
        if not self.entries:
            return "No stored insights yet."

        # Sort by timestamp (most recent first)
        sorted_keys = sorted(
            self.entries.keys(),
            key=lambda k: self.entries[k].get('timestamp', 0),
            reverse=True)

        lines = [f"Persistent memory ({len(self.entries)} insights):"]
        for key in sorted_keys:
            entry = self.entries[key]
            val = entry.get('value', '')
            # Truncate long values for the summary
            val_str = str(val)
            if len(val_str) > 200:
                val_str = val_str[:200] + '...'
            lines.append(f"  - {key}: {val_str}")

        return '\n'.join(lines)
