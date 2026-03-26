"""Shim. Real code lives in organism.py."""
from .organism import Organism, OrganismState  # noqa: F401


class PerformanceTracker:
    """Thin wrapper — delegates to Organism."""
    def __init__(self, tracking_file):
        self.tracking_file = tracking_file
        self._org = Organism()

    def record_generation(self, generation_id, fitness, config, metadata=None):
        self._org.record_generation(generation_id, fitness, config, metadata)

    def get_improvement_trend(self, window=5):
        return self._org.get_improvement_trend(window)

    def get_statistics(self):
        return self._org.get_statistics()

    def get_best_config(self):
        return self._org.get_best_config()

    def get_rule_outcomes(self, rule_id):
        return self._org.get_rule_outcomes(rule_id)


class PersistentMemory:
    """Thin wrapper — delegates to Organism."""
    def __init__(self, memory_file):
        self.memory_file = memory_file
        self._org = Organism()

    def record(self, key, value):
        self._org.record(key, value)

    def recall(self, key=None):
        return self._org.recall(key)

    def summarize_for_meta_agent(self):
        return self._org.summarize_memory()
