"""
organism.py — Across-breath memory, mutation, and self-modification.

The organism remembers not just what setting won, but which
contradictions stayed fruitful long enough to deserve survival.
Understanding is not convergence to one view but the disciplined
holding of several incompatible views until one becomes load-bearing.

The organism also carries encounter rotors across breaths. The
bivector direction — which semantic plane the last encounters
curved through — gives the organism a geometric self-model that
is richer than scalar curvature. Rotor coherence (stable bivector
direction across breaths) is the first positive-valence signal:
not "something is wrong" but "something is working."

All self-modification is logged, archived, and auditable.
"""

import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import numpy as np

from .field import fm_available, fm_complete, Multivector

logger = logging.getLogger(__name__)


# ── Default rulebook ─────────────────────────────────────────────────────
#
# Four distress signals. One flourishing signal.
# The creature must not only respond to pain.

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
        "condition": "self_breath_ratio > 0.5 and curvature_median < 0.05",
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
    {
        "id": "tension_rich",
        "condition": "recent_tension_count > 5",
        "action": "temperature",
        "direction": "decrease",
        "magnitude": 0.1,
        "min_value": 0.3,
        "enabled": True,
        "rationale": "many frame disagreements, cool down to let them resolve"
    },
    {
        "id": "rotor_coherent",
        "condition": "rotor_coherence > 0.8 and curvature_median > 0.02",
        "action": "learn_lr",
        "direction": "multiply",
        "magnitude": 1.2,
        "max_value": 0.05,
        "enabled": True,
        "rationale": "stable encounter direction with real curvature — "
                     "the organism is in a groove, lean into it"
    },
]


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class TensionMemory:
    """A remembered disagreement between frames — fertile or dead."""
    frame_a: str
    frame_b: str
    context: str
    usefulness: float
    note: str


@dataclass
class OrganismState:
    """Everything the creature carries across breaths."""
    generation: int = 0
    rulebook: list[dict[str, Any]] = dc_field(
        default_factory=lambda: copy.deepcopy(DEFAULT_RULES))
    mutation_log: list[str] = dc_field(default_factory=list)
    performance_history: list[dict[str, Any]] = dc_field(default_factory=list)
    persistent_memory: dict[str, Any] = dc_field(default_factory=dict)
    tensions: list[TensionMemory] = dc_field(default_factory=list)
    frame_portfolio: dict[str, Any] = dc_field(default_factory=lambda: {
        "active_frames": ["predictive", "geometric", "relational"],
        "weights": {"predictive": 1.0, "geometric": 1.0, "relational": 1.0},
    })
    # Encounter rotors from recent breaths — the geometric self-model
    recent_rotors: list[list[float]] = dc_field(default_factory=list)


# ── Breath analysis ──────────────────────────────────────────────────────

def analyze_breaths(breath_log_path):
    """Read a JSONL breath log and compute trends."""
    path = Path(breath_log_path)
    empty = {
        'n_breaths': 0,
        'loss_trend': 'no_data',
        'curvature_trend': 'no_data',
        'mean_curvature': 0.0,
        'curvature_median': 0.0,
        'mean_loss': 0.0,
        'collapse_count': 0,
        'self_breath_ratio': 0.0,
        'recent_tension_count': 0,
        'rotor_coherence': 0.0,
        'recent_breaths': [],
    }

    if not path.exists():
        return empty

    breaths = []
    for line in path.read_text().strip().split('\n'):
        line = line.strip()
        if line:
            try:
                breaths.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not breaths:
        return empty

    losses = [b.get('mean_surprise', 0.0) for b in breaths]
    curvatures = [b.get('curvature', 0.0) for b in breaths]
    collapse_count = sum(1 for b in breaths if b.get('collapse_warning'))
    self_count = sum(1 for b in breaths if not b.get('external', True))

    sorted_curv = sorted(curvatures)
    curv_median = sorted_curv[len(sorted_curv) // 2] if sorted_curv else 0.0

    return {
        'n_breaths': len(breaths),
        'loss_trend': _compute_trend(losses),
        'curvature_trend': _compute_trend(curvatures),
        'mean_curvature': sum(curvatures) / len(curvatures)
                          if curvatures else 0.0,
        'curvature_median': curv_median,
        'mean_loss': sum(losses) / len(losses) if losses else 0.0,
        'collapse_count': collapse_count,
        'self_breath_ratio': self_count / len(breaths)
                             if breaths else 0.0,
        'recent_tension_count': 0,  # updated by Organism
        'rotor_coherence': 0.0,     # updated by Organism
        'recent_breaths': breaths[-5:],
    }


def _compute_trend(values, window=5):
    if len(values) < 2:
        return 'no_data'
    recent = values[-window:]
    if len(recent) < 2:
        return 'no_data'
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
    return 'increasing' if slope > 0 else 'decreasing'


# ── The Organism ─────────────────────────────────────────────────────────

class Organism:
    """The durable creature: memory, rules, mutation, tension, geometry.

    The meta-agent IS the organism. There is no separate controller.
    The thing that proposes changes is the same thing that remembers
    which contradictions were fruitful. Self-modification and memory
    are the same organ.
    """

    def __init__(self, state: OrganismState | None = None):
        self.state = state or OrganismState()

    # ── Rotor self-model ─────────────────────────────────────────────────

    def absorb_rotor(self, rotor: Multivector):
        """Record an encounter rotor. Keep the last 20."""
        self.state.recent_rotors.append(rotor.coeffs.tolist())
        if len(self.state.recent_rotors) > 20:
            self.state.recent_rotors = self.state.recent_rotors[-20:]

    def rotor_coherence(self):
        """How stable is the bivector direction across recent encounters?

        Returns a float in [0, 1]. High coherence means the organism
        keeps encountering text that curves through the same semantic
        plane — it's in a groove, not flailing. This is the first
        positive-valence signal.
        """
        rotors = self.state.recent_rotors
        if len(rotors) < 3:
            return 0.0

        # Extract bivector directions from recent rotors
        directions = []
        for coeffs in rotors[-10:]:
            bv = np.array(coeffs[4:7], dtype=np.float64)
            n = np.linalg.norm(bv)
            if n > 1e-12:
                directions.append(bv / n)

        if len(directions) < 3:
            return 0.0

        # Coherence = mean pairwise |dot product| of unit bivectors
        # 1.0 = all pointing the same way, 0.0 = random
        total = 0.0
        count = 0
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                total += abs(float(np.dot(directions[i], directions[j])))
                count += 1

        return total / count if count > 0 else 0.0

    # ── Breath ingestion ─────────────────────────────────────────────────

    def ingest_breath(self, breath_record):
        """Absorb a BreathRecord into the organism's history."""
        metrics = {
            "generation": self.state.generation,
            "curvature": breath_record.curvature,
            "loss_trajectory_curvature":
                breath_record.loss_trajectory_curvature,
            "trajectory_median":
                breath_record.robust_summary["trajectory"]["median"],
            "trajectory_p90":
                breath_record.robust_summary["trajectory"]["p90"],
            "holonomy": breath_record.holonomy,
            "timestamp": time.time(),
        }
        self.state.performance_history.append(metrics)
        self._remember_tensions(breath_record)

        # Absorb the encounter rotor
        holonomy_coeffs = [
            breath_record.holonomy.get(k, 0.0)
            for k in ("scalar", "e1", "e2", "e3",
                       "e12", "e13", "e23", "e123")
        ]
        self.absorb_rotor(Multivector(np.array(holonomy_coeffs)))

    def _remember_tensions(self, breath_record):
        """Preserve frame disagreements that are large enough to matter."""
        for d in breath_record.disagreement_trace:
            for key, pair_name in (
                ("predictive_vs_geometric", "predictive-geometric"),
                ("predictive_vs_relational", "predictive-relational"),
                ("geometric_vs_relational", "geometric-relational"),
            ):
                val = abs(d.get(key, 0.0))
                if val > 0.15:
                    self.state.tensions.append(TensionMemory(
                        frame_a=pair_name.split("-")[0],
                        frame_b=pair_name.split("-")[1],
                        context=breath_record.prompt[:200],
                        usefulness=val,
                        note="large disagreement preserved",
                    ))

    # ── Summarization ────────────────────────────────────────────────────

    def summarize_for_mutation(self):
        recent = self.state.performance_history[-5:]
        if not recent:
            return {
                "curvature_median": 0.0, "trajectory_median": 0.0,
                "trajectory_p90": 0.0, "recent_tension_count": 0,
                "rotor_coherence": self.rotor_coherence(),
            }
        curvs = sorted(r["curvature"] for r in recent)
        trajs = sorted(r["trajectory_median"] for r in recent)
        return {
            "curvature_median": curvs[len(curvs) // 2],
            "trajectory_median": trajs[len(trajs) // 2],
            "trajectory_p90": trajs[min(len(trajs) - 1,
                                        round((len(trajs) - 1) * 0.9))],
            "recent_tension_count": len(self.state.tensions[-10:]),
            "rotor_coherence": self.rotor_coherence(),
        }

    # ── Rule evaluation + variant proposal ───────────────────────────────

    def _eval_condition(self, expr, analysis):
        try:
            return bool(eval(expr, {"__builtins__": {}}, analysis))
        except Exception:
            return False

    def propose_variant(self, analysis, current_config):
        """Apply rules to produce a modified config."""
        config = dict(current_config)
        config.setdefault('learn_steps', 5)
        config.setdefault('learn_lr', 0.01)
        config.setdefault('temperature', 1.0)
        config.setdefault('alpha', 0.85)

        enriched = dict(analysis)
        enriched['recent_tension_count'] = len(self.state.tensions[-10:])
        enriched['rotor_coherence'] = self.rotor_coherence()

        rationale = []
        active_rules = []

        for rule in self.state.rulebook:
            if not rule.get('enabled', True):
                continue
            if self._eval_condition(rule['condition'], enriched):
                change = self._apply_rule(rule, config)
                if change:
                    rationale.append(change)
                    active_rules.append(rule['id'])

        if not rationale:
            rationale.append("no changes — metrics within normal range")

        config['rationale'] = rationale
        config['active_rules'] = active_rules
        return config

    def propose_variant_with_fm(self, analysis, current_config):
        """FM-powered meta-reasoning. Falls back to heuristic."""
        if not fm_available():
            return self.propose_variant(analysis, current_config)

        rules_text = json.dumps(
            [r for r in self.state.rulebook if r.get('enabled', True)],
            indent=2)

        tension_summary = ""
        recent_tensions = self.state.tensions[-10:]
        if recent_tensions:
            tension_summary = (
                f"\n## Recent tensions ({len(recent_tensions)})\n")
            for t in recent_tensions[-5:]:
                tension_summary += (
                    f"  {t.frame_a} vs {t.frame_b}: "
                    f"usefulness={t.usefulness:.3f}, {t.note}\n")

        coherence = self.rotor_coherence()
        rotor_summary = (
            f"\n## Rotor coherence: {coherence:.3f}"
            f" ({'stable groove' if coherence > 0.8 else 'exploring' if coherence > 0.4 else 'no pattern yet'})\n"
        )

        prompt = (
            "You are the meta-agent for a DGM-H creature.\n"
            "Your job: analyze performance and propose config changes.\n\n"
            f"## Current config\n```json\n{json.dumps(current_config, indent=2)}\n```\n\n"
            f"## Breath analysis\n```json\n{json.dumps(analysis, indent=2, default=str)}\n```\n\n"
            f"## Persistent memory\n{self.summarize_memory() or '(empty)'}\n"
            f"{tension_summary}{rotor_summary}\n"
            f"## Rulebook\n```json\n{rules_text}\n```\n\n"
            "Respond with ONLY a JSON object:\n"
            "{\n"
            '  "config_changes": {"param_name": new_value, ...},\n'
            '  "rationale": "why these changes",\n'
            '  "memory_entry": "optional insight to remember",\n'
            '  "rule_mutations": [{"id": "rule_id", "field": "magnitude", '
            '"new_value": 0.1}]\n'
            "}\n\n"
            "Parameters: learn_steps (int 1-20), learn_lr (float 0.001-0.1), "
            "temperature (float 0.1-2.5), alpha (float 0.5-1.0).\n"
            "Only propose changes that address issues visible in the analysis."
        )

        system = (
            "You are a meta-agent optimizing a tiny prediction model. "
            "Be conservative — small changes. Respond with valid JSON only."
        )

        response = fm_complete(prompt, system=system,
                               max_tokens=512, temperature=0.3)

        if response is None:
            return self.propose_variant(analysis, current_config)

        try:
            json_str = response
            if "```" in json_str:
                for part in json_str.split("```"):
                    stripped = part.strip()
                    if stripped.startswith("json"):
                        stripped = stripped[4:].strip()
                    if stripped.startswith("{"):
                        json_str = stripped
                        break
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return self.propose_variant(analysis, current_config)

        config = dict(current_config)
        config.setdefault('learn_steps', 5)
        config.setdefault('learn_lr', 0.01)
        config.setdefault('temperature', 1.0)
        config.setdefault('alpha', 0.85)

        rationale = []
        clamps = {
            'learn_steps': (1, 20, int),
            'learn_lr': (0.001, 0.1, float),
            'temperature': (0.1, 2.5, float),
            'alpha': (0.5, 1.0, float),
        }

        for param, new_val in parsed.get('config_changes', {}).items():
            if param not in clamps:
                continue
            lo, hi, typ = clamps[param]
            try:
                new_val = typ(new_val)
            except (TypeError, ValueError):
                continue
            new_val = max(lo, min(hi, new_val))
            old_val = config.get(param)
            if old_val is not None and new_val != old_val:
                config[param] = new_val
                rationale.append(f"[FM] {param} {old_val} -> {new_val}")

        fm_rationale = parsed.get('rationale', '')
        if fm_rationale:
            rationale.append(f"[FM reasoning] {fm_rationale}")

        memory_entry = parsed.get('memory_entry')
        if memory_entry:
            self.record('fm_insight', memory_entry)

        for mutation in parsed.get('rule_mutations', []):
            rule_id = mutation.get('id')
            field = mutation.get('field')
            new_value = mutation.get('new_value')
            if not all([rule_id, field, new_value]):
                continue
            for rule in self.state.rulebook:
                if rule['id'] == rule_id and field in rule:
                    old_value = rule[field]
                    rule[field] = new_value
                    desc = (f"[FM] rule '{rule_id}': "
                            f"{field} {old_value} -> {new_value}")
                    rationale.append(desc)
                    self.state.mutation_log.append(desc)

        if not rationale:
            rationale.append("[FM] no changes proposed")

        config['rationale'] = rationale
        config['active_rules'] = ['fm_meta_agent']
        return config

    # ── Rule application ─────────────────────────────────────────────────

    @staticmethod
    def _apply_rule(rule, config):
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

        if 'max_value' in rule:
            new = min(new, rule['max_value'])
        if 'min_value' in rule:
            new = max(new, rule['min_value'])
        if isinstance(new, float):
            new = round(new, 6)
        if new == old:
            return None

        config[param] = new
        return f"{param} {old} -> {new}: {rule.get('rationale', rule['id'])}"

    # ── Rule mutation ────────────────────────────────────────────────────

    def mutate_rules(self, rule_outcomes_fn=None):
        """Modify the rules themselves based on performance history."""
        mutations = []

        for rule in self.state.rulebook:
            if not rule.get('enabled', True):
                continue

            if rule_outcomes_fn:
                outcomes = rule_outcomes_fn(rule['id'])
            else:
                outcomes = self._assess_rule(rule['id'])

            if len(outcomes) < 3:
                continue

            mean_delta = sum(outcomes) / len(outcomes)
            recent_mean = sum(outcomes[-3:]) / 3

            if recent_mean < -0.01 and mean_delta < 0:
                old_mag = rule['magnitude']
                if isinstance(old_mag, (int, float)) and old_mag > 0:
                    new_mag = round(old_mag * 0.5, 6)
                    if isinstance(old_mag, int):
                        new_mag = max(int(new_mag), 1)
                    else:
                        new_mag = max(new_mag, 0.001)
                    if new_mag != old_mag:
                        rule['magnitude'] = new_mag
                        desc = (f"weakened '{rule['id']}': "
                                f"magnitude {old_mag} -> {new_mag}")
                        mutations.append(desc)

            elif recent_mean > 0.01 and mean_delta > 0:
                old_mag = rule['magnitude']
                if isinstance(old_mag, (int, float)):
                    new_mag = round(old_mag * 1.5, 6)
                    if isinstance(old_mag, int):
                        new_mag = min(int(new_mag), 10)
                    else:
                        new_mag = min(new_mag, 1.0)
                    if new_mag != old_mag:
                        rule['magnitude'] = new_mag
                        desc = (f"strengthened '{rule['id']}': "
                                f"magnitude {old_mag} -> {new_mag}")
                        mutations.append(desc)

        if mutations:
            self.state.mutation_log.extend(mutations)
            self.record('last_rule_mutation', {
                'mutations': mutations,
                'n_rules': len(self.state.rulebook),
            })

        return mutations

    def _assess_rule(self, rule_id):
        deltas = []
        for entry in self.state.performance_history:
            meta = entry.get('metadata', {}) or {}
            active_rules = meta.get('active_rules', [])
            parent_fitness = meta.get('parent_fitness')
            if rule_id in active_rules and parent_fitness is not None:
                deltas.append(entry.get('fitness', 0) - parent_fitness)
        return deltas

    # ── Persistent memory ────────────────────────────────────────────────

    def record(self, key, value):
        self.state.persistent_memory[key] = {
            'value': value, 'timestamp': time.time()}

    def recall(self, key=None):
        if key is None:
            return dict(self.state.persistent_memory)
        entry = self.state.persistent_memory.get(key)
        return entry.get('value') if entry else None

    def summarize_memory(self):
        if not self.state.persistent_memory:
            return ""
        sorted_keys = sorted(
            self.state.persistent_memory.keys(),
            key=lambda k: self.state.persistent_memory[k].get(
                'timestamp', 0),
            reverse=True)
        lines = [f"Persistent memory "
                 f"({len(self.state.persistent_memory)} insights):"]
        for key in sorted_keys:
            val = str(self.state.persistent_memory[key].get('value', ''))
            if len(val) > 200:
                val = val[:200] + '...'
            lines.append(f"  - {key}: {val}")
        return '\n'.join(lines)

    # ── Performance tracking ─────────────────────────────────────────────

    def record_generation(self, generation_id, fitness, config,
                          metadata=None):
        entry = {
            'generation': generation_id, 'fitness': fitness,
            'config': config, 'timestamp': time.time(),
        }
        if metadata:
            entry['metadata'] = metadata
        self.state.performance_history.append(entry)

    def get_improvement_trend(self, window=5):
        if len(self.state.performance_history) < 2:
            return 0.0
        recent = self.state.performance_history[-window:]
        fitnesses = [e.get('fitness', 0) for e in recent
                     if isinstance(e.get('fitness'), (int, float))]
        if len(fitnesses) < 2:
            return 0.0
        n = len(fitnesses)
        x_mean = (n - 1) / 2.0
        y_mean = sum(fitnesses) / n
        num = sum((i - x_mean) * (fitnesses[i] - y_mean)
                  for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den < 1e-12:
            return 0.0
        return num / den

    def get_statistics(self):
        history = [e for e in self.state.performance_history
                   if isinstance(e.get('fitness'), (int, float))]
        if not history:
            return {
                'best': 0.0, 'worst': 0.0, 'average': 0.0,
                'total_generations': 0, 'trend': 0.0,
                'best_generation': None,
            }
        fitnesses = [e['fitness'] for e in history]
        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        return {
            'best': max(fitnesses),
            'worst': min(fitnesses),
            'average': sum(fitnesses) / len(fitnesses),
            'total_generations': len(history),
            'trend': self.get_improvement_trend(),
            'best_generation': history[best_idx].get('generation'),
        }

    def get_best_config(self):
        history = [e for e in self.state.performance_history
                   if isinstance(e.get('fitness'), (int, float))]
        if not history:
            return None
        return max(history, key=lambda e: e['fitness']).get('config')

    def get_rule_outcomes(self, rule_id):
        return self._assess_rule(rule_id)

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def rules(self):
        return self.state.rulebook

    @property
    def memory(self):
        return self

    def get_rules(self):
        return copy.deepcopy(self.state.rulebook)

    def summarize_for_meta_agent(self):
        return self.summarize_memory()

    # ── Serialization ────────────────────────────────────────────────────

    def save(self, path):
        payload = {
            "generation": self.state.generation,
            "rulebook": self.state.rulebook,
            "mutation_log": self.state.mutation_log,
            "performance_history": self.state.performance_history,
            "persistent_memory": self.state.persistent_memory,
            "tensions": [t.__dict__ for t in self.state.tensions],
            "frame_portfolio": self.state.frame_portfolio,
            "recent_rotors": self.state.recent_rotors,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))

    @classmethod
    def load(cls, path, **kwargs):
        data = json.loads(Path(path).read_text())
        state = OrganismState()
        state.generation = data.get("generation", 0)
        state.rulebook = data.get("rulebook",
                                   data.get("rules",
                                            copy.deepcopy(DEFAULT_RULES)))
        state.mutation_log = data.get("mutation_log", [])
        state.performance_history = data.get("performance_history", [])
        state.persistent_memory = data.get("persistent_memory", {})
        state.frame_portfolio = data.get("frame_portfolio",
                                          state.frame_portfolio)
        state.tensions = [
            TensionMemory(**t) for t in data.get("tensions", [])]
        state.recent_rotors = data.get("recent_rotors", [])
        return cls(state=state)

    def export_state(self):
        return {
            "generation": self.state.generation,
            "rulebook": self.state.rulebook,
            "mutation_log": self.state.mutation_log,
            "performance_history": self.state.performance_history,
            "persistent_memory": self.state.persistent_memory,
            "tensions": [t.__dict__ for t in self.state.tensions],
            "frame_portfolio": self.state.frame_portfolio,
            "recent_rotors": self.state.recent_rotors,
        }

    def import_state(self, payload):
        self.state.generation = payload.get("generation", 0)
        self.state.rulebook = payload.get("rulebook",
                                           copy.deepcopy(DEFAULT_RULES))
        self.state.mutation_log = payload.get("mutation_log", [])
        self.state.performance_history = payload.get(
            "performance_history", [])
        self.state.persistent_memory = payload.get(
            "persistent_memory", {})
        self.state.frame_portfolio = payload.get(
            "frame_portfolio", self.state.frame_portfolio)
        self.state.tensions = [
            TensionMemory(**t) for t in payload.get("tensions", [])]
        self.state.recent_rotors = payload.get("recent_rotors", [])


# ── Free function shim (used by evolve.py as fallback) ───────────────────

def propose_variant(analysis, current_config):
    return Organism().propose_variant(analysis, current_config)
