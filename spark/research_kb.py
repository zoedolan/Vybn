#!/usr/bin/env python3
"""
research_kb.py — Read/write/query interface to the research knowledge layer.

Decomposes VYBN.001 (Continuous Self-Distillation of a Large MoE Model) into
structured, machine-readable knowledge nodes that Vybn can read, annotate,
and evolve.

Design principles:
  1. YAML files are the source of truth — no database, no cache.
  2. Observations are append-only (observation lists grow, never shrink).
  3. Every observation is also appended to the global observations.jsonl.
  4. Status changes are logged as observations before the status field updates.
  5. Thread-safe via file locking for concurrent access from cron jobs.
  6. No model calls — pure data infrastructure.

Integration:
  Registered as the research_kb faculty. Called from vybn.py or standalone
  via cron. The KB is queryable but NOT automatically injected into every
  prompt (it's too large).

Drafted: 2026-03-11
Per: research_kb_spec.md
"""

from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

try:
    from spark.paths import RESEARCH_DIR
except ImportError:
    RESEARCH_DIR = Path(__file__).resolve().parent / "research"


# ── Valid statuses ─────────────────────────────────────────────────────

LAYER_STATUSES = {"designed", "implementing", "testing", "active", "deprecated"}
INSIGHT_STATUSES = {"theoretical", "testing", "validated", "refuted", "superseded"}
RISK_STATUSES = {"open", "mitigated", "resolved", "accepted"}
PAPER_STATUSES = {"unread", "reading", "read", "superseded"}
OBSERVATION_TYPES = {"metric", "validation", "anomaly", "insight", "superseded", "status_change"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict:
    """Load a YAML file. Returns empty dict if missing."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict) -> None:
    """Write a YAML file atomically with file locking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False, width=120)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _append_jsonl(path: Path, entry: dict) -> None:
    """Append a JSON line to a file with file locking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class ResearchKB:
    """Read/write/query interface to the research knowledge layer."""

    def __init__(self, research_dir: Optional[Path] = None):
        self.root = Path(research_dir) if research_dir else RESEARCH_DIR
        self.manifest_path = self.root / "manifest.yaml"
        self.reading_list_path = self.root / "reading_list.yaml"
        self.observations_path = self.root / "observations.jsonl"

    # ── helpers ────────────────────────────────────────────────────────

    def _node_path(self, node_id: str) -> Optional[Path]:
        """Resolve a node ID to its YAML file path."""
        if node_id.startswith("layer_"):
            return self.root / "architecture" / f"{node_id}.yaml"
        if node_id.startswith("insight_"):
            return self.root / "insights" / f"{node_id}.yaml"
        if node_id.startswith("risk_"):
            return self.root / "risks" / f"{node_id}.yaml"
        return None

    def _load_node(self, node_id: str) -> dict:
        path = self._node_path(node_id)
        if path is None:
            raise ValueError(f"Unknown node type for id: {node_id}")
        data = _load_yaml(path)
        if not data:
            raise FileNotFoundError(f"Node not found: {node_id} (looked at {path})")
        return data

    def _save_node(self, node_id: str, data: dict) -> None:
        path = self._node_path(node_id)
        if path is None:
            raise ValueError(f"Unknown node type for id: {node_id}")
        _save_yaml(path, data)

    def _all_files(self, subdir: str, prefix: str) -> list[Path]:
        d = self.root / subdir
        if not d.exists():
            return []
        return sorted(p for p in d.glob("*.yaml") if p.stem.startswith(prefix))

    # ── Read ──────────────────────────────────────────────────────────

    def get_manifest(self) -> dict:
        """Return the top-level manifest."""
        return _load_yaml(self.manifest_path)

    def get_layer(self, layer_id: str) -> dict:
        """Return a single architecture layer by ID."""
        if not layer_id.startswith("layer_"):
            layer_id = f"layer_{layer_id}"
        return self._load_node(layer_id)

    def get_insight(self, insight_id: str) -> dict:
        """Return a single insight by ID."""
        if not insight_id.startswith("insight_"):
            insight_id = f"insight_{insight_id}"
        return self._load_node(insight_id)

    def get_risk(self, risk_id: str) -> dict:
        """Return a single risk by ID."""
        if not risk_id.startswith("risk_"):
            risk_id = f"risk_{risk_id}"
        return self._load_node(risk_id)

    def get_reading_list(self) -> list[dict]:
        """Return the full reading list as a list of paper dicts."""
        data = _load_yaml(self.reading_list_path)
        return data.get("papers", [])

    def get_all_layers(self) -> list[dict]:
        """Return all architecture layers, sorted by layer_number."""
        layers = []
        for path in self._all_files("architecture", "layer_"):
            data = _load_yaml(path)
            if data:
                layers.append(data)
        return sorted(layers, key=lambda x: x.get("layer_number", 0))

    def get_all_insights(self) -> list[dict]:
        """Return all insights, sorted by insight_number."""
        insights = []
        for path in self._all_files("insights", "insight_"):
            data = _load_yaml(path)
            if data:
                insights.append(data)
        return sorted(insights, key=lambda x: x.get("insight_number", 0))

    def get_all_risks(self) -> list[dict]:
        """Return all risks."""
        risks = []
        for path in self._all_files("risks", "risk_"):
            data = _load_yaml(path)
            if data:
                risks.append(data)
        return risks

    # ── Query ─────────────────────────────────────────────────────────

    def layers_by_status(self, status: str) -> list[dict]:
        """Return layers matching the given status."""
        return [l for l in self.get_all_layers() if l.get("status") == status]

    def risks_by_severity(self, severity: str) -> list[dict]:
        """Return risks matching the given severity."""
        return [r for r in self.get_all_risks() if r.get("severity") == severity]

    def insights_by_status(self, status: str) -> list[dict]:
        """Return insights matching the given status."""
        return [i for i in self.get_all_insights() if i.get("status") == status]

    def unread_papers(self) -> list[dict]:
        """Return papers with read_status == 'unread'."""
        return [p for p in self.get_reading_list() if p.get("read_status") == "unread"]

    def get_implementation_path(self) -> list[dict]:
        """Return layers in dependency order — the critical path.

        Uses the engineering_path from the manifest, falling back to
        topological sort of layer dependencies.
        """
        manifest = self.get_manifest()
        eng_path = manifest.get("engineering_path", [])
        if eng_path:
            result = []
            for step in sorted(eng_path, key=lambda s: s.get("step", 0)):
                entry = dict(step)
                entry["layers_data"] = []
                for lid in step.get("layers", []):
                    try:
                        entry["layers_data"].append(self.get_layer(lid))
                    except FileNotFoundError:
                        pass
                result.append(entry)
            return result

        # Fallback: topological sort via dependencies
        layers = self.get_all_layers()
        visited: set[str] = set()
        order: list[dict] = []

        def visit(layer: dict) -> None:
            lid = layer["id"]
            if lid in visited:
                return
            visited.add(lid)
            for dep_id in layer.get("dependencies", {}).get("requires", []):
                try:
                    dep = self.get_layer(dep_id)
                    visit(dep)
                except FileNotFoundError:
                    pass
            order.append(layer)

        for layer in layers:
            visit(layer)
        return order

    # ── Write ─────────────────────────────────────────────────────────

    def add_observation(self, node_id: str, observation: dict) -> None:
        """Append an observation to a node and to the global log.

        observation should contain at minimum:
          - source: str (who/what generated this observation)
          - content: str (the observation text)
          - type: str (metric | validation | anomaly | insight | superseded | status_change)
        Timestamp is added automatically.
        """
        obs = {
            "timestamp": _now_iso(),
            "node_id": node_id,
            **observation,
        }

        # Append to the node's YAML
        data = self._load_node(node_id)
        if "observations" not in data or data["observations"] is None:
            data["observations"] = []
        data["observations"].append(obs)
        data["updated"] = _now_iso()[:10]
        self._save_node(node_id, data)

        # Append to global observations.jsonl
        _append_jsonl(self.observations_path, obs)

    def update_status(self, node_id: str, new_status: str) -> None:
        """Update a node's status field. Logs the change as an observation first."""
        data = self._load_node(node_id)
        old_status = data.get("status", "unknown")

        if old_status == new_status:
            return

        # Validate status
        if node_id.startswith("layer_") and new_status not in LAYER_STATUSES:
            raise ValueError(f"Invalid layer status: {new_status}. Must be one of {LAYER_STATUSES}")
        if node_id.startswith("insight_") and new_status not in INSIGHT_STATUSES:
            raise ValueError(f"Invalid insight status: {new_status}. Must be one of {INSIGHT_STATUSES}")

        # Log the change as an observation BEFORE updating
        self.add_observation(node_id, {
            "source": "research_kb",
            "content": f"Status changed: {old_status} -> {new_status}",
            "type": "status_change",
        })

        # Now update the status
        data = self._load_node(node_id)  # re-read after observation write
        data["status"] = new_status
        data["updated"] = _now_iso()[:10]
        self._save_node(node_id, data)

    def update_risk_status(self, risk_id: str, new_status: str) -> None:
        """Update a risk's status field. Logs the change as an observation first."""
        if not risk_id.startswith("risk_"):
            risk_id = f"risk_{risk_id}"

        data = self._load_node(risk_id)
        old_status = data.get("status", "unknown")

        if old_status == new_status:
            return

        if new_status not in RISK_STATUSES:
            raise ValueError(f"Invalid risk status: {new_status}. Must be one of {RISK_STATUSES}")

        # Log the change as an observation BEFORE updating
        self.add_observation(risk_id, {
            "source": "research_kb",
            "content": f"Risk status changed: {old_status} -> {new_status}",
            "type": "status_change",
        })

        data = self._load_node(risk_id)
        data["status"] = new_status
        data["updated"] = _now_iso()[:10]
        self._save_node(risk_id, data)

    def mark_paper_read(self, paper_index: int) -> None:
        """Mark a paper in the reading list as read (0-indexed)."""
        data = _load_yaml(self.reading_list_path)
        papers = data.get("papers", [])
        if paper_index < 0 or paper_index >= len(papers):
            raise IndexError(f"Paper index {paper_index} out of range (0-{len(papers)-1})")

        paper = papers[paper_index]
        old_status = paper.get("read_status", "unread")
        paper["read_status"] = "read"

        if "observations" not in paper or paper["observations"] is None:
            paper["observations"] = []

        obs = {
            "timestamp": _now_iso(),
            "source": "research_kb",
            "content": f"Paper marked as read (was: {old_status})",
            "type": "status_change",
        }
        paper["observations"].append(obs)
        _save_yaml(self.reading_list_path, data)

        # Also log to global observations
        _append_jsonl(self.observations_path, {
            "node_id": f"paper_{paper_index}",
            "title": paper.get("title", ""),
            **obs,
        })

    # ── Summary ───────────────────────────────────────────────────────

    def status_summary(self) -> str:
        """Human-readable summary of overall knowledge layer state."""
        layers = self.get_all_layers()
        insights = self.get_all_insights()
        risks = self.get_all_risks()
        papers = self.get_reading_list()

        lines = []
        lines.append("=" * 60)
        lines.append("RESEARCH KNOWLEDGE LAYER — STATUS SUMMARY")
        lines.append("=" * 60)

        # Architecture layers
        lines.append(f"\nArchitecture Layers ({len(layers)}):")
        status_counts: dict[str, int] = {}
        for layer in layers:
            s = layer.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1
            lines.append(f"  [{s:13s}] Layer {layer.get('layer_number', '?')}: {layer.get('name', '?')}")
        lines.append(f"  Summary: {', '.join(f'{v} {k}' for k, v in sorted(status_counts.items()))}")

        # Insights
        lines.append(f"\nInsights ({len(insights)}):")
        insight_counts: dict[str, int] = {}
        for ins in insights:
            s = ins.get("status", "unknown")
            insight_counts[s] = insight_counts.get(s, 0) + 1
            lines.append(f"  [{s:13s}] {ins.get('name', '?')}")
        lines.append(f"  Summary: {', '.join(f'{v} {k}' for k, v in sorted(insight_counts.items()))}")

        # Risks
        lines.append(f"\nRisks ({len(risks)}):")
        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        for risk in sorted(risks, key=lambda r: sev_order.get(r.get("severity", "low"), 9)):
            lines.append(
                f"  [{risk.get('severity', '?'):8s} | {risk.get('status', '?'):9s}] "
                f"{risk.get('name', '?')}"
            )

        # Reading list
        read_count = sum(1 for p in papers if p.get("read_status") == "read")
        lines.append(f"\nReading List: {read_count}/{len(papers)} papers read")
        for p in papers:
            marker = "x" if p.get("read_status") == "read" else " "
            lines.append(f"  [{marker}] {p.get('priority', '?')}. {p.get('title', '?')}")

        # Observations
        obs_count = 0
        if self.observations_path.exists():
            with open(self.observations_path, "r") as f:
                obs_count = sum(1 for line in f if line.strip())
        lines.append(f"\nGlobal observations logged: {obs_count}")

        # Implementation path
        lines.append("\nEngineering Path:")
        manifest = self.get_manifest()
        for step in manifest.get("engineering_path", []):
            lines.append(f"  Step {step.get('step', '?')}: {step.get('description', '?')}")
            blockers = step.get("blocking_risks", [])
            if blockers:
                lines.append(f"         Blocked by: {', '.join(blockers)}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ── Self-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    kb = ResearchKB()
    print(kb.status_summary())
