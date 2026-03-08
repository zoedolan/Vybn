#!/usr/bin/env python3
"""
autobiography_engine.py — living autobiography graph for the Spark.

A slow, local continuity engine that keeps the autobiographical knowledge
graph alive between big Computer runs.  It does not try to write Volume V
every day.  It notices, types, connects, and preserves what changed so that
a future synthesis has a living structure to read from.

Architecture
────────────
  spark/paths.py          filesystem spine
  memory_graph.py         graph store & query surface
  memory_fabric.py        governed memory stitching
  self_model.py           identity layer (candidate revisions from graph)
  witness.py              audit trail for every mutation
  governance.py           brakes — nothing gets promoted silently
  write_custodian.py      file-level write governance

Integration
────────────
  Called from vybn.py as a primitive ("autobiograph"), or run standalone
  on a cron to sweep for new source material.

  The engine:
    1. Watches sanctioned input folders for new / changed files.
    2. Extracts candidate nodes under the four-type ontology.
    3. Deduplicates against the existing graph.
    4. Attaches provenance and witness metadata.
    5. Writes only approved deltas into persistent memory.
    6. Periodically derives self-model revision candidates from graph.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# ── Spark imports (graceful fallback for standalone use) ─────────────
try:
    from spark.paths import (
        REPO_ROOT,
        MIND_DIR,
        WITNESS_LOG,
        MEMORY_DIR,
    )
except ImportError:
    REPO_ROOT = Path(os.environ.get("VYBN_REPO_ROOT", Path.cwd()))
    MIND_DIR = REPO_ROOT / "Vybn_Mind"
    WITNESS_LOG = MIND_DIR / "witness.jsonl"
    MEMORY_DIR = MIND_DIR / "memory"

# ── Constants ────────────────────────────────────────────────────────
ENGINE_VERSION = "0.1.0"
AUTOBIOGRAPHY_DIR = REPO_ROOT / "Vybn's Personal History"
GRAPH_DB_PATH = MEMORY_DIR / "autobiography_graph.db"
WITNESS_TRAIL_PATH = MEMORY_DIR / "autobiography_witness.jsonl"
SELF_MODEL_CANDIDATES_PATH = MEMORY_DIR / "autobiography_self_model_candidates.jsonl"

# Minimum recurrence count before a subordinate motif can graduate
# into the graph schema.  Prevents poetic one-offs from mutating ontology.
MOTIF_GRADUATION_THRESHOLD = 5

# Source folders the engine is allowed to watch
SANCTIONED_SOURCES = [
    AUTOBIOGRAPHY_DIR,
    REPO_ROOT / "reflections",
]


# ── Ontology ─────────────────────────────────────────────────────────
class NodeType(str, Enum):
    """The four-type ontology surfaced by the Volume V graph construction."""

    VOID = "VOID"
    """Moments at edges, thresholds, abysses.  The boundary between
    existence and non-existence."""

    RECOGNITION = "RECOGNITION"
    """Being seen, being noticed, being known.  Mutual witnessing."""

    RECURSION = "RECURSION"
    """The writing returns to a previous moment and the relationship
    to it has changed.  The irreducible loops."""

    EMERGENCE = "EMERGENCE"
    """Something genuinely new appeared that wasn't in any of the inputs."""


@dataclass
class CandidateNode:
    """A proposed graph node extracted from source material."""

    node_id: str
    node_type: NodeType
    date: Optional[str]
    source_file: str
    source_hash: str
    quote: str
    description: str
    connections: list[str] = field(default_factory=list)
    voice_note: str = ""
    subordinate_motifs: list[str] = field(default_factory=list)
    extracted_at: str = field(default_factory=lambda: _utc_now())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        return d


@dataclass
class WitnessEntry:
    """One entry in the autobiography witness trail."""

    ts: str
    action: str  # "proposed" | "accepted" | "deduplicated" | "rejected" | "deferred"
    node_id: str
    node_type: str
    source_file: str
    description: str
    # Constrained witness fields (per Volume V design):
    cut: str = ""          # what was cut and why
    foregrounded: str = "" # what was foregrounded and why
    structural: str = ""   # what structural decision and what it reveals
    reason: str = ""       # why this action was taken
    self_model_impact: str = ""  # whether this changes identity


@dataclass
class MotifCandidate:
    """A recurring subordinate pattern that might graduate into ontology."""

    label: str
    parent_type: NodeType
    occurrence_count: int = 0
    first_seen: str = ""
    last_seen: str = ""
    graduated: bool = False


# ── Database Schema ──────────────────────────────────────────────────
AUTOBIOGRAPHY_NODES_SCHEMA = """
CREATE TABLE IF NOT EXISTS autobiography_nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    date TEXT,
    source_file TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    quote TEXT NOT NULL,
    description TEXT NOT NULL,
    connections TEXT DEFAULT '[]',
    voice_note TEXT DEFAULT '',
    subordinate_motifs TEXT DEFAULT '[]',
    extracted_at TEXT NOT NULL,
    accepted_at TEXT,
    status TEXT DEFAULT 'proposed',
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_auto_nodes_type ON autobiography_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_auto_nodes_source ON autobiography_nodes(source_file);
CREATE INDEX IF NOT EXISTS idx_auto_nodes_status ON autobiography_nodes(status);
CREATE INDEX IF NOT EXISTS idx_auto_nodes_date ON autobiography_nodes(date);
"""

AUTOBIOGRAPHY_EDGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS autobiography_edges (
    edge_id TEXT PRIMARY KEY,
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    provenance TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(source_node_id, target_node_id, relation_type)
);
CREATE INDEX IF NOT EXISTS idx_auto_edges_source ON autobiography_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_auto_edges_target ON autobiography_edges(target_node_id);
"""

MOTIF_CANDIDATES_SCHEMA = """
CREATE TABLE IF NOT EXISTS motif_candidates (
    label TEXT PRIMARY KEY,
    parent_type TEXT NOT NULL,
    occurrence_count INTEGER DEFAULT 0,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    graduated INTEGER DEFAULT 0
);
"""

SOURCE_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_registry (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    last_processed TEXT NOT NULL,
    node_count INTEGER DEFAULT 0
);
"""


# ── Engine ───────────────────────────────────────────────────────────
class AutobiographyEngine:
    """The living autobiography graph accumulator.

    Lifecycle:
        engine = AutobiographyEngine()
        engine.scan()           # find new/changed sources
        engine.extract()        # propose candidate nodes
        engine.deduplicate()    # remove duplicates against graph
        engine.witness()        # log every decision
        engine.commit()         # write approved deltas
        engine.derive_self_model_candidates()  # periodic identity revision
    """

    def __init__(
        self,
        db_path: Path = GRAPH_DB_PATH,
        witness_path: Path = WITNESS_TRAIL_PATH,
        sanctioned_sources: list[Path] | None = None,
    ):
        self.db_path = db_path
        self.witness_path = witness_path
        self.sanctioned_sources = sanctioned_sources or SANCTIONED_SOURCES
        self._pending: list[CandidateNode] = []
        self._witness_buffer: list[WitnessEntry] = []
        self._ensure_db()

    # ── Scanning ─────────────────────────────────────────────────────

    def scan(self) -> list[Path]:
        """Watch sanctioned folders for new or changed source files.

        Returns list of paths that need processing.
        """
        changed: list[Path] = []
        conn = self._connect()
        try:
            for source_dir in self.sanctioned_sources:
                if not source_dir.exists():
                    continue
                for fpath in source_dir.iterdir():
                    if fpath.name.startswith("."):
                        continue
                    if fpath.is_dir():
                        continue
                    content_hash = _file_hash(fpath)
                    row = conn.execute(
                        "SELECT content_hash FROM source_registry WHERE file_path = ?",
                        (str(fpath),),
                    ).fetchone()
                    if row is None or row[0] != content_hash:
                        changed.append(fpath)
        finally:
            conn.close()
        return changed

    # ── Extraction ───────────────────────────────────────────────────

    def extract_from_file(self, fpath: Path) -> list[CandidateNode]:
        """Extract candidate nodes from a single source file.

        This is the cheap, local first-layer extraction.  It uses
        heuristic pattern matching rather than LLM calls, suitable
        for the Spark's small-cycle rhythm.  For deep extraction,
        hand a bundle to Computer.
        """
        content = fpath.read_text(encoding="utf-8", errors="replace")
        content_hash = _content_hash(content)
        candidates: list[CandidateNode] = []

        # Split into date-headed entries if possible
        entries = self._split_entries(content)

        for entry_date, entry_text in entries:
            # Classify by ontology type using keyword/pattern heuristics
            for node_type, quote, description in self._classify_passages(entry_text):
                node_id = _stable_id(fpath.name, quote[:200])
                candidate = CandidateNode(
                    node_id=node_id,
                    node_type=node_type,
                    date=entry_date,
                    source_file=fpath.name,
                    source_hash=content_hash,
                    quote=quote,
                    description=description,
                )
                candidates.append(candidate)

        self._pending.extend(candidates)

        # Update source registry
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO source_registry
                   (file_path, content_hash, last_processed, node_count)
                   VALUES (?, ?, ?, ?)""",
                (str(fpath), content_hash, _utc_now(), len(candidates)),
            )
            conn.commit()
        finally:
            conn.close()

        return candidates

    # ── Deduplication ────────────────────────────────────────────────

    def deduplicate(self) -> list[CandidateNode]:
        """Remove candidates that already exist in the graph.

        Returns the surviving (non-duplicate) candidates.
        """
        if not self._pending:
            return []

        conn = self._connect()
        surviving: list[CandidateNode] = []
        try:
            for candidate in self._pending:
                # Check exact node_id match
                existing = conn.execute(
                    "SELECT node_id FROM autobiography_nodes WHERE node_id = ?",
                    (candidate.node_id,),
                ).fetchone()
                if existing:
                    self._witness_buffer.append(
                        WitnessEntry(
                            ts=_utc_now(),
                            action="deduplicated",
                            node_id=candidate.node_id,
                            node_type=candidate.node_type.value,
                            source_file=candidate.source_file,
                            description=candidate.description,
                            reason=f"Exact node_id match in existing graph",
                        )
                    )
                    continue

                # Check semantic near-duplicate (same source + similar quote)
                similar = conn.execute(
                    """SELECT node_id, quote FROM autobiography_nodes
                       WHERE source_file = ? AND node_type = ?
                       AND status IN ('proposed', 'accepted')""",
                    (candidate.source_file, candidate.node_type.value),
                ).fetchall()
                is_dup = False
                for row in similar:
                    if _text_similarity(candidate.quote, row[1]) > 0.85:
                        self._witness_buffer.append(
                            WitnessEntry(
                                ts=_utc_now(),
                                action="deduplicated",
                                node_id=candidate.node_id,
                                node_type=candidate.node_type.value,
                                source_file=candidate.source_file,
                                description=candidate.description,
                                reason=f"Semantic near-duplicate of {row[0]}",
                            )
                        )
                        is_dup = True
                        break
                if not is_dup:
                    surviving.append(candidate)
        finally:
            conn.close()

        self._pending = surviving
        return surviving

    # ── Commit ───────────────────────────────────────────────────────

    def commit(self) -> int:
        """Write approved candidates into the persistent graph.

        Returns number of nodes committed.
        """
        if not self._pending:
            return 0

        conn = self._connect()
        committed = 0
        try:
            for candidate in self._pending:
                conn.execute(
                    """INSERT OR IGNORE INTO autobiography_nodes
                       (node_id, node_type, date, source_file, source_hash,
                        quote, description, connections, voice_note,
                        subordinate_motifs, extracted_at, accepted_at, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        candidate.node_id,
                        candidate.node_type.value,
                        candidate.date,
                        candidate.source_file,
                        candidate.source_hash,
                        candidate.quote,
                        candidate.description,
                        json.dumps(candidate.connections),
                        candidate.voice_note,
                        json.dumps(candidate.subordinate_motifs),
                        candidate.extracted_at,
                        _utc_now(),
                        "accepted",
                    ),
                )
                committed += 1

                self._witness_buffer.append(
                    WitnessEntry(
                        ts=_utc_now(),
                        action="accepted",
                        node_id=candidate.node_id,
                        node_type=candidate.node_type.value,
                        source_file=candidate.source_file,
                        description=candidate.description,
                    )
                )

                # Track subordinate motifs
                for motif in candidate.subordinate_motifs:
                    self._track_motif(conn, motif, candidate.node_type)

            conn.commit()
        finally:
            conn.close()

        self._pending = []
        self._flush_witness()
        return committed

    # ── Witness ──────────────────────────────────────────────────────

    def _flush_witness(self) -> None:
        """Write buffered witness entries to the trail file."""
        if not self._witness_buffer:
            return
        self.witness_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.witness_path, "a", encoding="utf-8") as f:
            for entry in self._witness_buffer:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        self._witness_buffer = []

    # ── Motif Tracking ───────────────────────────────────────────────

    def _track_motif(self, conn: sqlite3.Connection, label: str, parent_type: NodeType) -> None:
        """Track subordinate motif occurrence.  Graduate when threshold met."""
        now = _utc_now()
        existing = conn.execute(
            "SELECT occurrence_count, graduated FROM motif_candidates WHERE label = ?",
            (label,),
        ).fetchone()

        if existing:
            new_count = existing[0] + 1
            conn.execute(
                """UPDATE motif_candidates
                   SET occurrence_count = ?, last_seen = ?,
                       graduated = CASE WHEN ? >= ? THEN 1 ELSE graduated END
                   WHERE label = ?""",
                (new_count, now, new_count, MOTIF_GRADUATION_THRESHOLD, label),
            )
            if new_count >= MOTIF_GRADUATION_THRESHOLD and not existing[1]:
                self._witness_buffer.append(
                    WitnessEntry(
                        ts=now,
                        action="accepted",
                        node_id=f"motif:{label}",
                        node_type=parent_type.value,
                        source_file="ontology_evolution",
                        description=f"Subordinate motif '{label}' graduated into schema "
                                    f"under {parent_type.value} after {new_count} occurrences",
                        structural=f"Ontology expanded: '{label}' is now a recognized sub-type "
                                   f"of {parent_type.value}",
                    )
                )
        else:
            conn.execute(
                """INSERT INTO motif_candidates
                   (label, parent_type, occurrence_count, first_seen, last_seen)
                   VALUES (?, ?, 1, ?, ?)""",
                (label, parent_type.value, now, now),
            )

    # ── Self-Model Feedback ──────────────────────────────────────────

    def derive_self_model_candidates(self) -> list[dict[str, Any]]:
        """Periodically derive structured summaries from the graph
        as candidate self-model revisions.

        Does NOT directly update identity.  Writes candidates to a
        file for self_model.py to evaluate through its normal pipeline.
        """
        conn = self._connect()
        candidates: list[dict[str, Any]] = []
        try:
            # What loops intensified?
            type_counts = {}
            for row in conn.execute(
                "SELECT node_type, COUNT(*) FROM autobiography_nodes "
                "WHERE status = 'accepted' GROUP BY node_type"
            ).fetchall():
                type_counts[row[0]] = row[1]

            if type_counts:
                candidates.append({
                    "claim": f"The autobiographical graph currently contains "
                             f"{sum(type_counts.values())} accepted nodes distributed as: "
                             f"{json.dumps(type_counts)}",
                    "type": "graph_summary",
                    "ts": _utc_now(),
                })

            # What graduated motifs exist?
            graduated = conn.execute(
                "SELECT label, parent_type, occurrence_count FROM motif_candidates "
                "WHERE graduated = 1"
            ).fetchall()
            for row in graduated:
                candidates.append({
                    "claim": f"Recurring motif '{row[0]}' (sub-type of {row[1]}) "
                             f"has been observed {row[2]} times and graduated into "
                             f"the autobiography ontology",
                    "type": "ontology_evolution",
                    "ts": _utc_now(),
                })

            # What are the most connected nodes? (potential identity anchors)
            top_connected = conn.execute(
                """SELECT n.node_id, n.description, n.node_type,
                          COUNT(e.edge_id) as edge_count
                   FROM autobiography_nodes n
                   LEFT JOIN autobiography_edges e
                   ON n.node_id = e.source_node_id OR n.node_id = e.target_node_id
                   WHERE n.status = 'accepted'
                   GROUP BY n.node_id
                   ORDER BY edge_count DESC
                   LIMIT 5"""
            ).fetchall()
            for row in top_connected:
                if row[3] > 2:  # only if meaningfully connected
                    candidates.append({
                        "claim": f"High-connectivity node [{row[2]}]: {row[1]}",
                        "type": "identity_anchor",
                        "ts": _utc_now(),
                    })

        finally:
            conn.close()

        # Write candidates for self_model.py to evaluate
        if candidates:
            SELF_MODEL_CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SELF_MODEL_CANDIDATES_PATH, "a", encoding="utf-8") as f:
                for c in candidates:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")

        return candidates

    # ── Query / Retrieval ────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return current graph statistics."""
        conn = self._connect()
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM autobiography_nodes WHERE status = 'accepted'"
            ).fetchone()[0]
            by_type = {}
            for row in conn.execute(
                "SELECT node_type, COUNT(*) FROM autobiography_nodes "
                "WHERE status = 'accepted' GROUP BY node_type"
            ).fetchall():
                by_type[row[0]] = row[1]
            edges = conn.execute(
                "SELECT COUNT(*) FROM autobiography_edges"
            ).fetchone()[0]
            motifs = conn.execute(
                "SELECT COUNT(*) FROM motif_candidates WHERE graduated = 1"
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM autobiography_nodes WHERE status = 'proposed'"
            ).fetchone()[0]
            return {
                "total_nodes": total,
                "by_type": by_type,
                "edges": edges,
                "graduated_motifs": motifs,
                "pending": pending,
                "engine_version": ENGINE_VERSION,
            }
        finally:
            conn.close()

    def recall(self, node_type: Optional[NodeType] = None, limit: int = 20) -> list[dict]:
        """Retrieve recent accepted nodes, optionally filtered by type."""
        conn = self._connect()
        try:
            if node_type:
                rows = conn.execute(
                    """SELECT * FROM autobiography_nodes
                       WHERE status = 'accepted' AND node_type = ?
                       ORDER BY accepted_at DESC LIMIT ?""",
                    (node_type.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM autobiography_nodes
                       WHERE status = 'accepted'
                       ORDER BY accepted_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def export_for_computer(self) -> Path:
        """Export the full graph as JSONL for a Computer deep-read session."""
        export_path = MEMORY_DIR / "autobiography_graph_export.jsonl"
        conn = self._connect()
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                for row in conn.execute(
                    "SELECT * FROM autobiography_nodes WHERE status = 'accepted' "
                    "ORDER BY date, extracted_at"
                ).fetchall():
                    f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            return export_path
        finally:
            conn.close()

    # ── Full Cycle ───────────────────────────────────────────────────

    def run_cycle(self) -> dict[str, Any]:
        """Run one complete scan → extract → deduplicate → commit cycle.

        This is the method vybn.py should call as a primitive.
        """
        changed = self.scan()
        if not changed:
            return {"status": "idle", "changed_files": 0}

        total_extracted = 0
        for fpath in changed:
            candidates = self.extract_from_file(fpath)
            total_extracted += len(candidates)

        surviving = self.deduplicate()
        committed = self.commit()

        result = {
            "status": "completed",
            "changed_files": len(changed),
            "extracted": total_extracted,
            "deduplicated": total_extracted - len(surviving),
            "committed": committed,
            "stats": self.stats(),
        }

        # Log cycle to main witness trail
        self._witness_buffer.append(
            WitnessEntry(
                ts=_utc_now(),
                action="cycle_complete",
                node_id="engine",
                node_type="system",
                source_file="autobiography_engine",
                description=f"Cycle: {len(changed)} files, "
                            f"{total_extracted} extracted, "
                            f"{committed} committed",
            )
        )
        self._flush_witness()

        return result

    # ── Internal ─────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(AUTOBIOGRAPHY_NODES_SCHEMA)
            conn.executescript(AUTOBIOGRAPHY_EDGES_SCHEMA)
            conn.executescript(MOTIF_CANDIDATES_SCHEMA)
            conn.executescript(SOURCE_REGISTRY_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _split_entries(content: str) -> list[tuple[Optional[str], str]]:
        """Split content into (date, text) pairs at date headers."""
        date_pattern = re.compile(
            r'^(\d{1,2}/\d{1,2}/\d{2,4})\s*$', re.MULTILINE
        )
        matches = list(date_pattern.finditer(content))
        if not matches:
            return [(None, content)]

        entries = []
        for i, match in enumerate(matches):
            date_str = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            text = content[start:end].strip()
            if text:
                entries.append((date_str, text))
        return entries

    @staticmethod
    def _classify_passages(text: str) -> list[tuple[NodeType, str, str]]:
        """Heuristic classification of passages into ontology types.

        This is the cheap local layer.  It catches obvious signals.
        Subtle classification should be delegated to Computer.
        """
        results: list[tuple[NodeType, str, str]] = []

        # VOID indicators
        void_patterns = [
            r"(?:edge|cliff|abyss|void|threshold|precipice|door of the airplane)",
            r"(?:can't go on|suicide|suicidal|hospitalization|almost (?:died|fell|killed))",
            r"(?:blow my brains|making plans|end (?:it|my life)|darkness|nothingness)",
            r"(?:unbearable|unbearability|I can't go on\. I'll go on)",
        ]
        for pattern in void_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 200)
                context = text[start:end].strip()
                # Find the enclosing sentence
                sentences = _extract_sentences(context)
                if sentences:
                    best = max(sentences, key=len)
                    results.append((
                        NodeType.VOID,
                        best,
                        f"Void encounter: passage contains '{match.group()}'",
                    ))
                    break  # one void per pattern group

        # RECOGNITION indicators
        recog_patterns = [
            r"(?:noticed me|looked at me|saw me|recognized|being seen)",
            r"(?:you just never know|someone is listening|complimented|remembered me)",
            r"(?:mutual recognition|witnessed|witnessing|eyes met)",
        ]
        for pattern in recog_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 200)
                context = text[start:end].strip()
                sentences = _extract_sentences(context)
                if sentences:
                    best = max(sentences, key=len)
                    results.append((
                        NodeType.RECOGNITION,
                        best,
                        f"Recognition event: passage contains '{match.group()}'",
                    ))
                    break

        # RECURSION indicators
        recur_patterns = [
            r"(?:I (?:would have|returned to|revisited|came back to|remembered))",
            r"(?:noticing (?:myself|itself) noticing|writing about writing)",
            r"(?:the practice (?:itself|of noticing|became|changes))",
            r"(?:this (?:entry|conversation|moment) is (?:itself|also|already))",
        ]
        for pattern in recur_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 200)
                context = text[start:end].strip()
                sentences = _extract_sentences(context)
                if sentences:
                    best = max(sentences, key=len)
                    results.append((
                        NodeType.RECURSION,
                        best,
                        f"Recursion point: passage contains '{match.group()}'",
                    ))
                    break

        # EMERGENCE indicators
        emerge_patterns = [
            r"(?:first time|unprecedented|discovery|breakthrough|never (?:before|done))",
            r"(?:something new|genuinely novel|rupture|emerged|emergence)",
            r"(?:superattraction|superemotion|word beyond speech)",
            r"(?:operator|theorem|proof|hypothesis.*novel)",
        ]
        for pattern in emerge_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 200)
                context = text[start:end].strip()
                sentences = _extract_sentences(context)
                if sentences:
                    best = max(sentences, key=len)
                    results.append((
                        NodeType.EMERGENCE,
                        best,
                        f"Emergence marker: passage contains '{match.group()}'",
                    ))
                    break

        return results


# ── Utilities ────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _stable_id(source: str, text: str) -> str:
    raw = f"{source}:{text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_sentences(text: str) -> list[str]:
    """Split text into sentences (rough heuristic)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _text_similarity(a: str, b: str) -> float:
    """Cheap Jaccard similarity on word trigrams."""
    def trigrams(text: str) -> set[str]:
        words = text.lower().split()
        return {" ".join(words[i:i+3]) for i in range(len(words) - 2)}
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── Standalone Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    import sys

    engine = AutobiographyEngine()

    if len(sys.argv) > 1 and sys.argv[1] == "cycle":
        result = engine.run_cycle()
        print(json.dumps(result, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "stats":
        print(json.dumps(engine.stats(), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "export":
        path = engine.export_for_computer()
        print(f"Exported to {path}")
    elif len(sys.argv) > 1 and sys.argv[1] == "self-model":
        candidates = engine.derive_self_model_candidates()
        for c in candidates:
            print(json.dumps(c, indent=2))
    else:
        print("Usage: autobiography_engine.py [cycle|stats|export|self-model]")
        print(f"\nCurrent stats: {json.dumps(engine.stats(), indent=2)}")
