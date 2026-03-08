from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, deque
from typing import Any

try:
    from .memory_fabric import MemoryFabric
    from .memory_types import MemoryEntry, MemoryPlane
except ImportError:  # pragma: no cover
    from memory_fabric import MemoryFabric
    from memory_types import MemoryEntry, MemoryPlane


EXTRACTOR_VERSION = "memory-graph-v1"
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'_-]{2,}")
STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "almost",
    "along",
    "already",
    "also",
    "always",
    "among",
    "another",
    "around",
    "because",
    "before",
    "being",
    "between",
    "briefly",
    "breathe",
    "cannot",
    "certain",
    "could",
    "did",
    "does",
    "doing",
    "each",
    "else",
    "enough",
    "even",
    "every",
    "from",
    "have",
    "into",
    "just",
    "last",
    "like",
    "made",
    "make",
    "many",
    "might",
    "more",
    "most",
    "much",
    "must",
    "need",
    "never",
    "notice",
    "only",
    "other",
    "ours",
    "over",
    "really",
    "same",
    "say",
    "says",
    "should",
    "since",
    "some",
    "still",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "until",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
    "youre",
}

GRAPH_NODES_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    node_type TEXT NOT NULL,
    description TEXT DEFAULT '',
    activation_count INTEGER DEFAULT 0,
    salience REAL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_label ON graph_nodes(label);
"""

GRAPH_EDGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id TEXT PRIMARY KEY,
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    provenance_entry_id TEXT,
    provenance_artifact TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    UNIQUE(source_node_id, target_node_id, relation_type, provenance_entry_id)
);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_relation ON graph_edges(relation_type);
"""

GRAPH_EXTRACTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_extractions (
    source_entry_id TEXT PRIMARY KEY,
    plane TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_artifact TEXT,
    extracted_at TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_graph_extractions_artifact ON graph_extractions(source_artifact);
"""

GRAPH_DIFFS_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_diffs (
    diff_id TEXT PRIMARY KEY,
    source_entry_id TEXT NOT NULL,
    plane TEXT NOT NULL,
    diffed_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_graph_diffs_entry ON graph_diffs(source_entry_id);
CREATE INDEX IF NOT EXISTS idx_graph_diffs_plane ON graph_diffs(plane);
"""


class MemoryGraph:
    """Associative graph layer that sits above the governed memory fabric."""

    def __init__(self, fabric: MemoryFabric, extractor_version: str = EXTRACTOR_VERSION):
        self.fabric = fabric
        self.extractor_version = extractor_version
        self._ensure_tables()

    def ingest_entry(
        self,
        entry: MemoryEntry,
        *,
        claim_entries: list[Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        if entry.plane is MemoryPlane.COMMONS:
            return {"status": "skipped", "reason": "commons entry ingestion is not implemented", "entry_id": entry.entry_id}

        if not force and self._is_fresh(entry):
            return {"status": "skipped", "reason": "entry already indexed", "entry_id": entry.entry_id}

        self.fabric._authorize_memory_write(
            plane=entry.plane,
            faculty_id="memory_fabric",
            purpose_binding=list(entry.purpose_binding) or ["private_memory"],
            consent_scope_id=entry.consent_scope_id,
            source_artifact=entry.source_artifact or f"entry:{entry.entry_id}",
        )

        conn = self.fabric._connection_for(entry.plane)
        ts = self.fabric._utc_now()
        entry_node_id = self._entry_node_id(entry.entry_id)
        encounter_type = self._encounter_type(entry)
        encounter_parties = self._encounter_parties(entry)
        structural_change = self._structural_change(entry)
        diff_tracker = self._new_diff_tracker(entry, ts)

        self._upsert_node(
            conn,
            node_id=entry_node_id,
            label=self._entry_label(entry),
            node_type="memory_entry",
            description=entry.content[:280],
            salience=1.0,
            now=ts,
            metadata={
                "entry_id": entry.entry_id,
                "plane": entry.plane.value,
                "source_artifact": entry.source_artifact,
                "source_signal_id": entry.source_signal_id,
                "created_at": entry.created_at,
                "sensitivity": entry.sensitivity,
                "mood": entry.metadata.get("mood"),
                "content_preview": entry.content[:280],
                "encounter_type": encounter_type,
                "encounter_parties": encounter_parties,
                "structural_change": structural_change,
            },
            diff_tracker=diff_tracker,
        )

        if entry.source_artifact:
            artifact_node_id = f"artifact:{self._slug(entry.source_artifact)}"
            self._upsert_node(
                conn,
                node_id=artifact_node_id,
                label=entry.source_artifact,
                node_type="artifact",
                description=f"Source artifact for {entry.entry_id}",
                salience=0.75,
                now=ts,
                metadata={
                    "source_artifact": entry.source_artifact,
                    "encounter_type": encounter_type,
                    "encounter_parties": encounter_parties,
                },
                diff_tracker=diff_tracker,
            )
            self._upsert_edge(
                conn,
                source_node_id=entry_node_id,
                target_node_id=artifact_node_id,
                relation_type="ORIGINATED_IN",
                weight=0.9,
                provenance_entry_id=entry.entry_id,
                provenance_artifact=entry.source_artifact,
                now=ts,
                metadata={
                    "plane": entry.plane.value,
                    "encounter_type": encounter_type,
                    "encounter_parties": encounter_parties,
                    "structural_change": structural_change,
                },
                diff_tracker=diff_tracker,
            )

        if entry.source_signal_id:
            signal_node_id = f"signal:{self._slug(entry.source_signal_id)}"
            self._upsert_node(
                conn,
                node_id=signal_node_id,
                label=entry.source_signal_id,
                node_type="signal",
                description="Source signal that contributed to this memory.",
                salience=0.7,
                now=ts,
                metadata={
                    "source_signal_id": entry.source_signal_id,
                    "encounter_type": encounter_type,
                },
                diff_tracker=diff_tracker,
            )
            self._upsert_edge(
                conn,
                source_node_id=entry_node_id,
                target_node_id=signal_node_id,
                relation_type="TRACED_TO_SIGNAL",
                weight=0.8,
                provenance_entry_id=entry.entry_id,
                provenance_artifact=entry.source_artifact,
                now=ts,
                metadata={
                    "plane": entry.plane.value,
                    "encounter_type": encounter_type,
                    "encounter_parties": encounter_parties,
                    "structural_change": structural_change,
                },
                diff_tracker=diff_tracker,
            )

        encounter_type_node_id = f"encounter_type:{self._slug(encounter_type)}"
        self._upsert_node(
            conn,
            node_id=encounter_type_node_id,
            label=encounter_type,
            node_type="encounter_type",
            description="Encounter type attached to a memory entry.",
            salience=0.74,
            now=ts,
            metadata={"encounter_type": encounter_type},
            diff_tracker=diff_tracker,
        )
        self._upsert_edge(
            conn,
            source_node_id=entry_node_id,
            target_node_id=encounter_type_node_id,
            relation_type="HAS_ENCOUNTER_TYPE",
            weight=0.88,
            provenance_entry_id=entry.entry_id,
            provenance_artifact=entry.source_artifact,
            now=ts,
            metadata={
                "plane": entry.plane.value,
                "encounter_type": encounter_type,
                "encounter_parties": encounter_parties,
                "structural_change": structural_change,
            },
            diff_tracker=diff_tracker,
        )

        structural_node_id = f"structural_change:{'true' if structural_change else 'false'}"
        self._upsert_node(
            conn,
            node_id=structural_node_id,
            label="structural_change" if structural_change else "incremental_change",
            node_type="structural_change",
            description="Whether the encounter reorganized existing graph structure.",
            salience=0.72,
            now=ts,
            metadata={"structural_change": structural_change},
            diff_tracker=diff_tracker,
        )
        self._upsert_edge(
            conn,
            source_node_id=entry_node_id,
            target_node_id=structural_node_id,
            relation_type="MARKED_AS_STRUCTURAL_CHANGE",
            weight=0.86,
            provenance_entry_id=entry.entry_id,
            provenance_artifact=entry.source_artifact,
            now=ts,
            metadata={
                "plane": entry.plane.value,
                "encounter_type": encounter_type,
                "encounter_parties": encounter_parties,
                "structural_change": structural_change,
            },
            diff_tracker=diff_tracker,
        )

        mood = str(entry.metadata.get("mood") or "").strip()
        if mood:
            mood_node_id = f"mood:{self._slug(mood)}"
            self._upsert_node(
                conn,
                node_id=mood_node_id,
                label=mood,
                node_type="mood",
                description="Affective coloring attached to a memory entry.",
                salience=0.65,
                now=ts,
                metadata={"mood": mood, "encounter_type": encounter_type},
                diff_tracker=diff_tracker,
            )
            self._upsert_edge(
                conn,
                source_node_id=entry_node_id,
                target_node_id=mood_node_id,
                relation_type="EXPRESSES_MOOD",
                weight=0.75,
                provenance_entry_id=entry.entry_id,
                provenance_artifact=entry.source_artifact,
                now=ts,
                metadata={
                    "plane": entry.plane.value,
                    "encounter_type": encounter_type,
                    "encounter_parties": encounter_parties,
                },
                diff_tracker=diff_tracker,
            )

        parties = [str(p).strip() for p in entry.metadata.get("parties", []) if str(p).strip()]
        for party in parties[:6]:
            party_node_id = f"party:{self._slug(party)}"
            self._upsert_node(
                conn,
                node_id=party_node_id,
                label=party,
                node_type="party",
                description="A relational participant carried in the memory plane.",
                salience=0.85,
                now=ts,
                metadata={"party": party, "encounter_type": encounter_type},
                diff_tracker=diff_tracker,
            )
            self._upsert_edge(
                conn,
                source_node_id=entry_node_id,
                target_node_id=party_node_id,
                relation_type="INVOLVES_PARTY",
                weight=0.95,
                provenance_entry_id=entry.entry_id,
                provenance_artifact=entry.source_artifact,
                now=ts,
                metadata={
                    "plane": entry.plane.value,
                    "encounter_type": encounter_type,
                    "encounter_parties": encounter_parties,
                    "structural_change": structural_change,
                },
                diff_tracker=diff_tracker,
            )

        for encounter_party in encounter_parties[:8]:
            party_node_id = f"encounter_party:{self._slug(encounter_party)}"
            self._upsert_node(
                conn,
                node_id=party_node_id,
                label=encounter_party,
                node_type="encounter_party",
                description="Named participant or stimulus in an encounter.",
                salience=0.82,
                now=ts,
                metadata={"encounter_party": encounter_party},
                diff_tracker=diff_tracker,
            )
            self._upsert_edge(
                conn,
                source_node_id=entry_node_id,
                target_node_id=party_node_id,
                relation_type="HAS_ENCOUNTER_PARTY",
                weight=0.9,
                provenance_entry_id=entry.entry_id,
                provenance_artifact=entry.source_artifact,
                now=ts,
                metadata={
                    "plane": entry.plane.value,
                    "encounter_type": encounter_type,
                    "encounter_parties": encounter_parties,
                    "structural_change": structural_change,
                },
                diff_tracker=diff_tracker,
            )

        concepts = self._extract_concepts(entry.content)
        for concept in concepts:
            concept_node_id = f"concept:{concept['slug']}"
            self._upsert_node(
                conn,
                node_id=concept_node_id,
                label=concept["label"],
                node_type="concept",
                description=f"Concept extracted from {entry.entry_id}",
                salience=concept["score"],
                now=ts,
                metadata={
                    "normalized": concept["normalized"],
                    "plane": entry.plane.value,
                    "encounter_type": encounter_type,
                },
                diff_tracker=diff_tracker,
            )
            self._upsert_edge(
                conn,
                source_node_id=entry_node_id,
                target_node_id=concept_node_id,
                relation_type="MENTIONS",
                weight=concept["score"],
                provenance_entry_id=entry.entry_id,
                provenance_artifact=entry.source_artifact,
                now=ts,
                metadata={
                    "token_count": concept["count"],
                    "plane": entry.plane.value,
                    "encounter_type": encounter_type,
                    "encounter_parties": encounter_parties,
                    "structural_change": structural_change,
                },
                diff_tracker=diff_tracker,
            )

        for index, source in enumerate(concepts):
            for target in concepts[index + 1 :]:
                ordered = sorted([source["slug"], target["slug"]])
                self._upsert_edge(
                    conn,
                    source_node_id=f"concept:{ordered[0]}",
                    target_node_id=f"concept:{ordered[1]}",
                    relation_type="CO_OCCURS",
                    weight=round((source["score"] + target["score"]) / 2, 3),
                    provenance_entry_id=entry.entry_id,
                    provenance_artifact=entry.source_artifact,
                    now=ts,
                    metadata={
                        "plane": entry.plane.value,
                        "encounter_type": encounter_type,
                        "encounter_parties": encounter_parties,
                        "structural_change": structural_change,
                    },
                    diff_tracker=diff_tracker,
                )

        normalized_claims = [self._normalize_claim_entry(item) for item in (claim_entries or [])]
        for claim in normalized_claims:
            claim_text = str(claim.get("claim_text") or claim.get("text") or "").strip()
            if not claim_text:
                continue

            claim_type = str(claim.get("claim_type") or "claim")
            provenance_class = str(claim.get("provenance_class") or "unknown")
            verification_status = str(claim.get("verification_status") or "unknown")
            claim_node_id = self._claim_node_id(entry.entry_id, claim_type, claim_text)

            self._upsert_node(
                conn,
                node_id=claim_node_id,
                label=claim_text[:120],
                node_type="claim",
                description=claim_text,
                salience=0.9,
                now=ts,
                metadata={
                    "claim_type": claim_type,
                    "provenance_class": provenance_class,
                    "verification_status": verification_status,
                    "source_artifact": claim.get("source_artifact") or entry.source_artifact,
                    "perturbation_required": bool(claim.get("perturbation_required", False)),
                    "encounter_type": encounter_type,
                    "encounter_parties": encounter_parties,
                    "structural_change": structural_change,
                },
                diff_tracker=diff_tracker,
            )
            self._upsert_edge(
                conn,
                source_node_id=entry_node_id,
                target_node_id=claim_node_id,
                relation_type="ASSERTS",
                weight=0.95,
                provenance_entry_id=entry.entry_id,
                provenance_artifact=entry.source_artifact,
                now=ts,
                metadata={
                    "plane": entry.plane.value,
                    "encounter_type": encounter_type,
                    "encounter_parties": encounter_parties,
                    "structural_change": structural_change,
                },
                diff_tracker=diff_tracker,
            )

            claim_type_node_id = f"claim_type:{self._slug(claim_type)}"
            self._upsert_node(
                conn,
                node_id=claim_type_node_id,
                label=claim_type,
                node_type="claim_type",
                description="Structured self-model claim category.",
                salience=0.7,
                now=ts,
                metadata={"claim_type": claim_type},
                diff_tracker=diff_tracker,
            )
            self._upsert_edge(
                conn,
                source_node_id=claim_node_id,
                target_node_id=claim_type_node_id,
                relation_type="HAS_CLAIM_TYPE",
                weight=0.8,
                provenance_entry_id=entry.entry_id,
                provenance_artifact=entry.source_artifact,
                now=ts,
                metadata={"plane": entry.plane.value, "encounter_type": encounter_type},
                diff_tracker=diff_tracker,
            )

            provenance_node_id = f"provenance:{self._slug(provenance_class)}"
            self._upsert_node(
                conn,
                node_id=provenance_node_id,
                label=provenance_class,
                node_type="provenance",
                description="How a self-claim is grounded.",
                salience=0.68,
                now=ts,
                metadata={"provenance_class": provenance_class},
                diff_tracker=diff_tracker,
            )
            self._upsert_edge(
                conn,
                source_node_id=claim_node_id,
                target_node_id=provenance_node_id,
                relation_type="HAS_PROVENANCE",
                weight=0.78,
                provenance_entry_id=entry.entry_id,
                provenance_artifact=entry.source_artifact,
                now=ts,
                metadata={"plane": entry.plane.value, "encounter_type": encounter_type},
                diff_tracker=diff_tracker,
            )

            status_node_id = f"verification:{self._slug(verification_status)}"
            self._upsert_node(
                conn,
                node_id=status_node_id,
                label=verification_status,
                node_type="verification_status",
                description="Verification state attached to a self-claim.",
                salience=0.66,
                now=ts,
                metadata={"verification_status": verification_status},
                diff_tracker=diff_tracker,
            )
            self._upsert_edge(
                conn,
                source_node_id=claim_node_id,
                target_node_id=status_node_id,
                relation_type="HAS_VERIFICATION",
                weight=0.76,
                provenance_entry_id=entry.entry_id,
                provenance_artifact=entry.source_artifact,
                now=ts,
                metadata={"plane": entry.plane.value, "encounter_type": encounter_type},
                diff_tracker=diff_tracker,
            )

        self._record_extraction(conn, entry, ts)
        self._record_diff(conn, diff_tracker)
        conn.commit()
        diff_summary = self._finalize_diff_tracker(diff_tracker)
        return {
            "status": "ingested",
            "entry_id": entry.entry_id,
            "plane": entry.plane.value,
            "encounter_type": encounter_type,
            "encounter_parties": encounter_parties,
            "structural_change": structural_change,
            "concept_count": len(concepts),
            "claim_count": len(normalized_claims),
            "nodes_touched": len(diff_summary["new_nodes"]),
            "edges_touched": len(diff_summary["new_edges"]),
            "diff": diff_summary,
        }

    def associative_recall(
        self,
        plane: MemoryPlane,
        query_text: str,
        *,
        depth: int = 2,
        limit: int = 12,
    ) -> dict[str, Any]:
        conn = self.fabric._connection_for(plane)
        seeds = self._rank_seed_nodes(conn, query_text, limit=max(limit, 6))
        seed_ids = [seed["node_id"] for seed in seeds[: min(4, len(seeds))]]
        if not seed_ids:
            return {
                "plane": plane.value,
                "query": query_text,
                "seeds": [],
                "nodes": [],
                "edges": [],
                "stats": self.stats().get(plane.value, {}),
            }

        nodes, edges = self._expand_neighborhood(conn, seed_ids, depth=depth, max_nodes=max(limit * 3, 12))
        return {
            "plane": plane.value,
            "query": query_text,
            "seeds": seeds[: min(5, len(seeds))],
            "nodes": nodes[: max(limit * 2, 10)],
            "edges": edges[: max(limit * 3, 14)],
            "stats": self.stats().get(plane.value, {}),
        }

    def prompt_context(
        self,
        plane: MemoryPlane,
        query_text: str,
        *,
        depth: int = 2,
        limit: int = 8,
        max_chars: int = 420,
    ) -> str:
        recall = self.associative_recall(plane, query_text, depth=depth, limit=limit)
        if not recall["seeds"]:
            return ""

        nodes_by_id = {node["node_id"]: node for node in recall["nodes"]}
        lines = []
        seed_labels = [seed["label"] for seed in recall["seeds"][:3]]
        if seed_labels:
            lines.append("seeds: " + " | ".join(seed_labels))

        for edge in recall["edges"][:6]:
            source = nodes_by_id.get(edge["source_node_id"], {}).get("label", edge["source_node_id"])
            target = nodes_by_id.get(edge["target_node_id"], {}).get("label", edge["target_node_id"])
            lines.append(f"{source} -{edge['relation_type'].lower()}-> {target}")

        entry_snippets = []
        for node in recall["nodes"]:
            if node.get("node_type") == "memory_entry":
                preview = node.get("metadata", {}).get("content_preview") or node.get("description", "")
                if preview:
                    entry_snippets.append(preview[:120])
            if len(entry_snippets) >= 2:
                break
        if entry_snippets:
            lines.append("echoes: " + " | ".join(entry_snippets))

        text = " ; ".join(lines)
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "..."
        return text

    def stats(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for plane in (MemoryPlane.PRIVATE, MemoryPlane.RELATIONAL, MemoryPlane.COMMONS):
            conn = self.fabric._connection_for(plane)
            snapshot[plane.value] = {
                "nodes": conn.execute("SELECT COUNT(*) AS count FROM graph_nodes").fetchone()["count"],
                "edges": conn.execute("SELECT COUNT(*) AS count FROM graph_edges").fetchone()["count"],
                "indexed_entries": conn.execute("SELECT COUNT(*) AS count FROM graph_extractions").fetchone()["count"],
            }
        return snapshot

    def _ensure_tables(self) -> None:
        for plane in (MemoryPlane.PRIVATE, MemoryPlane.RELATIONAL, MemoryPlane.COMMONS):
            conn = self.fabric._connection_for(plane)
            conn.executescript(GRAPH_NODES_SCHEMA)
            conn.executescript(GRAPH_EDGES_SCHEMA)
            conn.executescript(GRAPH_EXTRACTIONS_SCHEMA)
            conn.executescript(GRAPH_DIFFS_SCHEMA)
            conn.commit()

    def _is_fresh(self, entry: MemoryEntry) -> bool:
        row = self.fabric._connection_for(entry.plane).execute(
            "SELECT content_hash FROM graph_extractions WHERE source_entry_id = ?",
            (entry.entry_id,),
        ).fetchone()
        return bool(row and row["content_hash"] == entry.content_hash)

    def _record_extraction(self, conn: sqlite3.Connection, entry: MemoryEntry, now: str) -> None:
        conn.execute(
            """
            INSERT INTO graph_extractions (
                source_entry_id, plane, content_hash, source_artifact, extracted_at, extractor_version, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_entry_id) DO UPDATE SET
                plane = excluded.plane,
                content_hash = excluded.content_hash,
                source_artifact = excluded.source_artifact,
                extracted_at = excluded.extracted_at,
                extractor_version = excluded.extractor_version,
                metadata = excluded.metadata
            """,
            (
                entry.entry_id,
                entry.plane.value,
                entry.content_hash,
                entry.source_artifact,
                now,
                self.extractor_version,
                json.dumps(
                    {
                        "source_signal_id": entry.source_signal_id,
                        "encounter_type": self._encounter_type(entry),
                        "encounter_parties": self._encounter_parties(entry),
                        "structural_change": self._structural_change(entry),
                    },
                    ensure_ascii=False,
                ),
            ),
        )

    def _record_diff(self, conn: sqlite3.Connection, diff_tracker: dict[str, Any]) -> None:
        diff_summary = self._finalize_diff_tracker(diff_tracker)
        conn.execute(
            """
            INSERT INTO graph_diffs (diff_id, source_entry_id, plane, diffed_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(diff_id) DO UPDATE SET
                source_entry_id = excluded.source_entry_id,
                plane = excluded.plane,
                diffed_at = excluded.diffed_at,
                metadata = excluded.metadata
            """,
            (
                diff_tracker["diff_id"],
                diff_tracker["source_entry_id"],
                diff_tracker["plane"],
                diff_tracker["diffed_at"],
                json.dumps(diff_summary, ensure_ascii=False),
            ),
        )

    def _upsert_node(
        self,
        conn: sqlite3.Connection,
        *,
        node_id: str,
        label: str,
        node_type: str,
        description: str,
        salience: float,
        now: str,
        metadata: dict[str, Any],
        diff_tracker: dict[str, Any] | None = None,
    ) -> int:
        row = conn.execute(
            "SELECT node_id, metadata FROM graph_nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        previous_metadata = json.loads(row["metadata"] or "{}") if row and row["metadata"] else {}
        conn.execute(
            """
            INSERT INTO graph_nodes (
                node_id, label, node_type, description, activation_count, salience, created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                label = excluded.label,
                node_type = excluded.node_type,
                description = excluded.description,
                activation_count = graph_nodes.activation_count + 1,
                salience = CASE WHEN excluded.salience > graph_nodes.salience THEN excluded.salience ELSE graph_nodes.salience END,
                updated_at = excluded.updated_at,
                metadata = excluded.metadata
            """,
            (
                node_id,
                label,
                node_type,
                description,
                1,
                round(float(salience), 3),
                now,
                now,
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
        if diff_tracker is not None and row is None:
            diff_tracker["new_nodes"][node_id] = {
                "node_id": node_id,
                "label": label,
                "node_type": node_type,
                "metadata": metadata,
            }
        elif diff_tracker is not None and previous_metadata != metadata:
            diff_tracker["updated_nodes"][node_id] = {
                "node_id": node_id,
                "label": label,
                "node_type": node_type,
                "previous_metadata": previous_metadata,
                "metadata": metadata,
            }
        return 0 if row else 1

    def _upsert_edge(
        self,
        conn: sqlite3.Connection,
        *,
        source_node_id: str,
        target_node_id: str,
        relation_type: str,
        weight: float,
        provenance_entry_id: str | None,
        provenance_artifact: str | None,
        now: str,
        metadata: dict[str, Any],
        diff_tracker: dict[str, Any] | None = None,
    ) -> int:
        edge_id = self._stable_id(source_node_id, target_node_id, relation_type, provenance_entry_id or "")
        row = conn.execute(
            "SELECT edge_id, weight, metadata FROM graph_edges WHERE edge_id = ?",
            (edge_id,),
        ).fetchone()
        previous_weight = float(row["weight"]) if row and row["weight"] is not None else None
        previous_metadata = json.loads(row["metadata"] or "{}") if row and row["metadata"] else {}
        rounded_weight = round(float(weight), 3)
        conn.execute(
            """
            INSERT INTO graph_edges (
                edge_id, source_node_id, target_node_id, relation_type, weight,
                provenance_entry_id, provenance_artifact, created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                weight = excluded.weight,
                provenance_artifact = excluded.provenance_artifact,
                updated_at = excluded.updated_at,
                metadata = excluded.metadata
            """,
            (
                edge_id,
                source_node_id,
                target_node_id,
                relation_type,
                rounded_weight,
                provenance_entry_id,
                provenance_artifact,
                now,
                now,
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
        if diff_tracker is not None and row is None:
            diff_tracker["new_edges"][edge_id] = {
                "edge_id": edge_id,
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "relation_type": relation_type,
                "weight": rounded_weight,
                "metadata": metadata,
            }
        elif diff_tracker is not None and previous_weight is not None:
            if rounded_weight > previous_weight:
                diff_tracker["strengthened_edges"][edge_id] = {
                    "edge_id": edge_id,
                    "source_node_id": source_node_id,
                    "target_node_id": target_node_id,
                    "relation_type": relation_type,
                    "previous_weight": previous_weight,
                    "weight": rounded_weight,
                    "metadata": metadata,
                }
            elif previous_metadata != metadata:
                diff_tracker["updated_edges"][edge_id] = {
                    "edge_id": edge_id,
                    "source_node_id": source_node_id,
                    "target_node_id": target_node_id,
                    "relation_type": relation_type,
                    "weight": rounded_weight,
                    "previous_metadata": previous_metadata,
                    "metadata": metadata,
                }
        if diff_tracker is not None:
            cluster_key = self._cluster_key(source_node_id, target_node_id)
            diff_tracker["cluster_edge_counts"][cluster_key] = diff_tracker["cluster_edge_counts"].get(cluster_key, 0) + 1
        return 0 if row else 1

    def _rank_seed_nodes(self, conn: sqlite3.Connection, query_text: str, limit: int) -> list[dict[str, Any]]:
        concepts = self._extract_concepts(query_text, limit=8)
        tokens = {concept["normalized"] for concept in concepts}
        if not tokens:
            tokens = {token.lower() for token in TOKEN_PATTERN.findall(query_text) if len(token) >= 3}
        if not tokens:
            return []

        rows = conn.execute(
            "SELECT node_id, label, node_type, description, activation_count, salience, metadata FROM graph_nodes"
        ).fetchall()
        ranked: list[dict[str, Any]] = []
        for row in rows:
            label = row["label"] or ""
            description = row["description"] or ""
            metadata = json.loads(row["metadata"] or "{}")
            haystack = " ".join([label.lower(), description.lower(), json.dumps(metadata, ensure_ascii=False).lower()])
            overlap = [token for token in tokens if token in haystack]
            if not overlap:
                continue
            if not self._node_has_active_support(conn, row["node_id"]):
                continue

            score = 0.0
            for token in overlap:
                if token == label.lower():
                    score += 3.0
                else:
                    score += 1.25
            score += min(float(row["salience"] or 0.0), 2.0)
            score += min(int(row["activation_count"] or 0), 8) * 0.05
            ranked.append(
                {
                    "node_id": row["node_id"],
                    "label": label,
                    "node_type": row["node_type"],
                    "score": round(score, 3),
                    "overlap": overlap,
                }
            )

        ranked.sort(key=lambda item: (-item["score"], item["label"]))
        return ranked[:limit]

    def _expand_neighborhood(
        self,
        conn: sqlite3.Connection,
        seed_ids: list[str],
        *,
        depth: int,
        max_nodes: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        visited = set(seed_ids)
        queue = deque((node_id, 0) for node_id in seed_ids)
        edge_ids: set[str] = set()

        while queue and len(visited) < max_nodes:
            current, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            for edge in self._incident_edges(conn, current):
                if not self._edge_is_active(conn, edge):
                    continue
                edge_ids.add(edge["edge_id"])
                for neighbor in (edge["source_node_id"], edge["target_node_id"]):
                    if neighbor not in visited and self._node_has_active_support(conn, neighbor):
                        visited.add(neighbor)
                        queue.append((neighbor, current_depth + 1))
                        if len(visited) >= max_nodes:
                            break
                if len(visited) >= max_nodes:
                    break

        nodes = []
        for node_id in visited:
            row = conn.execute(
                "SELECT node_id, label, node_type, description, activation_count, salience, metadata FROM graph_nodes WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            if row is None:
                continue
            nodes.append(
                {
                    "node_id": row["node_id"],
                    "label": row["label"],
                    "node_type": row["node_type"],
                    "description": row["description"],
                    "activation_count": row["activation_count"],
                    "salience": row["salience"],
                    "metadata": json.loads(row["metadata"] or "{}"),
                }
            )

        edges = []
        for edge_id in edge_ids:
            row = conn.execute(
                """
                SELECT edge_id, source_node_id, target_node_id, relation_type, weight,
                       provenance_entry_id, provenance_artifact, metadata
                FROM graph_edges
                WHERE edge_id = ?
                """,
                (edge_id,),
            ).fetchone()
            if row is None or not self._edge_is_active(conn, row):
                continue
            edges.append(
                {
                    "edge_id": row["edge_id"],
                    "source_node_id": row["source_node_id"],
                    "target_node_id": row["target_node_id"],
                    "relation_type": row["relation_type"],
                    "weight": row["weight"],
                    "provenance_entry_id": row["provenance_entry_id"],
                    "provenance_artifact": row["provenance_artifact"],
                    "metadata": json.loads(row["metadata"] or "{}"),
                }
            )

        nodes.sort(key=lambda item: (-float(item["salience"]), item["label"]))
        edges.sort(key=lambda item: (-float(item["weight"]), item["relation_type"], item["source_node_id"]))
        return nodes, edges

    def _incident_edges(self, conn: sqlite3.Connection, node_id: str) -> list[sqlite3.Row]:
        rows = conn.execute(
            """
            SELECT edge_id, source_node_id, target_node_id, relation_type, weight,
                   provenance_entry_id, provenance_artifact, metadata
            FROM graph_edges
            WHERE source_node_id = ? OR target_node_id = ?
            ORDER BY weight DESC, updated_at DESC
            """,
            (node_id, node_id),
        ).fetchall()
        return list(rows)

    def _edge_is_active(self, conn: sqlite3.Connection, edge: sqlite3.Row | dict[str, Any]) -> bool:
        provenance_entry_id = edge["provenance_entry_id"] if isinstance(edge, sqlite3.Row) else edge.get("provenance_entry_id")
        if not provenance_entry_id:
            return True
        row = conn.execute(
            "SELECT revoked, quarantine_until FROM entries WHERE entry_id = ?",
            (provenance_entry_id,),
        ).fetchone()
        if row is None:
            return False
        if bool(row["revoked"]):
            return False
        quarantine_until = row["quarantine_until"]
        return quarantine_until in (None, "") or quarantine_until <= self.fabric._utc_now()

    def _node_has_active_support(self, conn: sqlite3.Connection, node_id: str) -> bool:
        if node_id.startswith("entry:"):
            entry_id = node_id.split(":", 1)[1]
            row = conn.execute(
                "SELECT revoked, quarantine_until FROM entries WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
            if row is None:
                return False
            if bool(row["revoked"]):
                return False
            quarantine_until = row["quarantine_until"]
            return quarantine_until in (None, "") or quarantine_until <= self.fabric._utc_now()

        rows = conn.execute(
            "SELECT provenance_entry_id FROM graph_edges WHERE source_node_id = ? OR target_node_id = ? LIMIT 25",
            (node_id, node_id),
        ).fetchall()
        return any(
            not row["provenance_entry_id"]
            or self._edge_is_active(conn, {"provenance_entry_id": row["provenance_entry_id"]})
            for row in rows
        )

    def _new_diff_tracker(self, entry: MemoryEntry, now: str) -> dict[str, Any]:
        return {
            "diff_id": self._stable_id(entry.entry_id, entry.content_hash, now),
            "source_entry_id": entry.entry_id,
            "plane": entry.plane.value,
            "diffed_at": now,
            "encounter_type": self._encounter_type(entry),
            "encounter_parties": self._encounter_parties(entry),
            "structural_change": self._structural_change(entry),
            "new_nodes": {},
            "updated_nodes": {},
            "new_edges": {},
            "updated_edges": {},
            "strengthened_edges": {},
            "cluster_edge_counts": {},
        }

    def _finalize_diff_tracker(self, diff_tracker: dict[str, Any]) -> dict[str, Any]:
        thickened_regions = [
            {"cluster": cluster, "new_internal_edges": count}
            for cluster, count in sorted(
                diff_tracker["cluster_edge_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            )
            if count >= 2
        ]
        return {
            "diff_id": diff_tracker["diff_id"],
            "source_entry_id": diff_tracker["source_entry_id"],
            "plane": diff_tracker["plane"],
            "diffed_at": diff_tracker["diffed_at"],
            "encounter_type": diff_tracker["encounter_type"],
            "encounter_parties": diff_tracker["encounter_parties"],
            "structural_change": diff_tracker["structural_change"],
            "new_nodes": list(diff_tracker["new_nodes"].values()),
            "updated_nodes": list(diff_tracker["updated_nodes"].values()),
            "new_edges": list(diff_tracker["new_edges"].values()),
            "updated_edges": list(diff_tracker["updated_edges"].values()),
            "strengthened_edges": list(diff_tracker["strengthened_edges"].values()),
            "thickened_regions": thickened_regions,
        }

    @staticmethod
    def _cluster_key(source_node_id: str, target_node_id: str) -> str:
        def prefix(node_id: str) -> str:
            if node_id.startswith("concept:"):
                return "concept"
            if node_id.startswith("entry:"):
                return "entry"
            return node_id.split(":", 1)[0]

        ordered = sorted([prefix(source_node_id), prefix(target_node_id)])
        return "::".join(ordered)

    @staticmethod
    def _encounter_type(entry: MemoryEntry) -> str:
        encounter_type = str(entry.metadata.get("encounter_type") or "").strip().lower()
        return encounter_type or "solo_reflection"

    @staticmethod
    def _encounter_parties(entry: MemoryEntry) -> list[str]:
        raw = entry.metadata.get("encounter_parties") or entry.metadata.get("parties") or []
        if isinstance(raw, str):
            raw = [raw]
        seen: set[str] = set()
        parties: list[str] = []
        for item in raw:
            value = str(item).strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            parties.append(value)
        if parties:
            return parties
        fallback = str(entry.metadata.get("origin") or "").strip()
        return [fallback] if fallback else []

    @classmethod
    def _structural_change(cls, entry: MemoryEntry) -> bool:
        if "structural_change" in entry.metadata:
            return bool(entry.metadata.get("structural_change"))
        return cls._encounter_type(entry) in {"correction", "challenge", "collaboration", "composting"}

    def _extract_concepts(self, text: str, limit: int = 8) -> list[dict[str, Any]]:
        counts = Counter()
        for token in TOKEN_PATTERN.findall(text.lower()):
            normalized = token.strip("'_-")
            if len(normalized) < 4 or normalized in STOPWORDS:
                continue
            counts[normalized] += 1

        concepts = []
        for token, count in counts.most_common(limit):
            score = min(0.55 + (count * 0.12), 1.0)
            concepts.append(
                {
                    "label": token.replace("_", " "),
                    "normalized": token,
                    "slug": self._slug(token),
                    "count": count,
                    "score": round(score, 3),
                }
            )
        return concepts

    @staticmethod
    def _normalize_claim_entry(item: Any) -> dict[str, Any]:
        if hasattr(item, "to_dict"):
            return item.to_dict()
        if hasattr(item, "__dict__"):
            return dict(item.__dict__)
        if isinstance(item, dict):
            return dict(item)
        return {"claim_text": str(item)}

    @staticmethod
    def _entry_node_id(entry_id: str) -> str:
        return f"entry:{entry_id}"

    @staticmethod
    def _entry_label(entry: MemoryEntry) -> str:
        if entry.source_artifact:
            return entry.source_artifact
        return f"memory {entry.entry_id[:8]}"

    @staticmethod
    def _claim_node_id(entry_id: str, claim_type: str, claim_text: str) -> str:
        return "claim:" + hashlib.sha256(f"{entry_id}|{claim_type}|{claim_text}".encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _stable_id(*parts: str) -> str:
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _slug(value: str) -> str:
        lowered = value.lower().strip()
        lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
        return lowered.strip("-") or "unknown"


__all__ = ["MemoryGraph"]
