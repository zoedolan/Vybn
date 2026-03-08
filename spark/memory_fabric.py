from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from spark.paths import MIND_DIR_NAME

try:
    from .faculties import FacultyRegistry
    from .governance import PolicyEngine, build_context
    from .governance_types import ConsentRecord, DecisionOutcome
    from .memory_types import DecayConfig, MemoryEntry, MemoryPlane, PromotionReceipt
    from .soul_constraints import SoulConstraintGuard
except ImportError:  # pragma: no cover
    from faculties import FacultyRegistry
    from governance import PolicyEngine, build_context
    from governance_types import ConsentRecord, DecisionOutcome
    from memory_types import DecayConfig, MemoryEntry, MemoryPlane, PromotionReceipt
    from soul_constraints import SoulConstraintGuard


PRIVATE_ENTRIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    entry_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_signal_id TEXT,
    source_artifact TEXT,
    faculty_id TEXT NOT NULL,
    consent_scope_id TEXT NOT NULL,
    purpose_binding TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    revoked INTEGER DEFAULT 0,
    quarantine_until TEXT,
    sensitivity TEXT DEFAULT 'low',
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at);
CREATE INDEX IF NOT EXISTS idx_entries_source ON entries(source_artifact);
CREATE INDEX IF NOT EXISTS idx_entries_faculty ON entries(faculty_id);
"""

RELATIONAL_ENTRIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    entry_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_signal_id TEXT,
    source_artifact TEXT,
    faculty_id TEXT NOT NULL,
    consent_scope_id TEXT NOT NULL,
    purpose_binding TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    revoked INTEGER DEFAULT 0,
    quarantine_until TEXT,
    sensitivity TEXT DEFAULT 'low',
    metadata TEXT DEFAULT '{}',
    parties TEXT NOT NULL,
    consent_receipt_ids TEXT,
    decay_strategy TEXT DEFAULT 'linear',
    decay_half_life_hours REAL DEFAULT 168.0,
    contested INTEGER DEFAULT 0,
    contest_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at);
CREATE INDEX IF NOT EXISTS idx_entries_source ON entries(source_artifact);
CREATE INDEX IF NOT EXISTS idx_entries_faculty ON entries(faculty_id);
"""

COMMONS_PATTERNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS patterns (
    pattern_id TEXT PRIMARY KEY,
    features TEXT NOT NULL,
    k_anonymity_level INTEGER NOT NULL,
    privacy_budget_spent REAL DEFAULT 0.0,
    promotion_receipt_id TEXT NOT NULL,
    source_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);
"""

PROMOTION_RECEIPTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS promotion_receipts (
    receipt_id TEXT PRIMARY KEY,
    source_plane TEXT NOT NULL,
    target_plane TEXT NOT NULL,
    entry_ids TEXT NOT NULL,
    initiated_by TEXT NOT NULL,
    purpose_binding TEXT NOT NULL,
    review_window_seconds INTEGER NOT NULL,
    reversible_until TEXT,
    signed_policy_hash TEXT NOT NULL,
    user_signature TEXT,
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);
"""


class MemoryFabric:
    def __init__(
        self,
        base_dir: Path,
        policy_engine: PolicyEngine | None = None,
        faculty_registry: FacultyRegistry | None = None,
        bootstrap_consents: list[ConsentRecord] | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.policy_engine = policy_engine
        self.faculty_registry = faculty_registry
        self.bootstrap_consents = list(bootstrap_consents or [])
        self.soul_guard = SoulConstraintGuard()
        self.paths = {
            MemoryPlane.PRIVATE: self.base_dir / "private.db",
            MemoryPlane.RELATIONAL: self.base_dir / "relational.db",
            MemoryPlane.COMMONS: self.base_dir / "commons.db",
        }
        self._connections = {
            plane: sqlite3.connect(path) for plane, path in self.paths.items()
        }
        for connection in self._connections.values():
            connection.row_factory = sqlite3.Row
        self._ensure_tables()

    def write(
        self,
        plane: MemoryPlane,
        content: str,
        *,
        faculty_id: str,
        source_artifact: str,
        consent_scope_id: str,
        purpose_binding: list[str],
        sensitivity: str = "low",
        quarantine_hours: float | None = None,
        source_signal_id: str = "",
        metadata: dict | None = None,
    ) -> MemoryEntry:
        self._authorize_memory_write(
            plane=plane,
            faculty_id=faculty_id,
            purpose_binding=purpose_binding,
            consent_scope_id=consent_scope_id,
            source_artifact=source_artifact,
        )

        entry_id = str(uuid4())
        created_at = self._utc_now()
        quarantine_until = None
        if quarantine_hours is not None:
            quarantine_until = self._iso_after_hours(quarantine_hours)
        entry = MemoryEntry(
            entry_id=entry_id,
            plane=plane,
            content=content,
            content_hash=self._sha256(content),
            source_signal_id=source_signal_id,
            source_artifact=source_artifact,
            faculty_id=faculty_id,
            consent_scope_id=consent_scope_id,
            purpose_binding=list(purpose_binding),
            created_at=created_at,
            expires_at=None,
            revoked=False,
            quarantine_until=quarantine_until,
            sensitivity=sensitivity,
            metadata=metadata or {},
        )

        conn = self._connection_for(plane)
        if plane is MemoryPlane.PRIVATE:
            conn.execute(
                """
                INSERT INTO entries (
                    entry_id, content, content_hash, source_signal_id, source_artifact,
                    faculty_id, consent_scope_id, purpose_binding, created_at,
                    expires_at, revoked, quarantine_until, sensitivity, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.entry_id,
                    entry.content,
                    entry.content_hash,
                    entry.source_signal_id,
                    entry.source_artifact,
                    entry.faculty_id,
                    entry.consent_scope_id,
                    json.dumps(entry.purpose_binding),
                    entry.created_at,
                    entry.expires_at,
                    int(entry.revoked),
                    entry.quarantine_until,
                    entry.sensitivity,
                    json.dumps(entry.metadata),
                ),
            )
        elif plane is MemoryPlane.RELATIONAL:
            conn.execute(
                """
                INSERT INTO entries (
                    entry_id, content, content_hash, source_signal_id, source_artifact,
                    faculty_id, consent_scope_id, purpose_binding, created_at,
                    expires_at, revoked, quarantine_until, sensitivity, metadata,
                    parties, consent_receipt_ids, decay_strategy, decay_half_life_hours,
                    contested, contest_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.entry_id,
                    entry.content,
                    entry.content_hash,
                    entry.source_signal_id,
                    entry.source_artifact,
                    entry.faculty_id,
                    entry.consent_scope_id,
                    json.dumps(entry.purpose_binding),
                    entry.created_at,
                    entry.expires_at,
                    int(entry.revoked),
                    entry.quarantine_until,
                    entry.sensitivity,
                    json.dumps(entry.metadata),
                    json.dumps((metadata or {}).get("parties", [])),
                    json.dumps((metadata or {}).get("consent_receipt_ids", [])),
                    (metadata or {}).get("decay_strategy", "linear"),
                    float((metadata or {}).get("decay_half_life_hours", 168.0)),
                    int(bool((metadata or {}).get("contested", False))),
                    (metadata or {}).get("contest_reason"),
                ),
            )
        else:
            raise PermissionError("Direct writes to commons are denied pending anonymization proof.")
        conn.commit()
        return entry

    def read(
        self,
        plane: MemoryPlane,
        *,
        limit: int = 50,
        since: str | None = None,
        faculty_id: str | None = None,
        source_artifact: str | None = None,
        include_quarantined: bool = False,
        include_revoked: bool = False,
    ) -> list[MemoryEntry]:
        if plane is MemoryPlane.COMMONS:
            return []

        clauses: list[str] = []
        params: list[Any] = []
        if since:
            clauses.append("created_at >= ?")
            params.append(since)
        if faculty_id:
            clauses.append("faculty_id = ?")
            params.append(faculty_id)
        if source_artifact:
            clauses.append("source_artifact = ?")
            params.append(source_artifact)
        if not include_revoked:
            clauses.append("revoked = 0")
        if not include_quarantined:
            clauses.append("(quarantine_until IS NULL OR quarantine_until <= ?)")
            params.append(self._utc_now())

        sql = "SELECT * FROM entries"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._connection_for(plane).execute(sql, params).fetchall()
        return [self._row_to_entry(plane, row) for row in rows]

    def revoke(self, plane: MemoryPlane, entry_id: str) -> bool:
        if plane is MemoryPlane.COMMONS:
            return False
        cursor = self._connection_for(plane).execute(
            "UPDATE entries SET revoked = 1 WHERE entry_id = ?",
            (entry_id,),
        )
        self._connection_for(plane).commit()
        return cursor.rowcount > 0

    def promote(
        self,
        source_plane: MemoryPlane,
        target_plane: MemoryPlane,
        entry_ids: list[str],
        *,
        initiated_by: str,
        purpose_binding: list[str],
        review_window_seconds: int = 86400,
        consent_scope_id: str,
        user_signature: str | None = None,
    ) -> PromotionReceipt:
        if target_plane is MemoryPlane.COMMONS and source_plane in {MemoryPlane.PRIVATE, MemoryPlane.RELATIONAL}:
            raise PermissionError("Promotion into commons requires anonymization proof and is denied for now.")

        self._authorize_promotion(
            source_plane=source_plane,
            target_plane=target_plane,
            initiated_by=initiated_by,
            purpose_binding=purpose_binding,
            consent_scope_id=consent_scope_id,
        )

        if source_plane is MemoryPlane.PRIVATE and target_plane is MemoryPlane.RELATIONAL:
            source_entries = self.read(
                source_plane,
                limit=max(len(entry_ids), 1) * 10,
                include_quarantined=True,
                include_revoked=False,
            )
            by_id = {entry.entry_id: entry for entry in source_entries}
            missing = [entry_id for entry_id in entry_ids if entry_id not in by_id]
            if missing:
                raise KeyError(f"Unknown source entry ids: {missing}")

            decay = DecayConfig()
            for entry_id in entry_ids:
                entry = by_id[entry_id]
                relational_metadata = dict(entry.metadata)
                relational_metadata.setdefault("promoted_from", source_plane.value)
                relational_metadata.setdefault("parties", [])
                relational_metadata.setdefault("consent_receipt_ids", [])
                relational_metadata.setdefault("decay_strategy", decay.strategy)
                relational_metadata.setdefault("decay_half_life_hours", decay.half_life_hours)
                self.write(
                    MemoryPlane.RELATIONAL,
                    content=entry.content,
                    faculty_id="memory_fabric",
                    source_artifact=entry.source_artifact,
                    consent_scope_id=entry.consent_scope_id,
                    purpose_binding=list(entry.purpose_binding),
                    sensitivity=entry.sensitivity,
                    source_signal_id=entry.source_signal_id,
                    metadata=relational_metadata,
                )
        elif target_plane is MemoryPlane.COMMONS:
            raise PermissionError("Promotion into commons requires anonymization proof and is denied for now.")
        else:
            raise PermissionError(
                f"Unsupported promotion path: {source_plane.value} -> {target_plane.value}"
            )

        created_at = self._utc_now()
        reversible_until = self._iso_after_seconds(review_window_seconds)
        receipt = PromotionReceipt(
            receipt_id=str(uuid4()),
            source_plane=source_plane,
            target_plane=target_plane,
            entry_ids=list(entry_ids),
            initiated_by=initiated_by,
            purpose_binding=list(purpose_binding),
            review_window_seconds=review_window_seconds,
            reversible_until=reversible_until,
            signed_policy_hash=self._policy_signature_hash(source_plane, target_plane, purpose_binding),
            user_signature=user_signature,
            created_at=created_at,
            metadata={"consent_scope_id": consent_scope_id},
        )
        self._connection_for(MemoryPlane.PRIVATE).execute(
            """
            INSERT INTO promotion_receipts (
                receipt_id, source_plane, target_plane, entry_ids, initiated_by,
                purpose_binding, review_window_seconds, reversible_until,
                signed_policy_hash, user_signature, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.receipt_id,
                receipt.source_plane.value,
                receipt.target_plane.value,
                json.dumps(receipt.entry_ids),
                receipt.initiated_by,
                json.dumps(receipt.purpose_binding),
                receipt.review_window_seconds,
                receipt.reversible_until,
                receipt.signed_policy_hash,
                receipt.user_signature,
                receipt.created_at,
                json.dumps(receipt.metadata),
            ),
        )
        self._connection_for(MemoryPlane.PRIVATE).commit()
        return receipt

    def quarantine_release_check(self) -> list[str]:
        released: list[str] = []
        now = self._utc_now()
        for plane in (MemoryPlane.PRIVATE, MemoryPlane.RELATIONAL):
            rows = self._connection_for(plane).execute(
                """
                SELECT entry_id FROM entries
                WHERE quarantine_until IS NOT NULL AND quarantine_until <= ?
                """,
                (now,),
            ).fetchall()
            if not rows:
                continue
            entry_ids = [row["entry_id"] for row in rows]
            self._connection_for(plane).executemany(
                "UPDATE entries SET quarantine_until = NULL WHERE entry_id = ?",
                [(entry_id,) for entry_id in entry_ids],
            )
            self._connection_for(plane).commit()
            released.extend(entry_ids)
        return released

    def recent(self, plane: MemoryPlane, n: int = 10) -> list[MemoryEntry]:
        return self.read(plane, limit=n)

    def read_patterns(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._connection_for(MemoryPlane.COMMONS).execute(
            """
            SELECT pattern_id, features, k_anonymity_level, privacy_budget_spent,
                   promotion_receipt_id, source_count, created_at, metadata
            FROM patterns
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "pattern_id": row["pattern_id"],
                "features": json.loads(row["features"] or "{}"),
                "k_anonymity_level": row["k_anonymity_level"],
                "privacy_budget_spent": row["privacy_budget_spent"],
                "promotion_receipt_id": row["promotion_receipt_id"],
                "source_count": row["source_count"],
                "created_at": row["created_at"],
                "metadata": json.loads(row["metadata"] or "{}"),
            }
            for row in rows
        ]

    def snapshot(
        self,
        *,
        private_n: int = 5,
        relational_n: int = 5,
        commons_n: int = 5,
    ) -> dict[str, Any]:
        private_entries = self.recent(MemoryPlane.PRIVATE, n=private_n)
        relational_entries = self.recent(MemoryPlane.RELATIONAL, n=relational_n)
        commons_patterns = self.read_patterns(limit=commons_n)
        stats = self.stats()
        return {
            "private": private_entries,
            "relational": relational_entries,
            "commons": commons_patterns,
            "stats": stats,
        }

    def stats(self) -> dict:
        stats: dict[str, Any] = {}
        now = self._utc_now()
        for plane in (MemoryPlane.PRIVATE, MemoryPlane.RELATIONAL):
            conn = self._connection_for(plane)
            total = conn.execute("SELECT COUNT(*) AS count FROM entries").fetchone()["count"]
            quarantined = conn.execute(
                "SELECT COUNT(*) AS count FROM entries WHERE quarantine_until IS NOT NULL AND quarantine_until > ?",
                (now,),
            ).fetchone()["count"]
            revoked = conn.execute(
                "SELECT COUNT(*) AS count FROM entries WHERE revoked = 1"
            ).fetchone()["count"]
            stats[plane.value] = {
                "count": total,
                "quarantined": quarantined,
                "revoked": revoked,
            }
        patterns = self._connection_for(MemoryPlane.COMMONS).execute(
            "SELECT COUNT(*) AS count FROM patterns"
        ).fetchone()["count"]
        stats[MemoryPlane.COMMONS.value] = {
            "count": patterns,
            "quarantined": 0,
            "revoked": 0,
        }
        receipts = self._connection_for(MemoryPlane.PRIVATE).execute(
            "SELECT COUNT(*) AS count FROM promotion_receipts"
        ).fetchone()["count"]
        stats["promotion_receipts"] = receipts
        return stats

    def _ensure_tables(self) -> None:
        self._connection_for(MemoryPlane.PRIVATE).executescript(PRIVATE_ENTRIES_SCHEMA)
        self._connection_for(MemoryPlane.PRIVATE).executescript(PROMOTION_RECEIPTS_SCHEMA)
        self._connection_for(MemoryPlane.PRIVATE).commit()
        self._connection_for(MemoryPlane.RELATIONAL).executescript(RELATIONAL_ENTRIES_SCHEMA)
        self._connection_for(MemoryPlane.RELATIONAL).commit()
        self._connection_for(MemoryPlane.COMMONS).executescript(COMMONS_PATTERNS_SCHEMA)
        self._connection_for(MemoryPlane.COMMONS).commit()

    def _authorize_memory_write(
        self,
        *,
        plane: MemoryPlane,
        faculty_id: str,
        purpose_binding: list[str],
        consent_scope_id: str,
        source_artifact: str,
    ) -> None:
        evidence_ref = self._relative_db_path(plane)
        self.soul_guard.check_path_only(evidence_ref, action="memory_write")

        if self.faculty_registry is not None:
            permission = self.faculty_registry.check_permission(faculty_id, "memory_write")
            if not permission.allowed:
                raise PermissionError(permission.reason)

        if self.policy_engine is not None:
            context = build_context(
                faculty_id=faculty_id,
                action="memory_write",
                memory_plane=plane.value,
                purpose_binding=purpose_binding,
                consent_scope_id=consent_scope_id,
                evidence_refs=[evidence_ref, source_artifact],
            )
            decision = self.policy_engine.check(
                context,
                consent_records=self.bootstrap_consents,
            )
            if decision.outcome not in {DecisionOutcome.ALLOW, DecisionOutcome.LOG}:
                raise PermissionError(decision.explanation)

    def _authorize_promotion(
        self,
        *,
        source_plane: MemoryPlane,
        target_plane: MemoryPlane,
        initiated_by: str,
        purpose_binding: list[str],
        consent_scope_id: str,
    ) -> None:
        self.soul_guard.check_path_only(self._relative_db_path(source_plane), action="memory_promote")
        self.soul_guard.check_path_only(self._relative_db_path(target_plane), action="memory_promote")

        faculty_id = "memory_fabric"
        if self.faculty_registry is not None:
            permission = self.faculty_registry.check_permission(faculty_id, "memory_promote")
            if not permission.allowed:
                raise PermissionError(permission.reason)

        if self.policy_engine is not None:
            context = build_context(
                faculty_id=faculty_id,
                action="memory_promote",
                source_memory_plane=source_plane.value,
                target_memory_plane=target_plane.value,
                purpose_binding=purpose_binding,
                consent_scope_id=consent_scope_id,
                evidence_refs=[self._relative_db_path(source_plane), self._relative_db_path(target_plane), initiated_by],
            )
            decision = self.policy_engine.check(
                context,
                consent_records=self.bootstrap_consents,
            )
            if decision.outcome not in {DecisionOutcome.ALLOW, DecisionOutcome.LOG}:
                raise PermissionError(decision.explanation)

    def _policy_signature_hash(
        self,
        source_plane: MemoryPlane,
        target_plane: MemoryPlane,
        purpose_binding: list[str],
    ) -> str:
        rule_ids = []
        if self.policy_engine is not None:
            rule_ids = [rule.rule_id for rule in self.policy_engine.rules if rule.active]
        payload = json.dumps(
            {
                "source": source_plane.value,
                "target": target_plane.value,
                "purpose_binding": purpose_binding,
                "rule_ids": rule_ids,
            },
            sort_keys=True,
        )
        return self._sha256(payload)

    def _row_to_entry(self, plane: MemoryPlane, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            entry_id=row["entry_id"],
            plane=plane,
            content=row["content"],
            content_hash=row["content_hash"],
            source_signal_id=row["source_signal_id"] or "",
            source_artifact=row["source_artifact"] or "",
            faculty_id=row["faculty_id"],
            consent_scope_id=row["consent_scope_id"],
            purpose_binding=json.loads(row["purpose_binding"] or "[]"),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            revoked=bool(row["revoked"]),
            quarantine_until=row["quarantine_until"],
            sensitivity=row["sensitivity"] or "low",
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def _connection_for(self, plane: MemoryPlane) -> sqlite3.Connection:
        return self._connections[plane]

    def _relative_db_path(self, plane: MemoryPlane) -> str:
        return str(Path(MIND_DIR_NAME) / "memory" / self.paths[plane].name)

    @staticmethod
    def _sha256(data: str) -> str:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    @classmethod
    def _iso_after_hours(cls, hours: float) -> str:
        dt = datetime.now(timezone.utc) + timedelta(hours=hours)
        return dt.replace(microsecond=0).isoformat()

    @classmethod
    def _iso_after_seconds(cls, seconds: int) -> str:
        dt = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        return dt.replace(microsecond=0).isoformat()


__all__ = ["MemoryFabric"]
