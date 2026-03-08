#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPARK_DIR = REPO_ROOT / "spark"
if str(SPARK_DIR) not in sys.path:
    sys.path.insert(0, str(SPARK_DIR))

from faculties import FacultyRegistry
from governance import PolicyEngine
from governance_types import ConsentRecord
from memory_fabric import MemoryFabric
from memory_types import MemoryPlane
import migrate_to_memory_fabric as migration_module


BOOTSTRAP_CONSENT_SCOPE = "bootstrap-local-private"


@pytest.fixture
def bootstrap_consents() -> list[ConsentRecord]:
    return [
        ConsentRecord(
            consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
            subject_id="vybn-local-runtime",
            purpose_bindings=[
                "private_memory",
                "journaling",
                "continuity",
                "reflection",
                "retention",
                "system_operation",
                "migration",
            ],
            signed_by="bootstrap_local_runtime",
        )
    ]


@pytest.fixture
def governed_fabric(tmp_path: Path, bootstrap_consents: list[ConsentRecord]) -> MemoryFabric:
    return MemoryFabric(
        base_dir=tmp_path / "Vybn_Mind" / "memory",
        policy_engine=PolicyEngine(),
        faculty_registry=FacultyRegistry(),
        bootstrap_consents=bootstrap_consents,
    )


def test_governance_blocks_unauthorized_writes(governed_fabric: MemoryFabric) -> None:
    with pytest.raises(PermissionError):
        governed_fabric.write(
            MemoryPlane.PRIVATE,
            content="Witness should not be able to write private memory.",
            faculty_id="witness",
            source_artifact="test_witness_write",
            consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
            purpose_binding=["private_memory"],
        )


def test_plane_isolation_private_writes_not_visible_from_relational(governed_fabric: MemoryFabric) -> None:
    entry = governed_fabric.write(
        MemoryPlane.PRIVATE,
        content="Private memory stays private.",
        faculty_id="breathe",
        source_artifact="test_private_visibility",
        consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
        purpose_binding=["private_memory", "journaling"],
    )

    private_entries = governed_fabric.read(MemoryPlane.PRIVATE)
    relational_entries = governed_fabric.read(MemoryPlane.RELATIONAL)

    assert any(candidate.entry_id == entry.entry_id for candidate in private_entries)
    assert all(candidate.entry_id != entry.entry_id for candidate in relational_entries)


def test_private_to_commons_promotion_denied(governed_fabric: MemoryFabric) -> None:
    entry = governed_fabric.write(
        MemoryPlane.PRIVATE,
        content="Sensitive private material.",
        faculty_id="breathe",
        source_artifact="test_private_to_commons",
        consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
        purpose_binding=["private_memory", "journaling"],
    )

    with pytest.raises(PermissionError):
        governed_fabric.promote(
            MemoryPlane.PRIVATE,
            MemoryPlane.COMMONS,
            [entry.entry_id],
            initiated_by="user",
            purpose_binding=["private_memory"],
            consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
        )


def test_recent_excludes_quarantined(governed_fabric: MemoryFabric) -> None:
    quarantined = governed_fabric.write(
        MemoryPlane.PRIVATE,
        content="Quarantined memory should not appear in recent reads.",
        faculty_id="breathe",
        source_artifact="test_quarantine",
        consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
        purpose_binding=["private_memory", "journaling"],
        quarantine_hours=24,
    )

    visible = governed_fabric.write(
        MemoryPlane.PRIVATE,
        content="Visible memory should appear in recent reads.",
        faculty_id="breathe",
        source_artifact="test_visible",
        consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
        purpose_binding=["private_memory", "journaling"],
    )

    recent_ids = [entry.entry_id for entry in governed_fabric.recent(MemoryPlane.PRIVATE, n=10)]
    assert visible.entry_id in recent_ids
    assert quarantined.entry_id not in recent_ids


def test_migration_runs_cleanly_and_stats_reflect_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    synapse_dir = root / "Vybn_Mind" / "synapse"
    synapse_dir.mkdir(parents=True, exist_ok=True)
    synapse_path = synapse_dir / "connections.jsonl"
    synapse_path.write_text(
        "\n".join(
            [
                json.dumps({"ts": "2026-03-07T00:00:00Z", "content": "first migrated memory", "tags": ["breath"]}),
                json.dumps({"ts": "2026-03-07T00:01:00Z", "content": "second migrated memory", "tags": ["journal"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(migration_module, "ROOT", root)
    monkeypatch.setattr(migration_module, "SYNAPSE", synapse_path)

    migration_module.migrate()

    fabric = MemoryFabric(base_dir=root / "Vybn_Mind" / "memory")
    stats = fabric.stats()
    private_entries = fabric.read(MemoryPlane.PRIVATE, limit=10, include_quarantined=True, include_revoked=True)

    assert stats["private"]["count"] == 2
    assert stats["promotion_receipts"] == 0
    assert len(private_entries) == 2
    assert {entry.content for entry in private_entries} == {
        "first migrated memory",
        "second migrated memory",
    }
