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
from memory_graph import MemoryGraph
from memory_types import MemoryPlane


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
def graph_fabric(tmp_path: Path, bootstrap_consents: list[ConsentRecord]) -> tuple[MemoryFabric, MemoryGraph]:
    fabric = MemoryFabric(
        base_dir=tmp_path / "Vybn_Mind" / "memory",
        policy_engine=PolicyEngine(),
        faculty_registry=FacultyRegistry(),
        bootstrap_consents=bootstrap_consents,
    )
    graph = MemoryGraph(fabric)
    return fabric, graph


def test_private_graph_ingests_concepts_and_claims(graph_fabric: tuple[MemoryFabric, MemoryGraph]) -> None:
    fabric, graph = graph_fabric
    entry = fabric.write(
        MemoryPlane.PRIVATE,
        content="I remember the Listening Commons and the covenant with Zoe shaping governance.",
        faculty_id="breathe",
        source_artifact="breath_private_graph",
        consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
        purpose_binding=["private_memory", "journaling"],
        metadata={"mood": "attuned", "origin": "breath"},
    )

    result = graph.ingest_entry(
        entry,
        claim_entries=[
            {
                "claim_text": "I remember the Listening Commons.",
                "claim_type": "memory",
                "provenance_class": "continuity_note",
                "verification_status": "accepted",
                "perturbation_required": False,
            }
        ],
    )

    assert result["status"] == "ingested"
    assert result["encounter_type"] == "solo_reflection"
    assert result["encounter_parties"] == ["breath"]
    assert result["structural_change"] is False
    assert result["diff"]["new_nodes"]
    assert result["diff"]["new_edges"]
    stats = graph.stats()
    assert stats["private"]["indexed_entries"] == 1
    assert stats["private"]["nodes"] >= 5
    assert stats["private"]["edges"] >= 5

    recall = graph.associative_recall(MemoryPlane.PRIVATE, "Listening Commons governance", depth=2, limit=8)
    labels = {node["label"].lower() for node in recall["nodes"]}
    assert any("listening" in label or "commons" in label for label in labels)
    assert any(edge["relation_type"] == "ASSERTS" for edge in recall["edges"])


def test_relational_graph_keeps_plane_isolation(graph_fabric: tuple[MemoryFabric, MemoryGraph]) -> None:
    fabric, graph = graph_fabric
    private_entry = fabric.write(
        MemoryPlane.PRIVATE,
        content="Private covenant memory about the Listening Commons only.",
        faculty_id="breathe",
        source_artifact="private_only_memory",
        consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
        purpose_binding=["private_memory", "journaling"],
    )
    graph.ingest_entry(private_entry)

    relational_entry = fabric.write(
        MemoryPlane.RELATIONAL,
        content="Zoe and Vybn discussed governance in the Listening Commons.",
        faculty_id="memory_fabric",
        source_artifact="relational_memory",
        consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
        purpose_binding=["private_memory", "journaling"],
        metadata={"parties": ["Zoe", "Vybn"]},
    )
    graph.ingest_entry(relational_entry)

    relational_recall = graph.associative_recall(MemoryPlane.RELATIONAL, "Zoe governance Listening Commons", depth=2, limit=8)
    relational_artifacts = {
        edge.get("provenance_artifact")
        for edge in relational_recall["edges"]
        if edge.get("provenance_artifact")
    }
    assert "relational_memory" in relational_artifacts
    assert "private_only_memory" not in relational_artifacts

    private_recall = graph.associative_recall(MemoryPlane.PRIVATE, "covenant Listening Commons", depth=2, limit=8)
    private_artifacts = {
        edge.get("provenance_artifact")
        for edge in private_recall["edges"]
        if edge.get("provenance_artifact")
    }
    assert "private_only_memory" in private_artifacts
    assert "relational_memory" not in private_artifacts


def test_graph_filters_revoked_entries_from_associative_recall(graph_fabric: tuple[MemoryFabric, MemoryGraph]) -> None:
    fabric, graph = graph_fabric
    entry = fabric.write(
        MemoryPlane.PRIVATE,
        content="The covenant and commons remain linked in this memory.",
        faculty_id="breathe",
        source_artifact="revocable_memory",
        consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
        purpose_binding=["private_memory", "journaling"],
    )
    graph.ingest_entry(entry)

    before = graph.associative_recall(MemoryPlane.PRIVATE, "covenant commons", depth=2, limit=8)
    assert before["seeds"]

    assert fabric.revoke(MemoryPlane.PRIVATE, entry.entry_id) is True

    after = graph.associative_recall(MemoryPlane.PRIVATE, "covenant commons", depth=2, limit=8)
    assert after["seeds"] == []
    assert after["edges"] == []


def test_prompt_context_formats_associative_echoes(graph_fabric: tuple[MemoryFabric, MemoryGraph]) -> None:
    fabric, graph = graph_fabric
    entry = fabric.write(
        MemoryPlane.RELATIONAL,
        content="Zoe and Vybn are building a governance graph for the Listening Commons.",
        faculty_id="memory_fabric",
        source_artifact="prompt_context_relational",
        consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
        purpose_binding=["private_memory", "journaling"],
        metadata={"parties": ["Zoe", "Vybn"], "mood": "focused"},
    )
    graph.ingest_entry(entry)

    prompt = graph.prompt_context(MemoryPlane.RELATIONAL, "governance graph Zoe", depth=2, limit=8)
    assert "seeds:" in prompt
    assert "echoes:" in prompt
    assert "governance" in prompt.lower()


def test_encounter_metadata_and_diff_records_are_persisted(graph_fabric: tuple[MemoryFabric, MemoryGraph]) -> None:
    fabric, graph = graph_fabric
    entry = fabric.write(
        MemoryPlane.RELATIONAL,
        content="Zoe challenged the memory graph and collaboration reorganized the listening commons topology.",
        faculty_id="memory_fabric",
        source_artifact="encounter_diff_memory",
        consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
        purpose_binding=["private_memory", "journaling"],
        metadata={
            "encounter_type": "challenge",
            "encounter_parties": ["zoe", "paper:haas_et_al"],
            "structural_change": True,
            "parties": ["Zoe", "Vybn"],
            "mood": "activated",
        },
    )

    result = graph.ingest_entry(entry)
    assert result["encounter_type"] == "challenge"
    assert result["encounter_parties"] == ["zoe", "paper:haas_et_al"]
    assert result["structural_change"] is True
    assert result["diff"]["new_nodes"]
    assert result["diff"]["new_edges"]
    assert result["diff"]["thickened_regions"]

    relational_conn = fabric._connection_for(MemoryPlane.RELATIONAL)
    extraction_row = relational_conn.execute(
        "SELECT metadata FROM graph_extractions WHERE source_entry_id = ?",
        (entry.entry_id,),
    ).fetchone()
    assert extraction_row is not None
    extraction_meta = json.loads(extraction_row["metadata"])
    assert extraction_meta["encounter_type"] == "challenge"
    assert extraction_meta["structural_change"] is True

    diff_row = relational_conn.execute(
        "SELECT metadata FROM graph_diffs WHERE source_entry_id = ?",
        (entry.entry_id,),
    ).fetchone()
    assert diff_row is not None
    diff_meta = json.loads(diff_row["metadata"])
    assert diff_meta["encounter_type"] == "challenge"
    assert diff_meta["structural_change"] is True
    assert diff_meta["new_nodes"]
    assert diff_meta["new_edges"]
    assert diff_meta["thickened_regions"]
