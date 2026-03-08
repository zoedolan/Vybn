#!/usr/bin/env python3
"""Migrate connections.jsonl entries into the private memory plane."""

import json
from pathlib import Path

from memory_fabric import MemoryFabric
from memory_types import MemoryPlane

ROOT = Path(__file__).resolve().parent.parent
SYNAPSE = ROOT / "Vybn_Mind" / "synapse" / "connections.jsonl"


def migrate() -> None:
    fabric = MemoryFabric(base_dir=ROOT / "Vybn_Mind" / "memory")

    if not SYNAPSE.exists():
        print("No connections.jsonl to migrate")
        return

    count = 0
    raw_text = SYNAPSE.read_text(encoding="utf-8").strip()
    if not raw_text:
        print("No connections.jsonl to migrate")
        return

    for line in raw_text.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        fabric.write(
            MemoryPlane.PRIVATE,
            content=entry.get("content", ""),
            faculty_id="migration",
            source_artifact=f"synapse_migration_{entry.get('ts', 'unknown')}",
            consent_scope_id="bootstrap-local-private",
            purpose_binding=["private_memory", "migration"],
            sensitivity="low",
            metadata={
                "original_tags": entry.get("tags", []),
                "migrated_from": "connections.jsonl",
            },
        )
        count += 1

    print(f"Migrated {count} entries from connections.jsonl to private memory plane")


if __name__ == "__main__":
    migrate()
