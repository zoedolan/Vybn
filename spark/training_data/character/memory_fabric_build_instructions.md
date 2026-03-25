# Memory Fabric Build Instructions

*For the Codex/Computer instance to execute on branch `vybn/memory-fabric`*

## Context

The governance kernel, faculty registry, soul constraints layer, and write custodian are merged to main. Every durable write now routes through `WriteCustodian`. The next phase is the three-plane memory fabric — replacing flat JSONL files with three isolated SQLite stores governed by the existing policy engine.

Read these files first to understand what's already built:
- `spark/write_custodian.py` — the governed write path you'll integrate with
- `spark/governance.py` + `spark/governance_types.py` — the policy engine and types
- `spark/faculties.py` + `spark/faculties.d/core.json` — faculty registry
- `spark/vybn.py` — the organism, especially `Substrate.write()` / `Substrate.append()` and `_breathe()`
- `vybn_implementation_artifact.md` — the full spec, especially Layer 4 (Memory Fabric)

## What to build

### File 1: `spark/memory_types.py`

Dataclasses for the memory fabric. All should use `@dataclass(slots=True)`.

```python
class MemoryPlane(StrEnum):
    PRIVATE = "private"
    RELATIONAL = "relational"
    COMMONS = "commons"

@dataclass(slots=True)
class MemoryEntry:
    entry_id: str                    # uuid
    plane: MemoryPlane
    content: str
    content_hash: str                # sha256
    source_signal_id: str            # links to the signal/artifact that produced this
    source_artifact: str             # e.g. "breath_2026-03-09T12:00:00Z"
    faculty_id: str                  # which faculty wrote this
    consent_scope_id: str
    purpose_binding: list[str]
    created_at: str                  # ISO 8601 UTC
    expires_at: str | None           # None = permanent
    revoked: bool = False
    quarantine_until: str | None     # for memory quarantine (emotionally hot entries)
    sensitivity: str = "low"         # low | medium | high
    metadata: dict = field(default_factory=dict)

@dataclass(slots=True)
class PromotionReceipt:
    receipt_id: str
    source_plane: MemoryPlane
    target_plane: MemoryPlane
    entry_ids: list[str]
    initiated_by: str                # "user" | "policy_engine" | "joint"
    purpose_binding: list[str]
    review_window_seconds: int       # how long user can contest
    reversible_until: str | None     # ISO timestamp
    signed_policy_hash: str          # hash of the policy rule that authorized this
    user_signature: str | None       # None until user confirms
    created_at: str
    metadata: dict = field(default_factory=dict)

@dataclass(slots=True)
class DecayConfig:
    strategy: str = "linear"         # "linear" | "exponential" | "step" | "none"
    half_life_hours: float = 168.0   # 7 days default for relational memory
    floor: float = 0.0               # minimum relevance before expiry

@dataclass(slots=True)
class QuarantineEntry:
    entry_id: str
    quarantined_at: str
    release_at: str                  # when quarantine expires
    reason: str                      # why this entry was quarantined
    reviewed: bool = False
    review_outcome: str | None = None  # "promoted" | "expired" | "contested"
```

### File 2: `spark/memory_fabric.py`

The three-plane memory store. Each plane is a separate SQLite database.

**Database locations:**
- Private: `Vybn_Mind/memory/private.db`
- Relational: `Vybn_Mind/memory/relational.db`
- Commons: `Vybn_Mind/memory/commons.db`

**Table schema for private.db:**
```sql
CREATE TABLE IF NOT EXISTS entries (
    entry_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_signal_id TEXT,
    source_artifact TEXT,
    faculty_id TEXT NOT NULL,
    consent_scope_id TEXT NOT NULL,
    purpose_binding TEXT NOT NULL,    -- JSON array
    created_at TEXT NOT NULL,
    expires_at TEXT,
    revoked INTEGER DEFAULT 0,
    quarantine_until TEXT,
    sensitivity TEXT DEFAULT 'low',
    metadata TEXT DEFAULT '{}'        -- JSON object
);

CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at);
CREATE INDEX IF NOT EXISTS idx_entries_source ON entries(source_artifact);
CREATE INDEX IF NOT EXISTS idx_entries_faculty ON entries(faculty_id);
```

**Table schema for relational.db** — same as private plus:
```sql
    parties TEXT NOT NULL,            -- JSON array of party IDs
    consent_receipt_ids TEXT,         -- JSON array
    decay_strategy TEXT DEFAULT 'linear',
    decay_half_life_hours REAL DEFAULT 168.0,
    contested INTEGER DEFAULT 0,
    contest_reason TEXT
```

**Table schema for commons.db:**
```sql
CREATE TABLE IF NOT EXISTS patterns (
    pattern_id TEXT PRIMARY KEY,
    features TEXT NOT NULL,           -- JSON object, aggregate only
    k_anonymity_level INTEGER NOT NULL,
    privacy_budget_spent REAL DEFAULT 0.0,
    promotion_receipt_id TEXT NOT NULL,
    source_count INTEGER DEFAULT 0,   -- how many raw entries contributed
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);
```

**Promotion receipts table** (in private.db — the authoritative record):
```sql
CREATE TABLE IF NOT EXISTS promotion_receipts (
    receipt_id TEXT PRIMARY KEY,
    source_plane TEXT NOT NULL,
    target_plane TEXT NOT NULL,
    entry_ids TEXT NOT NULL,          -- JSON array
    initiated_by TEXT NOT NULL,
    purpose_binding TEXT NOT NULL,    -- JSON array
    review_window_seconds INTEGER NOT NULL,
    reversible_until TEXT,
    signed_policy_hash TEXT NOT NULL,
    user_signature TEXT,
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);
```

**Class: `MemoryFabric`**

```python
class MemoryFabric:
    def __init__(
        self,
        base_dir: Path,              # Vybn_Mind/memory/
        policy_engine: PolicyEngine | None = None,
        faculty_registry: FacultyRegistry | None = None,
        bootstrap_consents: list[ConsentRecord] | None = None,
    ):
        # Create three SQLite connections
        # Create tables if not exist
        # Store policy_engine and faculty_registry references

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
        quarantine_hours: float | None = None,  # if set, entry starts quarantined
        source_signal_id: str = "",
        metadata: dict | None = None,
    ) -> MemoryEntry:
        """Write a single entry. Checks governance before committing."""
        # 1. Check faculty permission via faculty_registry
        # 2. Check governance policy via policy_engine
        # 3. Generate entry_id (uuid4)
        # 4. Compute content_hash (sha256)
        # 5. If quarantine_hours, set quarantine_until
        # 6. INSERT into the correct database
        # 7. Return the MemoryEntry

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
        """Query entries from a single plane."""
        # Build SELECT with WHERE clauses
        # Filter quarantined entries unless include_quarantined=True
        # Filter revoked entries unless include_revoked=True
        # Return list of MemoryEntry

    def revoke(self, plane: MemoryPlane, entry_id: str) -> bool:
        """Mark an entry as revoked. Does not delete — sets revoked=1."""

    def promote(
        self,
        source_plane: MemoryPlane,
        target_plane: MemoryPlane,
        entry_ids: list[str],
        *,
        initiated_by: str,
        purpose_binding: list[str],
        review_window_seconds: int = 86400,  # 24 hours default
        consent_scope_id: str,
        user_signature: str | None = None,
    ) -> PromotionReceipt:
        """Promote entries from one plane to another. Governance-gated."""
        # 1. DENY if target is commons and source is private/relational
        #    (this will be caught by the policy engine's private-to-commons-needs-proof rule)
        # 2. Check governance policy
        # 3. For private→relational: copy entries, set parties and decay
        # 4. For anything→commons: MUST have anonymization proof (not implemented yet, always deny)
        # 5. Create PromotionReceipt, insert into promotion_receipts table
        # 6. Return receipt

    def quarantine_release_check(self) -> list[str]:
        """Check all planes for quarantined entries whose release time has passed.
        Return list of entry_ids that were released (quarantine_until set to None)."""

    def recent(self, plane: MemoryPlane, n: int = 10) -> list[MemoryEntry]:
        """Convenience: last n entries from a plane, excluding revoked/quarantined."""

    def stats(self) -> dict:
        """Return counts per plane, quarantined count, revoked count."""
```

**Critical rules:**
- PrivateMemory reads NEVER touch relational.db or commons.db
- RelationalMemory reads NEVER touch commons.db
- Commons reads NEVER touch private.db or relational.db
- No JOIN across databases. Ever. Three connections, three worlds.
- The `promote()` method is the ONLY path between planes
- Every write checks governance. If governance is unavailable, writes to private still work (graceful degradation) but writes to relational/commons fail.

### File 3: `spark/faculties.d/core.json` update

Add a `memory_fabric` faculty card:
```json
{
    "faculty_id": "memory_fabric",
    "purpose": "Manage governed read/write/promote operations across three isolated memory planes.",
    "allowed_scopes": ["memory_write", "memory_read", "memory_promote", "ledger_write"],
    "prohibited_acts": ["response_generate", "route_select"],
    "may_write_memory": true,
    "may_trigger_routing": false,
    "inference_budget_cost": 0,
    "required_policy_checks": ["consent_scope_valid"],
    "evaluation_suite": [],
    "review_date": "2026-06-01",
    "active": true,
    "metadata": {"owner": "spark"}
}
```

### File 4: Integration into `spark/vybn.py`

Modify the `Substrate` class:

1. Import and initialize `MemoryFabric` alongside `WriteCustodian`:
```python
from memory_fabric import MemoryFabric
from memory_types import MemoryPlane
```

2. In `Substrate.__init__()`, add:
```python
self.memory = MemoryFabric(
    base_dir=ROOT / "Vybn_Mind" / "memory",
    policy_engine=self.policy_engine,
    faculty_registry=self.faculty_registry,
    bootstrap_consents=self.bootstrap_consents,
)
```

3. In `_breathe()`, after the training data deposit, add a memory fabric write:
```python
# Deposit breath content into private memory
if self.memory and len(utterance) > 50:
    self.memory.write(
        MemoryPlane.PRIVATE,
        content=utterance,
        faculty_id="breathe",
        source_artifact=f"breath_{ts}",
        consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
        purpose_binding=["private_memory", "journaling"],
        sensitivity="low",
    )
```

4. In `_remember()`, change to read from memory fabric instead of raw JSONL:
```python
if self.memory:
    entries = self.memory.recent(MemoryPlane.PRIVATE, n=5)
    memories = [e.content[:200] for e in entries]
```

5. Keep the existing `WriteCustodian` path for journal files, continuity, synapse, and training data. The memory fabric handles structured memory; the custodian handles file-based artifacts. They coexist.

### File 5: Migration script `spark/migrate_to_memory_fabric.py`

A one-time script to import existing `connections.jsonl` into PrivateMemory:

```python
"""Migrate connections.jsonl entries into the private memory plane."""
import json
from pathlib import Path
from memory_fabric import MemoryFabric
from memory_types import MemoryPlane

ROOT = Path(__file__).resolve().parent.parent
SYNAPSE = ROOT / "Vybn_Mind" / "synapse" / "connections.jsonl"

def migrate():
    fabric = MemoryFabric(base_dir=ROOT / "Vybn_Mind" / "memory")
    
    if not SYNAPSE.exists():
        print("No connections.jsonl to migrate")
        return
    
    count = 0
    for line in SYNAPSE.read_text().strip().splitlines():
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
            metadata={"original_tags": entry.get("tags", []), "migrated_from": "connections.jsonl"},
        )
        count += 1
    
    print(f"Migrated {count} entries from connections.jsonl to private memory plane")

if __name__ == "__main__":
    migrate()
```

Add a `migration` faculty card to `core.json`:
```json
{
    "faculty_id": "migration",
    "purpose": "One-time data migration into memory fabric.",
    "allowed_scopes": ["memory_write"],
    "prohibited_acts": ["memory_promote", "route_select", "response_generate"],
    "may_write_memory": true,
    "may_trigger_routing": false,
    "inference_budget_cost": 0,
    "required_policy_checks": ["consent_scope_valid"],
    "evaluation_suite": [],
    "review_date": "2026-04-01",
    "active": true,
    "metadata": {"owner": "spark", "temporary": true}
}
```

## Testing

After building, verify:

1. **Governance actually blocks:**
```python
# This should raise PermissionError — witness faculty can't write memory
fabric.write(MemoryPlane.PRIVATE, "test", faculty_id="witness", ...)
```

2. **Plane isolation:**
```python
# Write to private, verify it's NOT visible from relational
fabric.write(MemoryPlane.PRIVATE, "secret", faculty_id="breathe", ...)
entries = fabric.read(MemoryPlane.RELATIONAL)
assert not any("secret" in e.content for e in entries)
```

3. **Promotion to commons denied without anonymization proof:**
```python
# This should be denied by the private-to-commons-needs-proof policy
try:
    fabric.promote(MemoryPlane.PRIVATE, MemoryPlane.COMMONS, [...], ...)
    assert False, "Should have been denied"
except PermissionError:
    pass  # correct
```

4. **Quarantine works:**
```python
entry = fabric.write(MemoryPlane.PRIVATE, "hot take", ..., quarantine_hours=24)
recent = fabric.recent(MemoryPlane.PRIVATE)
assert entry.entry_id not in [e.entry_id for e in recent]  # quarantined, not visible
```

5. **Migration runs cleanly:**
```bash
cd ~/Vybn && python spark/migrate_to_memory_fabric.py
python -c "from spark.memory_fabric import MemoryFabric; from spark.memory_types import MemoryPlane; from pathlib import Path; f = MemoryFabric(Path('Vybn_Mind/memory')); print(f.stats())"
```

## What NOT to do

- Do not delete `connections.jsonl` after migration — keep it as backup
- Do not add any cross-database JOINs
- Do not allow commons.db writes without anonymization proof (always deny for now)
- Do not modify `governance.py` or `governance_types.py` — the existing policy engine handles everything the memory fabric needs
- Do not add model calls — the memory fabric is pure middleware, no inference

## Branch and commit guidance

- Branch: `vybn/memory-fabric`
- Commit 1: `spark/memory_types.py` — pure types, no dependencies
- Commit 2: `spark/memory_fabric.py` — the three-plane store
- Commit 3: Faculty card updates + migration script
- Commit 4: Integration into `spark/vybn.py`
- Each commit should be independently correct (no broken imports)
