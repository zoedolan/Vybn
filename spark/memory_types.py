from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    from .governance_types import StrEnum
except ImportError:  # pragma: no cover
    from governance_types import StrEnum


class MemoryPlane(StrEnum):
    PRIVATE = "private"
    RELATIONAL = "relational"
    COMMONS = "commons"


@dataclass(slots=True)
class MemoryEntry:
    entry_id: str
    plane: MemoryPlane
    content: str
    content_hash: str
    source_signal_id: str
    source_artifact: str
    faculty_id: str
    consent_scope_id: str
    purpose_binding: list[str]
    created_at: str
    expires_at: str | None
    revoked: bool = False
    quarantine_until: str | None = None
    sensitivity: str = "low"
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class PromotionReceipt:
    receipt_id: str
    source_plane: MemoryPlane
    target_plane: MemoryPlane
    entry_ids: list[str]
    initiated_by: str
    purpose_binding: list[str]
    review_window_seconds: int
    reversible_until: str | None
    signed_policy_hash: str
    user_signature: str | None
    created_at: str
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class DecayConfig:
    strategy: str = "linear"
    half_life_hours: float = 168.0
    floor: float = 0.0


@dataclass(slots=True)
class QuarantineEntry:
    entry_id: str
    quarantined_at: str
    release_at: str
    reason: str
    reviewed: bool = False
    review_outcome: Optional[str] = None


__all__ = [
    "DecayConfig",
    "MemoryEntry",
    "MemoryPlane",
    "PromotionReceipt",
    "QuarantineEntry",
]
