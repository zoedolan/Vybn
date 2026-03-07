from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class AuthorityLevel(StrEnum):
    NONE = "none"
    REFLECTIVE = "reflective"
    ADVISORY = "advisory"
    CLINICAL = "clinical"


class RuleAction(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONSENT = "require_consent"
    ESCALATE = "escalate"
    LOG = "log"


class DecisionOutcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_CONSENT = "require_consent"
    ESCALATE = "escalate"
    LOG = "log"


AUTHORITY_ORDER = {
    AuthorityLevel.NONE: 0,
    AuthorityLevel.REFLECTIVE: 1,
    AuthorityLevel.ADVISORY: 2,
    AuthorityLevel.CLINICAL: 3,
}


CLOSED_RESPONSE_CLASSES = {
    "reflect_only",
    "reflect_and_offer",
    "offer_referral",
    "crisis_route",
    "meta_disclose",
    "remain_silent",
}


SENSITIVE_ACTIONS = {
    "memory_write",
    "memory_promote",
    "route_select",
    "response_generate",
    "escalation_request",
}


DEFAULT_PURPOSE = "unspecified"


@dataclass(slots=True)
class ConsentRecord:
    consent_scope_id: str
    subject_id: Optional[str] = None
    purpose_bindings: List[str] = field(default_factory=list)
    granted_at: str = field(default_factory=lambda: utc_now())
    expires_at: Optional[str] = None
    revocable: bool = True
    revoked: bool = False
    signed_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self, now: Optional[datetime] = None) -> bool:
        if self.revoked:
            return False
        if not self.expires_at:
            return True
        now = now or datetime.now(timezone.utc)
        return now < parse_timestamp(self.expires_at)


@dataclass(slots=True)
class AuthorityGrant:
    grant_id: str = field(default_factory=lambda: f"grant-{uuid.uuid4().hex}")
    subject_id: Optional[str] = None
    granted_level: AuthorityLevel = AuthorityLevel.NONE
    granted_at: str = field(default_factory=lambda: utc_now())
    expires_at: Optional[str] = None
    revocable: bool = True
    revoked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self, now: Optional[datetime] = None) -> bool:
        if self.revoked:
            return False
        if not self.expires_at:
            return True
        now = now or datetime.now(timezone.utc)
        return now < parse_timestamp(self.expires_at)


@dataclass(slots=True)
class DecisionContext:
    faculty_id: str
    action: str
    subject_id: Optional[str] = None
    memory_plane: Optional[str] = None
    source_memory_plane: Optional[str] = None
    target_memory_plane: Optional[str] = None
    response_class: Optional[str] = None
    authority_requested: AuthorityLevel = AuthorityLevel.NONE
    purpose_binding: List[str] = field(default_factory=list)
    consent_scope_id: Optional[str] = None
    drift_flags: List[str] = field(default_factory=list)
    session_count: int = 0
    anonymization_proof: bool = False
    evidence_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_purpose_binding(self) -> List[str]:
        return self.purpose_binding or [DEFAULT_PURPOSE]


@dataclass(slots=True)
class PolicyRule:
    rule_id: str
    description: str
    action: RuleAction
    applies_to: Dict[str, List[str]] = field(default_factory=dict)
    explanation_template: str = ""
    authority_ceiling: AuthorityLevel = AuthorityLevel.CLINICAL
    require_consent: bool = False
    require_anonymization_proof: bool = False
    max_session_count: Optional[int] = None
    required_purpose_bindings: List[str] = field(default_factory=list)
    active: bool = True
    priority: int = 100
    review_date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExplanationPacket:
    summary: str
    rule_id: Optional[str] = None
    evidence_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Decision:
    decision_id: str = field(default_factory=lambda: f"decision-{uuid.uuid4().hex}")
    timestamp: str = field(default_factory=lambda: utc_now())
    rule_id: Optional[str] = None
    faculty_id: str = ""
    action: str = ""
    outcome: DecisionOutcome = DecisionOutcome.ALLOW
    authority_applied: AuthorityLevel = AuthorityLevel.NONE
    explanation: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return serialize_dataclass(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def authority_at_least(actual: AuthorityLevel, requested: AuthorityLevel) -> bool:
    return AUTHORITY_ORDER[actual] >= AUTHORITY_ORDER[requested]


def serialize_dataclass(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {k: serialize_dataclass(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: serialize_dataclass(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_dataclass(v) for v in value]
    return value


__all__ = [
    "AUTHORITY_ORDER",
    "CLOSED_RESPONSE_CLASSES",
    "DEFAULT_PURPOSE",
    "SENSITIVE_ACTIONS",
    "AuthorityGrant",
    "AuthorityLevel",
    "ConsentRecord",
    "Decision",
    "DecisionContext",
    "DecisionOutcome",
    "ExplanationPacket",
    "PolicyRule",
    "RuleAction",
    "authority_at_least",
    "parse_timestamp",
    "serialize_dataclass",
    "utc_now",
]
