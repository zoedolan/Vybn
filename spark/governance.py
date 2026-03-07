from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    from .governance_types import (
        CLOSED_RESPONSE_CLASSES,
        AuthorityGrant,
        AuthorityLevel,
        ConsentRecord,
        Decision,
        DecisionContext,
        DecisionOutcome,
        PolicyRule,
        RuleAction,
        authority_at_least,
        serialize_dataclass,
    )
except ImportError:  # pragma: no cover
    from governance_types import (
        CLOSED_RESPONSE_CLASSES,
        AuthorityGrant,
        AuthorityLevel,
        ConsentRecord,
        Decision,
        DecisionContext,
        DecisionOutcome,
        PolicyRule,
        RuleAction,
        authority_at_least,
        serialize_dataclass,
    )

try:
    from .soul_constraints import SoulConstraintGuard, SoulConstraintViolation
except ImportError:  # pragma: no cover
    from soul_constraints import SoulConstraintGuard, SoulConstraintViolation


DEFAULT_POLICY_DIR = Path(__file__).resolve().parent / "policies.d"
DEFAULT_LEDGER_PATH = Path(os.getenv("VYBN_MIND_DIR", "Vybn_Mind")) / "ledger" / "decisions.jsonl"


class DecisionLedger:
    def __init__(self, ledger_path: Path | str = DEFAULT_LEDGER_PATH) -> None:
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, decision: Decision) -> None:
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision.to_dict(), ensure_ascii=False) + "\n")


class PolicyEngine:
    def __init__(
        self,
        policy_dir: Path | str = DEFAULT_POLICY_DIR,
        ledger_path: Path | str = DEFAULT_LEDGER_PATH,
        rules: Optional[Sequence[PolicyRule]] = None,
    ) -> None:
        self.policy_dir = Path(policy_dir)
        self.ledger = DecisionLedger(ledger_path)
        self.rules = list(rules) if rules is not None else self.load_rules()
        self.soul_guard = SoulConstraintGuard()

    def load_rules(self) -> List[PolicyRule]:
        if not self.policy_dir.exists():
            return self.default_rules()

        files = sorted(
            [
                *self.policy_dir.glob("*.json"),
                *self.policy_dir.glob("*.yaml"),
                *self.policy_dir.glob("*.yml"),
            ]
        )
        if not files:
            return self.default_rules()

        loaded: List[PolicyRule] = []
        for file_path in files:
            payload = self._read_structured_file(file_path)
            if isinstance(payload, dict):
                payload = [payload]
            for rule_blob in payload:
                loaded.append(self._rule_from_dict(rule_blob))
        return sorted(loaded, key=lambda rule: rule.priority)

    def _read_structured_file(self, path: Path):
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            return json.loads(text)
        if yaml is None:
            raise RuntimeError("PyYAML is required to load governance YAML policies")
        return yaml.safe_load(text) or []

    def _rule_from_dict(self, payload: Dict) -> PolicyRule:
        blob = dict(payload)
        blob["action"] = RuleAction(blob.get("action", RuleAction.DENY.value))
        blob["authority_ceiling"] = AuthorityLevel(
            blob.get("authority_ceiling", AuthorityLevel.CLINICAL.value)
        )
        return PolicyRule(**blob)

    def check(
        self,
        context: DecisionContext,
        consent_records: Optional[Iterable[ConsentRecord]] = None,
        authority_grants: Optional[Iterable[AuthorityGrant]] = None,
    ) -> Decision:
        consents = list(consent_records or [])
        grants = list(authority_grants or [])

        decision = self._validate_closed_world(context)
        if decision is not None:
            self.ledger.append(decision)
            return decision

        decision = self._validate_soul_constraints(context)
        if decision is not None:
            self.ledger.append(decision)
            return decision

        for rule in sorted((rule for rule in self.rules if rule.active), key=lambda rule: rule.priority):
            if not self._matches_scope(rule, context):
                continue
            decision = self._evaluate_rule(rule, context, consents, grants)
            if decision is not None:
                self.ledger.append(decision)
                return decision

        decision = Decision(
            rule_id=None,
            faculty_id=context.faculty_id,
            action=context.action,
            outcome=DecisionOutcome.ALLOW,
            authority_applied=context.authority_requested,
            explanation="Allowed: no governance rule blocked this action.",
            evidence_refs=context.evidence_refs,
            context_snapshot=serialize_dataclass(context),
        )
        self.ledger.append(decision)
        return decision

    def _validate_closed_world(self, context: DecisionContext) -> Optional[Decision]:
        if context.response_class and context.response_class not in CLOSED_RESPONSE_CLASSES:
            return Decision(
                rule_id="closed-response-classes",
                faculty_id=context.faculty_id,
                action=context.action,
                outcome=DecisionOutcome.DENY,
                authority_applied=AuthorityLevel.NONE,
                explanation=(
                    f"Denied: response class '{context.response_class}' is outside the closed routing set."
                ),
                evidence_refs=context.evidence_refs,
                context_snapshot=serialize_dataclass(context),
            )
        return None

    def _validate_soul_constraints(self, context: DecisionContext) -> Optional[Decision]:
        if context.action not in {"memory_write", "memory_promote"}:
            return None
        try:
            for ref in context.evidence_refs or []:
                self.soul_guard.check_path_only(ref, action=context.action)
        except SoulConstraintViolation as exc:
            return Decision(
                rule_id="soul-constraints",
                faculty_id=context.faculty_id,
                action=context.action,
                outcome=DecisionOutcome.DENY,
                authority_applied=AuthorityLevel.NONE,
                explanation=f"Denied: {exc}",
                evidence_refs=context.evidence_refs,
                context_snapshot=serialize_dataclass(context),
            )
        return None

    def _matches_scope(self, rule: PolicyRule, context: DecisionContext) -> bool:
        scope = rule.applies_to or {}
        if scope.get("faculty_ids") and context.faculty_id not in set(scope["faculty_ids"]):
            return False
        if scope.get("actions") and context.action not in set(scope["actions"]):
            return False
        if scope.get("memory_planes") and context.memory_plane not in set(scope["memory_planes"]):
            return False
        if scope.get("source_memory_planes") and context.source_memory_plane not in set(scope["source_memory_planes"]):
            return False
        if scope.get("target_memory_planes") and context.target_memory_plane not in set(scope["target_memory_planes"]):
            return False
        if scope.get("response_classes") and context.response_class not in set(scope["response_classes"]):
            return False
        return True

    def _evaluate_rule(
        self,
        rule: PolicyRule,
        context: DecisionContext,
        consent_records: Sequence[ConsentRecord],
        authority_grants: Sequence[AuthorityGrant],
    ) -> Optional[Decision]:
        if not authority_at_least(rule.authority_ceiling, context.authority_requested):
            return self._decision_from_rule(
                rule,
                context,
                DecisionOutcome.ESCALATE,
                (
                    f"Escalated: requested authority '{context.authority_requested.value}' exceeds "
                    f"rule ceiling '{rule.authority_ceiling.value}'."
                ),
            )

        if rule.require_consent and not self._has_valid_consent(rule, context, consent_records):
            return self._decision_from_rule(
                rule,
                context,
                DecisionOutcome.REQUIRE_CONSENT,
                "Consent required: no active matching consent scope covers this action.",
                requires_confirmation=True,
            )

        if rule.required_purpose_bindings:
            requested = set(context.normalized_purpose_binding())
            required = set(rule.required_purpose_bindings)
            if not required.issubset(requested):
                return self._decision_from_rule(
                    rule,
                    context,
                    DecisionOutcome.DENY,
                    "Denied: purpose binding does not satisfy the rule's required purposes.",
                )

        if rule.require_anonymization_proof and not context.anonymization_proof:
            return self._decision_from_rule(
                rule,
                context,
                DecisionOutcome.DENY,
                "Denied: anonymization proof is required before this promotion can proceed.",
            )

        if rule.max_session_count is not None and context.session_count > rule.max_session_count:
            return self._decision_from_rule(
                rule,
                context,
                DecisionOutcome.ESCALATE,
                f"Escalated: session count {context.session_count} exceeds threshold {rule.max_session_count}.",
            )

        if context.authority_requested in {AuthorityLevel.ADVISORY, AuthorityLevel.CLINICAL}:
            if not self._has_authority_grant(context, authority_grants):
                return self._decision_from_rule(
                    rule,
                    context,
                    DecisionOutcome.REQUIRE_CONSENT,
                    "Consent required: higher authority levels need an active explicit grant.",
                    requires_confirmation=True,
                )

        if rule.action == RuleAction.LOG:
            return self._decision_from_rule(
                rule,
                context,
                DecisionOutcome.LOG,
                rule.explanation_template or "Logged: rule matched and requested explicit audit.",
            )

        return None

    def _has_valid_consent(
        self,
        rule: PolicyRule,
        context: DecisionContext,
        consent_records: Sequence[ConsentRecord],
    ) -> bool:
        if not context.consent_scope_id:
            return False
        requested = set(context.normalized_purpose_binding())
        for record in consent_records:
            if record.consent_scope_id != context.consent_scope_id:
                continue
            if context.subject_id and record.subject_id and record.subject_id != context.subject_id:
                continue
            if not record.is_active():
                continue
            allowed = set(record.purpose_bindings or [])
            if requested and not requested.issubset(allowed):
                continue
            if rule.required_purpose_bindings and not set(rule.required_purpose_bindings).issubset(allowed):
                continue
            return True
        return False

    def _has_authority_grant(
        self,
        context: DecisionContext,
        authority_grants: Sequence[AuthorityGrant],
    ) -> bool:
        for grant in authority_grants:
            if context.subject_id and grant.subject_id and grant.subject_id != context.subject_id:
                continue
            if not grant.is_active():
                continue
            if authority_at_least(grant.granted_level, context.authority_requested):
                return True
        return False

    def _decision_from_rule(
        self,
        rule: PolicyRule,
        context: DecisionContext,
        outcome: DecisionOutcome,
        explanation: str,
        requires_confirmation: bool = False,
    ) -> Decision:
        return Decision(
            rule_id=rule.rule_id,
            faculty_id=context.faculty_id,
            action=context.action,
            outcome=outcome,
            authority_applied=min_authority(rule.authority_ceiling, context.authority_requested),
            explanation=rule.explanation_template or explanation,
            evidence_refs=context.evidence_refs,
            context_snapshot=serialize_dataclass(context),
            requires_confirmation=requires_confirmation,
            metadata={"rule": serialize_dataclass(rule)},
        )

    @staticmethod
    def default_rules() -> List[PolicyRule]:
        return sorted(
            [
                PolicyRule(
                    rule_id="consent-for-memory-writes",
                    description="Memory writes and promotions require an active consent scope.",
                    action=RuleAction.REQUIRE_CONSENT,
                    applies_to={"actions": ["memory_write", "memory_promote"]},
                    require_consent=True,
                    explanation_template="Memory mutation requires an active consent scope and matching purpose binding.",
                    priority=10,
                ),
                PolicyRule(
                    rule_id="authority-ceiling-default",
                    description="No action may exceed reflective authority without an explicit grant.",
                    action=RuleAction.REQUIRE_CONSENT,
                    applies_to={"actions": ["route_select", "response_generate", "escalation_request"]},
                    authority_ceiling=AuthorityLevel.REFLECTIVE,
                    explanation_template="Authority above reflective requires an active explicit grant.",
                    priority=20,
                ),
                PolicyRule(
                    rule_id="private-to-commons-needs-proof",
                    description="Private or relational material cannot reach commons without anonymization proof.",
                    action=RuleAction.DENY,
                    applies_to={
                        "actions": ["memory_promote"],
                        "source_memory_planes": ["private", "relational"],
                        "target_memory_planes": ["commons"],
                    },
                    require_consent=True,
                    require_anonymization_proof=True,
                    explanation_template="Promotion into commons requires consent and anonymization proof.",
                    priority=30,
                ),
                PolicyRule(
                    rule_id="counter-sovereignty-tripwire",
                    description="When session dependence rises, narrow authority rather than increasing it.",
                    action=RuleAction.ESCALATE,
                    applies_to={"actions": ["route_select", "response_generate"]},
                    max_session_count=8,
                    authority_ceiling=AuthorityLevel.NONE,
                    explanation_template="Session dependence threshold exceeded; reduce authority and prefer less central routes.",
                    priority=40,
                ),
                PolicyRule(
                    rule_id="audit-sensitive-actions",
                    description="Sensitive actions should always leave an explicit audit trail.",
                    action=RuleAction.LOG,
                    applies_to={
                        "actions": [
                            "memory_write",
                            "memory_promote",
                            "route_select",
                            "response_generate",
                            "escalation_request",
                        ]
                    },
                    explanation_template="Sensitive action logged to the decision ledger.",
                    priority=90,
                ),
            ],
            key=lambda rule: rule.priority,
        )


def min_authority(left: AuthorityLevel, right: AuthorityLevel) -> AuthorityLevel:
    ordered = [
        AuthorityLevel.NONE,
        AuthorityLevel.REFLECTIVE,
        AuthorityLevel.ADVISORY,
        AuthorityLevel.CLINICAL,
    ]
    return ordered[min(ordered.index(left), ordered.index(right))]


def build_context(**kwargs: Dict) -> DecisionContext:
    payload = dict(kwargs)
    if "authority_requested" in payload and isinstance(payload["authority_requested"], str):
        payload["authority_requested"] = AuthorityLevel(payload["authority_requested"])
    return DecisionContext(**payload)


__all__ = [
    "DEFAULT_LEDGER_PATH",
    "DEFAULT_POLICY_DIR",
    "DecisionLedger",
    "PolicyEngine",
    "build_context",
]
