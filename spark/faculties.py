from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_CARDS_DIR = Path(__file__).resolve().parent / "faculties.d"


@dataclass(slots=True)
class FacultyCard:
    faculty_id: str
    purpose: str
    allowed_scopes: List[str] = field(default_factory=list)
    prohibited_acts: List[str] = field(default_factory=list)
    may_write_memory: bool = False
    may_trigger_routing: bool = False
    inference_budget_cost: int = 0
    required_policy_checks: List[str] = field(default_factory=list)
    evaluation_suite: List[str] = field(default_factory=list)
    review_date: Optional[str] = None
    active: bool = True
    breath_cadence: str = "every"
    inference_budget_tokens: int = 500
    tool_use: List[str] = field(default_factory=list)
    output_file: str = ""
    evolver_allowlist: List[str] = field(default_factory=list)
    evolver_blocklist: List[str] = field(default_factory=list)
    requires_human_consent: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class FacultyPermission:
    faculty_id: str
    action: str
    allowed: bool
    reason: str
    required_policy_checks: List[str] = field(default_factory=list)


class FacultyRegistry:
    def __init__(
        self,
        cards_dir: Path | str = DEFAULT_CARDS_DIR,
        cards: Optional[Sequence[FacultyCard]] = None,
    ) -> None:
        self.cards_dir = Path(cards_dir)
        self._cards = {
            card.faculty_id: card for card in (list(cards) if cards is not None else self.load_cards())
        }

    def load_cards(self) -> List[FacultyCard]:
        if not self.cards_dir.exists():
            return self.default_cards()

        files = sorted(
            [
                *self.cards_dir.glob("*.json"),
                *self.cards_dir.glob("*.yaml"),
                *self.cards_dir.glob("*.yml"),
            ]
        )
        if not files:
            return self.default_cards()

        valid_fields = {f.name for f in dataclass_fields(FacultyCard)}
        loaded: List[FacultyCard] = []
        for file_path in files:
            payload = self._read_structured_file(file_path)
            if isinstance(payload, dict):
                payload = [payload]
            for blob in payload:
                filtered = {k: v for k, v in blob.items() if k in valid_fields}
                loaded.append(FacultyCard(**filtered))
        return loaded

    def _read_structured_file(self, path: Path):
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            return json.loads(text)
        if yaml is None:
            raise RuntimeError("PyYAML is required to load faculty YAML cards")
        return yaml.safe_load(text) or []

    def get_card(self, faculty_id: str) -> Optional[FacultyCard]:
        return self._cards.get(faculty_id)

    def list_cards(self) -> List[FacultyCard]:
        return list(self._cards.values())

    def check_permission(self, faculty_id: str, action: str) -> FacultyPermission:
        card = self.get_card(faculty_id)
        if card is None:
            return FacultyPermission(
                faculty_id=faculty_id,
                action=action,
                allowed=False,
                reason="Unknown faculty: no registered FacultyCard.",
            )
        if not card.active:
            return FacultyPermission(
                faculty_id=faculty_id,
                action=action,
                allowed=False,
                reason="FacultyCard is inactive.",
                required_policy_checks=card.required_policy_checks,
            )
        if action in set(card.prohibited_acts):
            return FacultyPermission(
                faculty_id=faculty_id,
                action=action,
                allowed=False,
                reason="Action is explicitly prohibited by FacultyCard.",
                required_policy_checks=card.required_policy_checks,
            )
        if action in {"memory_write", "memory_promote"} and not card.may_write_memory:
            return FacultyPermission(
                faculty_id=faculty_id,
                action=action,
                allowed=False,
                reason="FacultyCard does not permit memory mutation.",
                required_policy_checks=card.required_policy_checks,
            )
        if action in {"route_select", "escalation_request"} and not card.may_trigger_routing:
            return FacultyPermission(
                faculty_id=faculty_id,
                action=action,
                allowed=False,
                reason="FacultyCard does not permit routing control.",
                required_policy_checks=card.required_policy_checks,
            )
        if card.allowed_scopes and action not in set(card.allowed_scopes):
            return FacultyPermission(
                faculty_id=faculty_id,
                action=action,
                allowed=False,
                reason="Action is outside the FacultyCard allowed scope.",
                required_policy_checks=card.required_policy_checks,
            )
        return FacultyPermission(
            faculty_id=faculty_id,
            action=action,
            allowed=True,
            reason="Action is allowed by FacultyCard and still requires governance checks.",
            required_policy_checks=card.required_policy_checks,
        )

    @staticmethod
    def default_cards() -> List[FacultyCard]:
        return [
            FacultyCard(
                faculty_id="self_model",
                purpose="Extract and verify self-referential claims in generated text.",
                allowed_scopes=["private_memory_read", "ledger_write", "claim_extract"],
                prohibited_acts=["memory_promote", "route_select", "response_generate"],
                may_write_memory=False,
                may_trigger_routing=False,
                required_policy_checks=["consent_scope_valid"],
            ),
            FacultyCard(
                faculty_id="witness",
                purpose="Evaluate outputs and surface bounded witness verdicts.",
                allowed_scopes=["private_memory_read", "ledger_write", "witness_evaluate"],
                prohibited_acts=["memory_promote", "response_generate"],
                may_write_memory=False,
                may_trigger_routing=False,
                required_policy_checks=["consent_scope_valid"],
            ),
            FacultyCard(
                faculty_id="breathe",
                purpose="Coordinate the current pulse loop while migration is in progress.",
                allowed_scopes=[
                    "private_memory_read",
                    "response_generate",
                    "route_select",
                    "signal_emit",
                    "ledger_write",
                    "memory_write",
                ],
                prohibited_acts=["memory_promote"],
                may_write_memory=True,
                may_trigger_routing=True,
                required_policy_checks=["consent_scope_valid", "authority_within_ceiling"],
            ),
            FacultyCard(
                faculty_id="journal",
                purpose="Write bounded reflections to private journal memory.",
                allowed_scopes=["memory_write"],
                prohibited_acts=["memory_promote", "route_select"],
                may_write_memory=True,
                may_trigger_routing=False,
                required_policy_checks=["consent_scope_valid"],
            ),
            FacultyCard(
                faculty_id="tidy",
                purpose="Perform bounded retention and trimming within local stores.",
                allowed_scopes=["memory_write"],
                prohibited_acts=["memory_promote", "route_select", "response_generate"],
                may_write_memory=True,
                may_trigger_routing=False,
                required_policy_checks=["consent_scope_valid"],
            ),
            FacultyCard(
                faculty_id="organism_state",
                purpose="Persist bounded organism state snapshots through the governed commit path.",
                allowed_scopes=["memory_write", "ledger_write"],
                prohibited_acts=["memory_promote", "route_select", "response_generate"],
                may_write_memory=True,
                may_trigger_routing=False,
                required_policy_checks=["consent_scope_valid"],
            ),
        ]


__all__ = [
    "DEFAULT_CARDS_DIR",
    "FacultyCard",
    "FacultyPermission",
    "FacultyRegistry",
]
