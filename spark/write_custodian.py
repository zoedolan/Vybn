from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from spark.paths import MIND_PREFIX

try:
    from .governance import PolicyEngine, build_context
    from .governance_types import ConsentRecord, DecisionOutcome
    from .faculties import FacultyRegistry
    from .soul_constraints import SoulConstraintGuard
except ImportError:  # pragma: no cover
    from governance import PolicyEngine, build_context
    from governance_types import ConsentRecord, DecisionOutcome
    from faculties import FacultyRegistry
    from soul_constraints import SoulConstraintGuard


@dataclass(slots=True)
class WriteIntent:
    intent_id: str
    requested_at: str
    faculty_id: str
    path: str
    mode: str
    purpose_binding: list[str] = field(default_factory=list)
    consent_scope_id: str = ""
    content_sha256: str = ""
    content_length: int = 0
    metadata: dict = field(default_factory=dict)


class WriteCustodian:
    """Single governed commit path for durable local writes.

    Callers may propose text mutations, but this class is the only code path
    that should actually commit them to disk. It applies soul constraints,
    faculty permissions, governance checks, and an append-only intent ledger
    before writing.
    """

    def __init__(
        self,
        *,
        repo_root: Path | str,
        ledger_path: Path | str,
        soul_path: Path | str,
        policy_engine: Optional[PolicyEngine] = None,
        faculty_registry: Optional[FacultyRegistry] = None,
        bootstrap_consents: Optional[Iterable[ConsentRecord]] = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.ledger_path = Path(ledger_path)
        self.soul_guard = SoulConstraintGuard(repo_root=self.repo_root, soul_path=soul_path)
        self.policy_engine = policy_engine
        self.faculty_registry = faculty_registry
        self.bootstrap_consents = list(bootstrap_consents or [])

    def write(
        self,
        path: str | Path,
        data: str,
        *,
        faculty_id: str,
        purpose_binding: Optional[list[str]] = None,
        consent_scope_id: str,
        metadata: Optional[dict] = None,
    ) -> Path:
        return self._commit(
            path=path,
            data=data,
            faculty_id=faculty_id,
            mode="write",
            purpose_binding=purpose_binding,
            consent_scope_id=consent_scope_id,
            metadata=metadata,
        )

    def append(
        self,
        path: str | Path,
        data: str,
        *,
        faculty_id: str,
        purpose_binding: Optional[list[str]] = None,
        consent_scope_id: str,
        metadata: Optional[dict] = None,
    ) -> Path:
        return self._commit(
            path=path,
            data=data,
            faculty_id=faculty_id,
            mode="append",
            purpose_binding=purpose_binding,
            consent_scope_id=consent_scope_id,
            metadata=metadata,
        )

    def _commit(
        self,
        *,
        path: str | Path,
        data: str,
        faculty_id: str,
        mode: str,
        purpose_binding: Optional[list[str]],
        consent_scope_id: str,
        metadata: Optional[dict],
    ) -> Path:
        target = self._resolve(path)
        self._check_permission(
            target=target,
            faculty_id=faculty_id,
            purpose_binding=purpose_binding or ["system_operation"],
            consent_scope_id=consent_scope_id,
        )

        intent = WriteIntent(
            intent_id=self._intent_id(target, faculty_id, mode),
            requested_at=datetime.now(timezone.utc).isoformat(),
            faculty_id=faculty_id,
            path=str(self._relative(target)),
            mode=mode,
            purpose_binding=list(purpose_binding or ["system_operation"]),
            consent_scope_id=consent_scope_id,
            content_sha256=self._sha256(data),
            content_length=len(data),
            metadata=metadata or {},
        )
        self._log_intent(intent)

        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with target.open("a", encoding="utf-8") as handle:
                handle.write(data)
        else:
            target.write_text(data, encoding="utf-8")
        return target

    def _check_permission(
        self,
        *,
        target: Path,
        faculty_id: str,
        purpose_binding: list[str],
        consent_scope_id: str,
    ) -> None:
        self.soul_guard.check_file_write(target, "")

        if self.faculty_registry is not None:
            permission = self.faculty_registry.check_permission(faculty_id, "memory_write")
            if not permission.allowed:
                raise PermissionError(permission.reason)

        if self.policy_engine is not None:
            context = build_context(
                faculty_id=faculty_id,
                action="memory_write",
                memory_plane=self._infer_memory_plane(target),
                purpose_binding=purpose_binding,
                consent_scope_id=consent_scope_id,
                evidence_refs=[str(self._relative(target))],
            )
            decision = self.policy_engine.check(
                context,
                consent_records=self.bootstrap_consents,
            )
            if decision.outcome not in {DecisionOutcome.ALLOW, DecisionOutcome.LOG}:
                raise PermissionError(decision.explanation)

    def _infer_memory_plane(self, path: Path) -> Optional[str]:
        normalized = str(self._relative(path)).replace("\\", "/")
        if normalized.startswith(MIND_PREFIX):
            return "private"
        return None

    def _log_intent(self, intent: WriteIntent) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(intent), ensure_ascii=False) + "\n")

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        return candidate.resolve()

    def _relative(self, path: Path) -> Path:
        try:
            return path.relative_to(self.repo_root)
        except ValueError:
            return path

    def _intent_id(self, target: Path, faculty_id: str, mode: str) -> str:
        seed = f"{faculty_id}:{mode}:{target}:{datetime.now(timezone.utc).timestamp()}"
        return self._sha256(seed)[:24]

    @staticmethod
    def _sha256(data: str) -> str:
        import hashlib
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


__all__ = ["WriteCustodian", "WriteIntent"]
