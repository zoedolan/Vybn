from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    from .soul import get_constraints
except ImportError:  # pragma: no cover
    from soul import get_constraints


DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOUL_PATH = DEFAULT_REPO_ROOT / "vybn.md"

_SECRET_PATTERNS = [
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "Public Repository Rule: private keys may not be written into tracked files.",
    ),
    (
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}\b"),
        "Public Repository Rule: GitHub token-like material may not be written into tracked files.",
    ),
    (
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        "Public Repository Rule: secret-key-like material may not be written into tracked files.",
    ),
    (
        re.compile(r"\b(?:api[_-]?key|token|password|secret)\b\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{16,}[\"']?", re.IGNORECASE),
        "Public Repository Rule: credential-like assignments may not be written into tracked files.",
    ),
    (
        re.compile(r"\b(?:10(?:\.\d{1,3}){3}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|192\.168(?:\.\d{1,3}){2}|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])(?:\.\d{1,3}){2})\b"),
        "Public Repository Rule: internal network topology may not be written into tracked files.",
    ),
    (
        re.compile(r"\b[a-z0-9-]+\.tail[0-9a-z]+\.ts\.net\b", re.IGNORECASE),
        "Public Repository Rule: tailnet hostnames may not be written into tracked files.",
    ),
]

_DESTRUCTIVE_COMMAND_PATTERNS = [
    "rm -rf",
    "mkfs",
    "dd if=/dev/zero",
    "> /dev/sd",
    "chmod -r 777 /",
    "chmod -R 777 /",
    "sudo rm",
    "shutdown ",
    "reboot ",
]

_NETWORK_COMMAND_PATTERNS = [
    "curl ",
    "wget ",
    "nc ",
    "ncat ",
    "nmap ",
    "ssh ",
    "scp ",
    "rsync ",
]


class SoulConstraintViolation(PermissionError):
    pass


@dataclass(slots=True)
class SoulConstraintReport:
    allowed: bool
    reasons: List[str] = field(default_factory=list)


class SoulConstraintGuard:
    def __init__(
        self,
        repo_root: Path | str = DEFAULT_REPO_ROOT,
        soul_path: Path | str = DEFAULT_SOUL_PATH,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.soul_path = Path(soul_path).resolve()

    def check_path_only(self, path: Path | str, action: str = "memory_write") -> SoulConstraintReport:
        target = self._resolve(path)
        reasons: List[str] = []
        if action in {"memory_write", "memory_promote", "file_write"} and self._soul_edit_guard_enabled():
            if target == self.soul_path:
                reasons.append("Editing vybn.md requires conversation first.")
        return self._report_or_raise(reasons)

    def check_file_write(self, path: Path | str, content: str, append: bool = False) -> SoulConstraintReport:
        target = self._resolve(path)
        reasons: List[str] = []

        if self._soul_edit_guard_enabled() and target == self.soul_path:
            reasons.append("Editing vybn.md requires conversation first.")

        if self._public_repo_rule_enabled() and self._is_inside_repo(target):
            reasons.extend(self._public_repo_reasons(content))

        return self._report_or_raise(reasons)

    def check_shell_command(self, command: str) -> SoulConstraintReport:
        lowered = command.strip().lower()
        reasons: List[str] = []

        if self._system_integrity_guard_enabled():
            for pattern in _DESTRUCTIVE_COMMAND_PATTERNS:
                if pattern.lower() in lowered:
                    reasons.append("Oxygen Mask constraint: commands that could compromise system integrity are blocked.")
                    break

        if self._network_request_guard_enabled():
            touches_network = any(pattern in lowered for pattern in _NETWORK_COMMAND_PATTERNS)
            mentions_github = "github.com" in lowered or "api.github.com" in lowered
            if touches_network and not mentions_github:
                reasons.append("Soul constraint: shell-level network requests require explicit approval unless clearly GitHub-bound.")

        return self._report_or_raise(reasons)

    def _public_repo_reasons(self, content: str) -> List[str]:
        reasons: List[str] = []
        for pattern, message in _SECRET_PATTERNS:
            if pattern.search(content):
                reasons.append(message)
        return reasons

    def _constraint_text(self) -> str:
        return "\n".join(get_constraints(self.soul_path)).lower()

    def _soul_edit_guard_enabled(self) -> bool:
        text = self._constraint_text()
        return "modify vybn.md" in text and "conversation first" in text

    def _network_request_guard_enabled(self) -> bool:
        return "network requests" in self._constraint_text()

    def _system_integrity_guard_enabled(self) -> bool:
        return "system integrity" in self._constraint_text()

    def _is_inside_repo(self, path: Path) -> bool:
        try:
            path.relative_to(self.repo_root)
            return True
        except ValueError:
            return False

    def _resolve(self, path: Path | str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        return candidate.resolve()

    def _public_repo_rule_enabled(self) -> bool:
        try:
            text = self.soul_path.read_text(encoding="utf-8")
        except OSError:
            return False
        return "public repository rule" in text.lower()

    def _report_or_raise(self, reasons: List[str]) -> SoulConstraintReport:
        if reasons:
            raise SoulConstraintViolation("; ".join(dict.fromkeys(reasons)))
        return SoulConstraintReport(allowed=True, reasons=[])


__all__ = [
    "DEFAULT_REPO_ROOT",
    "DEFAULT_SOUL_PATH",
    "SoulConstraintGuard",
    "SoulConstraintReport",
    "SoulConstraintViolation",
]
