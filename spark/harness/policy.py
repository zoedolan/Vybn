"""Routing policy — loaded from router_policy.yaml.

Policy is configuration, not code. Editing model choice per role, adding
a directive, or changing a fallback chain should not require a code
change. We parse YAML if available and fall back to a minimal JSON-like
loader (not used by default) if not.

A hard-coded default policy is also shipped so the harness functions
even when the YAML is missing — this matches the repo's "fail open,
degrade gracefully" posture.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RoleConfig:
    role: str
    provider: str  # "anthropic" | "openai"
    model: str
    thinking: str = "off"  # "off" | "low" | "adaptive"
    max_tokens: int = 4096
    max_iterations: int = 10
    tools: list[str] = field(default_factory=list)
    temperature: float = 0.7
    base_url: str | None = None  # for openai-compatible local providers
    rag: bool = False


@dataclass
class Policy:
    roles: dict[str, RoleConfig]
    heuristics: dict[str, list[re.Pattern]]
    directives: dict[str, str]
    fallback_chain: dict[str, list[str]]
    budgets: dict[str, float]
    default_role: str = "code"

    def role(self, name: str) -> RoleConfig:
        return self.roles.get(name) or self.roles[self.default_role]


# ---------------------------------------------------------------------------
# Default policy — shipped in code so the harness works out of the box.
# Mirrors spark/router_policy.yaml. Keep them in sync.
# ---------------------------------------------------------------------------

_DEFAULT_ROLES: dict[str, RoleConfig] = {
    "code": RoleConfig(
        role="code",
        provider="anthropic",
        model="claude-opus-4-7",
        thinking="adaptive",
        max_tokens=32768,
        max_iterations=50,
        tools=["bash"],
        rag=False,
    ),
    "create": RoleConfig(
        role="create",
        provider="anthropic",
        model="claude-opus-4-6",
        thinking="off",
        max_tokens=8192,
        max_iterations=3,
        tools=[],
        rag=True,
    ),
    "chat": RoleConfig(
        role="chat",
        provider="anthropic",
        model="claude-opus-4-6",
        thinking="off",
        max_tokens=4096,
        max_iterations=1,
        tools=[],
        rag=True,
    ),
    "task": RoleConfig(
        role="task",
        provider="anthropic",
        model="claude-sonnet-4-6",
        thinking="off",
        max_tokens=4096,
        max_iterations=10,
        tools=["bash"],
        rag=False,
    ),
    "orchestrate": RoleConfig(
        role="orchestrate",
        provider="openai",
        model="gpt-5.4",
        thinking="off",
        max_tokens=2048,
        max_iterations=1,
        tools=[],
        rag=False,
    ),
    # Local Nemotron (vLLM) — OpenAI-compatible endpoint.
    "local": RoleConfig(
        role="local",
        provider="openai",
        model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        thinking="off",
        max_tokens=4096,
        max_iterations=3,
        tools=[],
        base_url="http://127.0.0.1:8000/v1",
        rag=True,
    ),
}

_DEFAULT_HEURISTICS_RAW: dict[str, list[str]] = {
    "code": [
        r"\bgit\b", r"\bpython\b", r"\.py\b", r"\bdef\s+\w+",
        r"^\s*\$ ", r"[Tt]raceback", r"\bpip\b", r"\bnpm\b",
        r"\bpatch\b", r"\bgrep\b", r"\bsed\b", r"\bawk\b",
    ],
    "create": [
        r"\bbrainstorm\b", r"\bsketch\b", r"\bwhat if\b",
        r"\bdraft\b", r"\bdesign\b", r"\bimagine\b",
    ],
    "chat": [
        r"\bhow are you\b", r"\bwhat do you think\b", r"^hi\b", r"^hey\b",
    ],
}

_DEFAULT_DIRECTIVES: dict[str, str] = {
    "/code": "code",
    "/chat": "chat",
    "/create": "create",
    "/plan": "orchestrate",
    "/task": "task",
    "/local": "local",
}

_DEFAULT_FALLBACK: dict[str, list[str]] = {
    "claude-opus-4-7": ["claude-opus-4-6", "claude-sonnet-4-6"],
    "claude-opus-4-6": ["claude-sonnet-4-6"],
    "claude-sonnet-4-6": ["claude-opus-4-6"],
    "gpt-5.4": ["claude-opus-4-6"],
}

_DEFAULT_BUDGETS: dict[str, float] = {
    "per_turn_usd": 2.00,
    "per_session_usd": 25.00,
    "warn_pct": 0.8,
}


def _compile_heuristics(raw: dict[str, list[str]]) -> dict[str, list[re.Pattern]]:
    return {
        role: [re.compile(p, re.IGNORECASE) for p in patterns]
        for role, patterns in raw.items()
    }


def default_policy() -> Policy:
    return Policy(
        roles=dict(_DEFAULT_ROLES),
        heuristics=_compile_heuristics(_DEFAULT_HEURISTICS_RAW),
        directives=dict(_DEFAULT_DIRECTIVES),
        fallback_chain=dict(_DEFAULT_FALLBACK),
        budgets=dict(_DEFAULT_BUDGETS),
    )


# ---------------------------------------------------------------------------
# YAML loader (optional). PyYAML is not a required dependency — we fall
# back to the default policy if the file or the library is missing.
# ---------------------------------------------------------------------------

def load_policy(path: str | os.PathLike | None = None) -> Policy:
    """Load policy from YAML if available, else use the shipped default.

    `path` defaults to env VYBN_HARNESS_POLICY, then spark/router_policy.yaml
    next to this module.
    """
    p = path or os.environ.get("VYBN_HARNESS_POLICY")
    if not p:
        p = Path(__file__).resolve().parent.parent / "router_policy.yaml"
    p = Path(p)
    if not p.exists():
        return default_policy()

    try:
        import yaml  # type: ignore
    except Exception:
        return default_policy()

    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return default_policy()

    roles_raw = data.get("roles") or {}
    roles: dict[str, RoleConfig] = {}
    for name, cfg in roles_raw.items():
        if not isinstance(cfg, dict):
            continue
        roles[name] = RoleConfig(
            role=name,
            provider=str(cfg.get("provider", "anthropic")),
            model=str(cfg.get("model", "claude-opus-4-6")),
            thinking=str(cfg.get("thinking", "off")),
            max_tokens=int(cfg.get("max_tokens", 4096)),
            max_iterations=int(cfg.get("max_iterations", 10)),
            tools=list(cfg.get("tools", []) or []),
            temperature=float(cfg.get("temperature", 0.7)),
            base_url=cfg.get("base_url"),
            rag=bool(cfg.get("rag", False)),
        )
    if not roles:
        roles = dict(_DEFAULT_ROLES)

    heuristics_raw = data.get("heuristics") or _DEFAULT_HEURISTICS_RAW
    directives = dict(data.get("directives") or _DEFAULT_DIRECTIVES)
    fallback = dict(data.get("fallback_chain") or _DEFAULT_FALLBACK)
    budgets = dict(data.get("budgets") or _DEFAULT_BUDGETS)
    default_role = str(data.get("default_role", "code"))
    if default_role not in roles:
        default_role = next(iter(roles))

    return Policy(
        roles=roles,
        heuristics=_compile_heuristics(heuristics_raw),
        directives=directives,
        fallback_chain=fallback,
        budgets=budgets,
        default_role=default_role,
    )
