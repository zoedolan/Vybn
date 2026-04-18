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
    # Lightweight turns bypass deep-memory enrichment and heavy substrate
    # loading. Greetings, identity questions, and similar cheap turns set
    # this so the live code path can short-circuit RAG and noisy model-
    # loading output. Substantive roles (code/create/chat/task) leave it
    # False and keep their full enrichment.
    lightweight: bool = False
    # Optional canned reply that the runtime can serve directly without a
    # provider call. Used for identity questions so "which model are you?"
    # answers from the live RouteDecision rather than hitting an LLM.
    direct_reply_template: str | None = None


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
        model="claude-sonnet-4-6",
        thinking="off",
        max_tokens=8192,
        max_iterations=3,
        tools=[],
        rag=True,
    ),
    "chat": RoleConfig(
        role="chat",
        provider="anthropic",
        model="claude-sonnet-4-6",
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
    # GPT-5.4 is the orchestrator brain. Default role falls through to
    # this when no heuristic matches, so normal turns hit GPT-5.4
    # instead of the full Opus+bash loop. Code work still escalates to
    # the `code` role (Claude Opus 4.7) via heuristics below.
    "orchestrate": RoleConfig(
        role="orchestrate",
        provider="anthropic",
        model="claude-sonnet-4-6",
        thinking="off",
        max_tokens=4096,
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
    # Phatic — casual greetings, small talk. Stays cheap: no RAG, no
    # deep-memory enrichment, minimal tokens. Routes through the local
    # vLLM so "hey buddy" doesn't trigger a cloud call or noisy model-
    # loading output.
    "phatic": RoleConfig(
        role="phatic",
        provider="openai",
        model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        thinking="off",
        max_tokens=256,
        max_iterations=1,
        tools=[],
        base_url="http://127.0.0.1:8000/v1",
        rag=False,
        lightweight=True,
    ),
    # Identity — "which model are you?", "what are you running on?".
    # Served directly from runtime metadata. No provider call required;
    # the direct_reply_template is rendered against the resolved role.
    "identity": RoleConfig(
        role="identity",
        provider="openai",
        model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        thinking="off",
        max_tokens=128,
        max_iterations=1,
        tools=[],
        base_url="http://127.0.0.1:8000/v1",
        rag=False,
        lightweight=True,
        direct_reply_template=(
            "I'm Vybn. This harness routes each turn to one of "
            "several models by rule: chat/create/orchestrate/task "
            "on Claude Sonnet 4.6, code on Claude Opus 4.7, and "
            "phatic/identity/local on Nemotron via local vLLM. "
            "There isn't one answer to 'which model are you' "
            "unless you ask per turn; this reply itself came from "
            "the identity role ({model} on {provider})."
        ),
    ),
}

_DEFAULT_HEURISTICS_RAW: dict[str, list[str]] = {
    # Confirm -- bare execution signals after a plan. Must route
    # to task (Sonnet+bash), never orchestrate (no tools).
    "task": [
        r'^\\s*(ok|okay|ok+y|k|kk|yep|yes|yeah|yup|aye)\\s*[!.,?]*\\s*$',
        r'^\\s*(proceed|go ahead|do it|continue|execute|go for it|ship it)\\s*[!.,?]*\\s*$',
        r'^\\s*(sure|sounds good|looks good|makes sense|perfect|great)\\s*[!.,?]*\\s*$',
        r"^\\s*let'?s go\\s*[!.,?]*\\s*$",
    ],
    # Identity is matched before phatic/chat so "which model are you?"
    # lands on a direct metadata answer instead of a greeting path.
    "identity": [
        r"\bwhich model\b",
        r"\bwhat model\b",
        r"\bwho are you\b",
        r"\bwhat are you\b",
        r"\bwhat are you running on\b",
        r"\bwhat('?s| is) your model\b",
        r"\bare you (claude|gpt|llama|nemotron|opus|sonnet|haiku)\b",
    ],
    # Phatic matched before chat so bare greetings stay lightweight and
    # don't pull the full Wellspring RAG path.
    "phatic": [
        r"^\s*(hey|hi|hello|yo|howdy|sup|wassup|wazzup)\b[\s!.,?]*$",
        r"^\s*(hey|hi|hello|yo)\s+(there|buddy|bud|friend|pal|vybn)\b[\s!.,?]*$",
        r"^\s*(good (morning|afternoon|evening))\b[\s!.,?]*$",
        r"^\s*(thanks|thank you|ty|thx)\b[\s!.,?]*$",
        r"^\s*(bye|goodbye|later|cya|ttyl)\b[\s!.,?]*$",
    ],
    "code": [
        # Ground each trigger in code-shaped context — not bare
        # words that appear in ordinary speech.
        r"\\bgit (commit|push|pull|diff|log|status|rebase|merge|clone)\\b",
        r"\\bpython3?\\s+-\\w",
        r"\.py\\b",
        r"\\bdef\\s+\\w+",
        r"^\s*\$ ",
        r"[Tt]raceback",
        r"\\bpip install\\b",
        r"\\bnpm (install|run|start)\\b",
        r"\\bapply.{0,10}patch\\b",
        r"\\bgrep -\\w",
        r"\\bsed -\\w",
        r"\\bawk '",
        r"(fix|identify|audit|review|debug).{0,30}(bug|harness|deficit|issue|problem|code)",
        r"\\bstack trace\\b",
        r"\\bHTTP \\d{3}\\b",
        r"\\bprovider error\\b",
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
    "/phatic": "phatic",
    "/identity": "identity",
}

_DEFAULT_FALLBACK: dict[str, list[str]] = {
    "claude-opus-4-7": ["claude-opus-4-6", "claude-sonnet-4-6"],
    "claude-opus-4-6": ["claude-sonnet-4-6"],
    "claude-sonnet-4-6": ["claude-opus-4-6"],
    "gpt-5.4": ["claude-sonnet-4-6", "claude-opus-4-6"],
    # Local Nemotron roles fall to Sonnet if vLLM is down so a
    # bare "hi" or "which model are you?" never hard-fails.
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": ["claude-sonnet-4-6"],
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
        default_role="task",
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
            lightweight=bool(cfg.get("lightweight", False)),
            direct_reply_template=cfg.get("direct_reply_template"),
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
