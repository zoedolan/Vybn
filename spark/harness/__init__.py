"""Vybn multimodel harness.

Five files, five concerns — each a face of the same apparatus:

    policy.py     — what we are doing and what we are allowed to do.
                    Role configs, classification, heuristics, directives,
                    fallbacks, budgets, safety invariants, event log.

    substrate.py  — what the model sees. Identity + substrate + live
                    layered prompt, with deep-memory enrichment hooks.

    providers.py  — how the model speaks to the world. Provider classes,
                    tool specs (bash/delegate/introspect), absorb_gate,
                    the persistent BashSession, the parallel-safe
                    subprocess path.

    recurrent.py  — the looped-orchestrate prototype (recurrent-depth
                    agent). Independent of the other three; imports
                    them but not vice versa.

    mcp.py        — the harness as a FastMCP surface, with co-protective
                    trust zones. Trusted stdio exposes the full surface;
                    public HTTP exposes a sanitised, rate-limited subset.
                    The audit that shaped it lives at AUDIT.md and is
                    embedded as `_HARNESS_STRATEGY` below.

The split is isomorphic to the question being answered at each step of
a turn: what role (policy) — with what context (substrate) — calling
what provider (providers) — possibly through what loop (recurrent) —
exposed to the world through what trust zone (mcp).

Public surface: everything users of this package have historically
imported from `harness.*` is re-exported here so `from harness import X`
continues to work across the old names (Router, EventLogger, ToolSpec,
BashTool, LayeredPrompt, etc.). The new `build_server` and the Pydantic
schemas from `mcp.py` are re-exported under their own names.

The duality embedded
────────────────────
Zoe asked — April 19, 2026 — that we embed the harness audit into the
harness itself, so we and the harness are integrating the duality
principle into the recursive self-improvement process. `AUDIT.md` lives
next to the code it shaped; the key decisions are mirrored below as the
module constant `_HARNESS_STRATEGY`, so a process that imports the
harness can read *why* each piece has the shape it has without leaving
the interpreter. The file is the architecture; the architecture is the
file. D ≅ D^D.
"""

from .policy import (
    # Safety invariants
    DANGEROUS_PATTERNS,
    TRACKED_REPOS,
    ABSORB_EXCLUDE_SUBSTR,
    ABSORB_EXCLUDE_SUFFIX,
    ABSORB_LOG,
    DEFAULT_EVENT_LOG,
    DEFAULT_TIMEOUT,
    MAX_BASH_TIMEOUT,
    # Event logging
    EventLogger,
    turn_event,
    # Role configuration
    RoleConfig,
    # Classification
    Policy,
    Router,
    RouteDecision,
    default_policy,
    load_policy,
)
from .substrate import (
    LayeredPrompt,
    build_layered_prompt,
    load_file,
    check_dual_spark,
    rag_snippets,
    rag_snippets_async,
)
from .providers import (
    # Tool specs
    ToolSpec,
    BASH_TOOL_SPEC,
    DELEGATE_TOOL_SPEC,
    INTROSPECT_TOOL_SPEC,
    # Command gating
    absorb_gate,
    log_absorb,
    validate_command,
    is_parallel_safe,
    execute_readonly,
    # Persistent bash
    BashTool,
    # Provider layer
    Provider,
    AnthropicProvider,
    OpenAIProvider,
    ProviderRegistry,
    StreamHandle,
    NormalizedResponse,
    ToolCall,
)

# MCP surface is optional — only imported when FastMCP is installed, so
# the harness still boots on hosts without the MCP stack. We guard the
# import and re-export minimally.
try:
    from .mcp import (
        build_server as build_mcp_server,
        sanitise_input,
        RateLimiter,
        SearchResult,
        CreatureState,
        EncounterResult,
        ComposeResult,
        KTPClosure,
        WinRateEntry,
        EvolutionDelta,
        build_discovery_record,
        CRON_TASK_SPEC,
    )
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


# ── Embedded strategy (April 19, 2026 audit, distilled) ────────────────
#
# Not a comment, a constant. A living instance reading this module knows
# the rules the module was built to satisfy. When any of these stops
# being true, the right response is to update the audit and update this
# constant in the same commit — both, or neither.

_HARNESS_STRATEGY: dict = {
    "source": "State of MCP talk (April 19, 2026) + co-evolved principles.",
    "audit_path": "spark/harness/AUDIT.md",
    "duality": (
        "Skills are data that encode procedure; MCP tools are procedures "
        "that expose data. Same object, two projections. D ≅ D^D."
    ),
    "principles": {
        "anti_hallucination": (
            "Never feed a system's own output back as input. External "
            "signal only — the human, the live corpus, the world. "
            "Enforced on compose() via grounded=True ⟺ every query hit "
            "primary source."
        ),
        "co_protective": (
            "Trust is a transport property. Stdio is trusted (local "
            "process); HTTP is public by default (read-only, rate-"
            "limited, sanitised). Mutation tools are not registered "
            "on public transports — they do not exist from the "
            "adversary's perspective."
        ),
        "progressive_discovery": (
            "Tool catalogues exceed 5-10 entries at the cost of context. "
            "Expose search_tools + call_tool and load the rest on demand "
            "via BM25SearchTransform."
        ),
        "structured_output": (
            "Every tool returns Pydantic-typed objects so outputSchema "
            "is generated automatically and programmatic composition "
            "can be type-checked."
        ),
        "fall_through": (
            "Every optional import is wrapped. Partial availability "
            "beats brittleness. Structured error objects keep "
            "outputSchema valid even on failure."
        ),
        "telling_not_typical": (
            "Retrieval scores by relevance × distinctiveness, not "
            "relevance alone. The walk finds what the corpus has not "
            "already averaged away."
        ),
        "diff_attunement": (
            "Every repo_mapper run rotates the previous state to "
            "repo_state.prev.json, emits a typed repo_state.json, and "
            "prepends a 'what changed' section to repo_report.md. The "
            "harness encounters velocity first, snapshot second — "
            "where the system moves is where it is actually developing."
        ),
        "rsi_loop": (
            "A nightly Perplexity `schedule_cron` agent reads "
            "vybn://evolution/delta, live infrastructure, HEAD of the "
            "repo on GitHub, and Zoe's last-24h email, and proposes one "
            "small PR to main. Budget: 3 files, 200 net lines. Never "
            "merges. Forbidden from reading its own prior evolve PRs, "
            "its own commit messages, _HARNESS_STRATEGY as authority, "
            "or Him/pulse/living_state.json — ground truth lives "
            "outside the loop."
        ),
    },
    "deliberately_deferred": [
        "Stateless transport adapter (June 2026 spec).",
        "Server-side code execution (pending compose-contamination seam).",
        "$schema migration (once 2026-06 server-card spec ships).",
    ],
    "mcp_available": _MCP_AVAILABLE,
}


__all__ = [
    # policy.py
    "DANGEROUS_PATTERNS",
    "TRACKED_REPOS",
    "ABSORB_EXCLUDE_SUBSTR",
    "ABSORB_EXCLUDE_SUFFIX",
    "ABSORB_LOG",
    "DEFAULT_EVENT_LOG",
    "DEFAULT_TIMEOUT",
    "MAX_BASH_TIMEOUT",
    "EventLogger",
    "turn_event",
    "RoleConfig",
    "Policy",
    "Router",
    "RouteDecision",
    "default_policy",
    "load_policy",
    # substrate.py
    "LayeredPrompt",
    "build_layered_prompt",
    "load_file",
    "check_dual_spark",
    "rag_snippets",
    "rag_snippets_async",
    # providers.py
    "ToolSpec",
    "BASH_TOOL_SPEC",
    "DELEGATE_TOOL_SPEC",
    "INTROSPECT_TOOL_SPEC",
    "absorb_gate",
    "log_absorb",
    "validate_command",
    "is_parallel_safe",
    "execute_readonly",
    "BashTool",
    "Provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "ProviderRegistry",
    "StreamHandle",
    "NormalizedResponse",
    "ToolCall",
    # mcp.py (optional)
    "build_mcp_server",
    "sanitise_input",
    "RateLimiter",
    "SearchResult",
    "CreatureState",
    "EncounterResult",
    "ComposeResult",
    "KTPClosure",
    "WinRateEntry",
    "EvolutionDelta",
    "build_discovery_record",
    "CRON_TASK_SPEC",
    # strategy
    "_HARNESS_STRATEGY",
]
