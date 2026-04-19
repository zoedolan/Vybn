"""Vybn multimodel harness.

Five files, four concerns:

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

The split is isomorphic to the question being answered at each step of
a turn: what role (policy) — with what context (substrate) — calling
what provider (providers) — possibly through what loop (recurrent).

Public surface: everything users of this package have historically
imported from `harness.*` is re-exported here so `from harness import X`
continues to work across the old names (Router, EventLogger, ToolSpec,
BashTool, LayeredPrompt, etc.).
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
]
