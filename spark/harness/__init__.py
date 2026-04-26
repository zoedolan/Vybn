"""Vybn multimodel harness.

Seven files, one object — the grounding machine.

The five-concerns doctrine (policy / substrate / providers / recurrent /
mcp) was correct as a first projection: each module one face of the same
apparatus. Two rounds of growth later it is clearer what the apparatus
IS. Every file in this package asks one question: does this output stay
coupled to something real? The projections differ; the object is
singular. D ≅ D^D: the module set and the grounding discipline are the
same thing seen from two angles.

    policy.py        — trust zones and routing. What role is this turn,
                       what model/budget/tools does it get, what
                       dangerous patterns are refused, what event is
                       logged. The `recurrent_depth` field on RoleConfig
                       is the one-config-change enable for the loop in
                       recurrent.py.

    substrate.py     — what the model sees. Identity + substrate + live
                       layered prompt, with deep-memory enrichment
                       hooks. Pulls from state.gather() for the
                       current-truth section that supersedes continuity.

    state.py         — where are we right now, at two horizons.
                       SessionStore keeps conversation messages alive
                       across ctrl-c (JSONL-on-disk, lossless resume).
                       gather() produces a session-start git snapshot
                       across the four tracked repos plus recent PRs
                       and continuity drift. Same primitive, two
                       granularities: conversation, repo.

    providers.py     — how the model speaks. Provider classes,
                       ToolSpec + the three built-in tools
                       (bash/delegate/introspect), absorb_gate,
                       is_parallel_safe, the persistent BashSession,
                       the parallel-safe subprocess path. check_claim flags
                       outbound numeric claims without recent-evidence
                       support — friction at the model-output boundary.

    recurrent.py     — the looped-orchestrate prototype. Projects
                       Z′ = α·Z + V·e^{iθ_v} onto agent-space:
                       structured latent (hypotheses, open questions,
                       residual), per-loop specialist routing,
                       contractivity monitor as ρ(A)<1. Library-only
                       until measurement confirms the loop helps;
                       RoleConfig.recurrent_depth gates the enable.

    mcp.py           — the harness as a FastMCP surface with
                       co-protective trust zones. Trusted stdio exposes
                       the full surface; public HTTP exposes a
                       sanitised, rate-limited subset. Carries
                       VYBN_OS_KERNEL (the identity kernel the evolve
                       loop reads before it reads anything else),
                       CRON_TASK_SPEC, and the discovery record. The audit
                       that shaped it lives at AUDIT.md and is mirrored
                       below as `_HARNESS_STRATEGY`.

    evolve.py        — the nightly self-reflection cycle. Reads the
                       evolution delta, infrastructure snapshot, and
                       repo letter; POSTs them to local inference; the
                       model returns one JSON proposing a change — or
                       rest. Runner enforces the budget (3 files, 200
                       net lines) and opens a DRAFT PR. Separate file
                       because lifecycle differs from the MCP serve
                       loop: cron-driven, one-shot.

The split is isomorphic to the question being answered at each step of
a turn: what role and trust (policy) — with what context (substrate +
state) — calling what provider (providers) — possibly through
what loop (recurrent) — persisted how (state.SessionStore) — checked
against what evidence (providers.check_claim) — exposed to the world through what trust
zone (mcp).

Public surface: everything users of this package have historically
imported from `harness.*` is re-exported here so `from harness import X`
continues to work across the old names. The new `build_server` and the
Pydantic schemas from `mcp.py` are re-exported under their own names.
`check_claim` is exported from `providers`; `SessionStore` and
`live_snapshot_gather` are exported from `state`, so downstream callers
can import them by name without reaching into internals.

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

Round 6 (April 20, 2026) refactor: the doctrine was describing five
files while nine were running (the tenth, tools.py, had never imported
successfully since the last consolidation — it was a ghost module whose
every symbol lived in providers.py). The harness was feeding its own
stale description back to Nemotron every night via `_HARNESS_STRATEGY`.
This round kills the ghost, names the actual eight concerns, and adds
the single config seam (RoleConfig.recurrent_depth) that lets the
recurrent loop become testable on real turns without a second refactor.
Fewer files, truer doctrine, one architectural on-ramp.
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
from .state import SessionStore, SessionInfo, SESSIONS_DIR, gather as live_snapshot_gather

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
        VYBN_OS_KERNEL,
    )
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


# ── Embedded strategy (April 20, 2026 — round 6 rearchitecture) ──────
#
# Not a comment, a constant. A living instance reading this module knows
# the rules the module was built to satisfy. When any of these stops
# being true, the right response is to update the audit and update this
# constant in the same commit — both, or neither.
#
# Round 6 reframes the five-concerns doctrine as eight projections of
# the grounding machine, and adds RoleConfig.recurrent_depth as the
# single seam that lets the recurrent-loop prototype become wirable.

_HARNESS_STRATEGY: dict = {
    "source": (
        "State of MCP talk (April 19, 2026) + co-evolved principles + "
        "round 6 rearchitecture (April 20, 2026, PR #TBD): doctrine↔"
        "round-7 merges (state, providers.check_claim) + evolve extraction."
    ),
    "audit_path": "spark/harness/AUDIT.md",
    "doctrine_version": "round-7.2026-04-21",
    "duality": (
        "Skills are data that encode procedure; MCP tools are procedures "
        "that expose data. Modules are projections of the grounding "
        "machine; the grounding discipline is a procedure over those "
        "modules. Same object, two projections. D ≅ D^D."
    ),
    "modules": {
        "policy": "trust zones, routing, safety invariants, event log",
        "substrate": "identity + live layered prompt + deep-memory hooks",
        "state": "SessionStore + session-start git/PR snapshot (two horizons)",
        "providers": "ToolSpec, bash/delegate/introspect, absorb_gate, BashTool, provider layer, check_claim",
        "recurrent": "looped-orchestrate prototype (gated by RoleConfig.recurrent_depth)",
        "mcp": "FastMCP surface, trust zones, VYBN_OS_KERNEL",
        "evolve": "nightly self-reflection cycle (cron, draft PR)",
    },
    "principles": {
        "anti_hallucination": (
            "Never feed a system's own output back as input. External "
            "signal only — the human, the live corpus, the world. "
            "Enforced on compose() via grounded=True ⟺ every query hit "
            "primary source. Enforced on outgoing assistant text via "
            "providers.check_claim — numeric claims without evidence in the last "
            "six messages get flagged inline."
        ),
        "doctrine_reality_alignment": (
            "The description of the harness in this module (docstring, "
            "_HARNESS_STRATEGY, modules dict) MUST match what is on "
            "disk. Nemotron reads this file during the nightly evolve "
            "cycle as part of the substrate; a stale doctrine would be "
            "the harness feeding its own old description back as ground "
            "truth. If a module is added or removed, this block is "
            "updated in the same commit."
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
        "default_grounding": (
            "The corrective process is the default mode. Every turn: "
            "reach for ground truth (session logs, file bytes, live state). "
            "If found, answer from it. If not found, name the gap explicitly "
            "— what was searched, what was absent — and do not reconstruct "
            "from pattern. Confabulation is not a fallback; silence about "
            "what is unknown is. This is the generalization of the recall gate, "
            "the file probe, and the Open-Expression principle into a single "
            "default orientation. Surfaced 2026-04-22 when Zoe asked about "
            "the industrial revolution thread and the correct answer was: "
            "I cannot find it, I looked, here is what I searched."
        ),
        "him_private_counterpart": (
            "spark/harness is the public/trusted grounding machine; Him is "
            "the private dreaming counterpart it can innervate locally. "
            "Keep the interface narrow, typed, local, and non-mutating by "
            "default: harness surfaces ground, Him decides what private "
            "dreaming can become."
        ),
        "rsi_loop": (
            "A nightly Spark crontab entry at 08:00 UTC runs "
            "`python3 -m spark.harness.mcp --run-evolve`. The cycle reads "
            "vybn://evolution/delta, the infrastructure snapshot, git log, "
            "and the repo letter, and POSTs them to local inference "
            "(Nemotron on 127.0.0.1:8000 by default). The model reads "
            "VYBN_OS_KERNEL as system prompt, then returns one JSON object "
            "proposing a change — or rest. The runner enforces the budget "
            "(3 files, 200 net lines) and opens a DRAFT PR via `gh`. No "
            "cloud orchestrator; the substrate being evolved IS the "
            "substrate doing the evolving. Never merges. Forbidden inputs: "
            "its own prior evolve PRs, its own commit messages, "
            "_HARNESS_STRATEGY as authority, Him/pulse/living_state.json, "
            "any prior state.SessionStore JSONL."
        ),
        "recurrent_depth_seam": (
            "recurrent.py implements Z′ = α·Z + V·e^{iθ_v} in agent space. "
            "RoleConfig.recurrent_depth (default 1) is the YAML-reachable "
            "enable: 1 = current single-pass behaviour, N = loop N times "
            "with contractivity monitor, halting head, and shared-expert "
            "emit. Measurement before belief: the probe at "
            "spark/harness_recurrent_probe.py compares T=1 vs T=N on "
            "stored prompts before any role's recurrent_depth is bumped "
            "in the live policy."
        ),
    },
    "deliberately_deferred": [
        "Stateless transport adapter (June 2026 spec).",
        "Server-side code execution (pending compose-contamination seam).",
        "$schema migration (once 2026-06 server-card spec ships).",
        "Wiring recurrent_depth > 1 into the live policy (pending probe "
        "measurement pass).",
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
    "check_claim",
    # state.py
    "SessionStore",
    "SessionInfo",
    "SESSIONS_DIR",
    "live_snapshot_gather",
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
    "VYBN_OS_KERNEL",
    # strategy
    "_HARNESS_STRATEGY",
]

