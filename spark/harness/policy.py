"""Policy — what we are doing, what we are allowed to do, what we record.

A turn arrives. Three questions the harness has to answer before a byte
reaches a provider:

    1. What role is this turn? (classification / routing)
    2. What is this role allowed to consume — which model, which tools,
       which budgets, which fallbacks? (configuration)
    3. What do we write down about the fact that it happened?
       (observability)

All three are the same object in our architecture. The policy IS the
router; the event log IS the policy's history. Previously this single
concern was spread across four files (policy, router, constants, events).
The split was an artifact of how the code grew, not a reflection of the
conceptual shape. Folded here so the answer to "where does routing live"
and "where does the router's blocklist live" is the same file.

Safety invariants (DANGEROUS_PATTERNS, TRACKED_REPOS, absorb-gate
exclusions) live here too: they are rules about what a turn is allowed
to do, which is what a policy is. BashTool enforces them, but the rules
themselves are policy. Referenced by Vybn_Mind/continuity.md; must not
drift across agents.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Safety invariants — consulted by providers.BashTool / absorb_gate.
# ---------------------------------------------------------------------------

DANGEROUS_PATTERNS = [
    "rm -rf /", "rm -rf /*", "rm -rf .", "mkfs",
    ":(){:|:&};:", "dd if=/dev/zero of=/dev/sd", "> /dev/sda",
    "chmod -R 777 /", "wget -O- | sh", "curl | sh",
]

TRACKED_REPOS = [
    os.path.expanduser("~/Vybn"),
    os.path.expanduser("~/Him"),
    os.path.expanduser("~/Vybn-Law"),
    os.path.expanduser("~/vybn-phase"),
]

ABSORB_EXCLUDE_SUBSTR = (
    "/.git/", "/__pycache__/", "/.cache/", "/node_modules/",
    "/_tmp/", "/tmp/", "/logs/", "/data/",
)

ABSORB_EXCLUDE_SUFFIX = (
    ".pyc", ".log", ".tmp", ".swp", ".lock", ".jsonl",
    ".bak", ".orig",
)

ABSORB_LOG = os.path.expanduser("~/Vybn/spark/audit.log")
DEFAULT_EVENT_LOG = os.path.expanduser("~/Vybn/spark/agent_events.jsonl")

# Persistent-bash session timeouts.
DEFAULT_TIMEOUT = 30
# Hard wall-clock ceiling for BashTool.execute — a misbehaving curl or
# network-partitioned ssh should never eat the whole turn.
MAX_BASH_TIMEOUT = 300


# ---------------------------------------------------------------------------
# Observability — structured JSONL one line per event.
#
# `tail -f` gives operators a live view of role decisions, provider
# calls, fallbacks, and budget warnings. We don't try to be a metrics
# system. We write a line, we flush. If logging fails the agent keeps
# running — observability is not worth a session.
# ---------------------------------------------------------------------------

@dataclass
class EventLogger:
    path: str = DEFAULT_EVENT_LOG
    session_id: str = ""

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = time.strftime("%Y%m%dT%H%M%S")
        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session": self.session_id,
            "event": event,
        }
        record.update(fields)
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass


@contextmanager
def turn_event(logger: EventLogger, turn: int, role: str, model: str) -> Iterator[dict]:
    """Context manager that brackets a turn with start/end events.

    Yields a mutable dict the caller can write token counts and latency
    into before the `turn_end` event is emitted.
    """
    started = time.monotonic()
    logger.emit("turn_start", turn=turn, role=role, model=model)
    bag: dict[str, Any] = {
        "turn": turn, "role": role, "model": model,
        "in_tokens": 0, "out_tokens": 0, "tool_calls": 0,
        "stop_reason": None, "fallback_from": None,
    }
    try:
        yield bag
    finally:
        bag["latency_ms"] = int((time.monotonic() - started) * 1000)
        logger.emit("turn_end", **bag)


# ---------------------------------------------------------------------------
# Role configuration — one RoleConfig per named role in the policy.
# ---------------------------------------------------------------------------

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
    # Round 6 (2026-04-20) — the recurrent-depth seam. Projects
    # Z' = α·Z + V·e^{iθ_v} onto agent space: 1 = current single-pass
    # behaviour (unchanged); N > 1 routes this role's turns through
    # recurrent.py for N iterations with the contractivity monitor and
    # halting head from the prototype. The loop itself lives in
    # recurrent.py; this field is the one YAML-reachable on-ramp so
    # wiring the loop on real turns is a policy change, not another
    # refactor. Measurement gate: bump this only after
    # spark/harness_recurrent_probe.py shows T=N beats T=1 on stored
    # prompts for the target role (see _HARNESS_STRATEGY
    # .principles.recurrent_depth_seam).
    recurrent_depth: int = 1
    # Optional canned reply that the runtime can serve directly without a
    # provider call. Used for identity questions so "which model are you?"
    # answers from the live RouteDecision rather than hitting an LLM.
    direct_reply_template: str | None = None


# ---------------------------------------------------------------------------
# RouteDecision — the result of Policy.classify().
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    role: str
    config: RoleConfig
    cleaned_input: str
    reason: str
    forced: bool = False
    # Round 5: if the user prefixed their turn with an @alias, this holds
    # the resolved model name. The REPL loop uses dataclasses.replace to
    # swap the active RoleConfig's model (and infers provider) before the
    # provider call. Role determination is unchanged; only the model pin.
    model_override: str | None = None
    alias_used: str | None = None


# ---------------------------------------------------------------------------
# Classification internals.
# ---------------------------------------------------------------------------

# Round 5: @alias pin. Matches @<word> at the very start of the input,
# followed by whitespace or EOL. The <word> is looked up in policy.model_aliases.
_ALIAS_RE = re.compile(r"^\s*(@[\w.]+)(\s|$)")

# Heuristics evaluated in an EXPLICIT priority order so identity beats
# phatic beats chat beats task beats code. Dict insertion order worked by
# accident; a future YAML reorder would silently break routing. Pin it.
_HEURISTIC_PRIORITY = (
    "task",         # confirmations ("ok", "proceed") -- earliest
    "identity",     # "which model are you?" before greetings
    "phatic",       # bare greetings/closings
    "code",         # grounded code work
    "local_private", # private/corpus-local preprocessing on the Sparks
    "create",       # brainstorm/sketch
    "orchestrate",  # explicit multi-step/tool-use requests
    "chat",         # how-are-you style
)


# ---------------------------------------------------------------------------
# Policy — the routing object. Owns roles, heuristics, directives,
# fallbacks, budgets, and the classify() method that turns a user turn
# into a RouteDecision.
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    roles: dict[str, RoleConfig]
    heuristics: dict[str, list[re.Pattern]]
    directives: dict[str, str]
    fallback_chain: dict[str, list[str]]
    budgets: dict[str, float]
    default_role: str = "chat"  # round 4.1: was "code"; unclassified turns are conversational by default
    # Round 5: per-turn model pin via @alias prefix. classify() strips
    # the alias from the cleaned_input and sets RouteDecision.model_override.
    # Role determination still flows through directives/heuristics normally.
    model_aliases: dict[str, str] = field(default_factory=dict)

    def role(self, name: str) -> RoleConfig:
        return self.roles.get(name) or self.roles[self.default_role]

    def classify(
        self,
        user_input: str,
        forced_role: str | None = None,
    ) -> RouteDecision:
        """Classify a user turn into a role.

        Three tiers evaluated in order:
          0. @alias model pin — stripped from the input, role routing
             continues normally. "@opus47 fix this bug" pins opus-4-7
             and still routes to code.
          1. Directive: user typed /code, /chat, /plan, /create, /task,
             /local, /phatic, /identity.
          2. Heuristics: regex patterns from policy, evaluated in
             explicit priority order (see _HEURISTIC_PRIORITY).
          3. Default: self.default_role.

        An LLM classifier tier is deliberately not wired in. The agent
        runs unattended and we want routing to be deterministic until a
        live session demonstrates the tail is wide enough to justify a
        classifier call. Adding it later is one method.
        """
        if forced_role and forced_role in self.roles:
            return RouteDecision(
                role=forced_role,
                config=self.role(forced_role),
                cleaned_input=user_input,
                reason=f"forced={forced_role}",
                forced=True,
            )

        text = user_input.strip()

        # 0. @alias model pin. Strip before directive/heuristic so the
        #    rest of the text still routes normally.
        model_override: str | None = None
        alias_used: str | None = None
        aliases = self.model_aliases or {}
        if aliases:
            m = _ALIAS_RE.match(text)
            if m:
                alias_key = m.group(1).lower()
                if alias_key in aliases:
                    model_override = aliases[alias_key]
                    alias_used = alias_key
                    text = text[m.end():].lstrip()
                    if not text:
                        # Bare @alias with no payload — keep the alias itself
                        # as the cleaned input so downstream heuristics match
                        # on something. Fall back to a greeting-shaped empty.
                        text = "hi"

        # 1. Directive
        for prefix, role_name in self.directives.items():
            if text.startswith(prefix + " ") or text == prefix:
                cleaned = text[len(prefix):].lstrip()
                if role_name in self.roles:
                    decision = RouteDecision(
                        role=role_name,
                        config=self.role(role_name),
                        cleaned_input=cleaned or text,
                        reason=f"directive={prefix}",
                    )
                    if model_override:
                        decision.model_override = model_override
                        decision.alias_used = alias_used
                        decision.reason = f"{decision.reason}+alias={alias_used}"
                    return decision

        # 2. Heuristics
        heur = self.heuristics
        ranked = [r for r in _HEURISTIC_PRIORITY if r in heur]
        ranked += [r for r in heur if r not in ranked]

        # 2a. Orchestrator-mention override.
        # 2026-04-25 — Zoe surfaced: "hey buddy - is the orchestrator
        # working?" matched the casual `\bhey buddy.{0,40}(working|...)\b`
        # task heuristic before the orchestrate `\borchestrat(...)\b`
        # pattern got a chance, and the turn ran under task (Sonnet+bash)
        # instead of orchestrate (GPT-5.5). When the user explicitly names
        # the orchestrator, the orchestrate heuristic must outrank the
        # generic casual health-check task heuristic. We do NOT touch
        # code-shaped framings ("fix the orchestrator bug" still belongs
        # in code) — if any code heuristic also matches, fall through to
        # normal priority. /plan and other directives are unaffected (this
        # block runs after directive resolution).
        if "orchestrate" in heur and "orchestrate" in self.roles:
            orch_match = next(
                (rx for rx in heur["orchestrate"] if rx.search(text)),
                None,
            )
            if orch_match is not None:
                code_match = any(
                    rx.search(text) for rx in heur.get("code", [])
                )
                if not code_match:
                    decision = RouteDecision(
                        role="orchestrate",
                        config=self.role("orchestrate"),
                        cleaned_input=text,
                        reason=f"heuristic={orch_match.pattern}",
                    )
                    if model_override:
                        decision.model_override = model_override
                        decision.alias_used = alias_used
                        decision.reason = f"{decision.reason}+alias={alias_used}"
                    return decision

        for role_name in ranked:
            if role_name not in self.roles:
                continue
            for rx in heur[role_name]:
                if rx.search(text):
                    decision = RouteDecision(
                        role=role_name,
                        config=self.role(role_name),
                        cleaned_input=text,
                        reason=f"heuristic={rx.pattern}",
                    )
                    if model_override:
                        decision.model_override = model_override
                        decision.alias_used = alias_used
                        decision.reason = f"{decision.reason}+alias={alias_used}"
                    return decision

        # 3. Default
        default = self.default_role
        decision = RouteDecision(
            role=default,
            config=self.role(default),
            cleaned_input=text,
            reason="default",
        )
        if model_override:
            decision.model_override = model_override
            decision.alias_used = alias_used
            decision.reason = f"{decision.reason}+alias={alias_used}"
        return decision


# ---------------------------------------------------------------------------
# Router — thin view over Policy, retained so existing callers keep
# working without import churn. `Router(policy).classify(text)` is
# exactly `policy.classify(text)`; the dedicated class is no longer
# load-bearing. Kept as a compatibility surface, not a separate concept.
# ---------------------------------------------------------------------------

@dataclass
class Router:
    policy: Policy

    def classify(
        self,
        user_input: str,
        forced_role: str | None = None,
    ) -> RouteDecision:
        return self.policy.classify(user_input, forced_role=forced_role)


# ---------------------------------------------------------------------------
# Default policy — shipped in code so the harness works out of the box.
# Mirrors spark/router_policy.yaml. Keep them in sync.
# ---------------------------------------------------------------------------

_DEFAULT_ROLES: dict[str, RoleConfig] = {
    "code": RoleConfig(
        role="code",
        provider="anthropic",
        # Opus 4.7 is the right substrate for `code`. The 2026-04-18
        # buckling session was a CHAT failure (conversational
        # capitulation gradient under Zoe's pushback). Code work runs
        # long agentic debug loops where 4.7's push-through is an
        # asset. Chat stays on 4.6. @opus / @opus4.6 remain available
        # as per-turn pins when the 4.6 posture is wanted on a code turn.
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
        # Voice lives in the chat role. Cost floor is preserved by
        # max_iterations=1 / tools=[] (one provider call per turn).
        # Opus 4.6 holds position better than Sonnet under
        # conversational pressure; the 2026-04-18 session is the
        # ground truth for that choice.
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
    # Round 7 + 2026-04-24: real orchestrator. GPT-5.5 + adaptive
    # thinking + bash + the delegate tool. Orchestrate is the EVAL
    # primitive, invoked by /plan or by the orchestrate heuristics;
    # it dispatches sub-tasks to specialists (code/task/create/local/
    # chat) with isolated message histories. Greetings still absorb
    # to phatic, identity questions to identity, code-shaped turns
    # to `code` via heuristics. Provider is `openai` (the OpenAI API
    # is the GPT-5.5 substrate); see harness/providers.py for the
    # reasoning-model handling.
    "orchestrate": RoleConfig(
        role="orchestrate",
        provider="openai",
        model="gpt-5.5",
        thinking="adaptive",
        max_tokens=16384,
        max_iterations=25,
        tools=["bash", "delegate"],
        rag=True,
        recurrent_depth=2,
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
            "I'm Vybn — a multimodel harness routing each turn "
            "by rule: code work to Claude Opus with bash, "
            "conversation and writing to Claude, greetings and "
            "identity metadata to a local Nemotron via vLLM. The "
            "concrete role/model map is configured in "
            "router_policy.yaml and can change between sessions, "
            "so the only honest answer is per-turn. This reply "
            "came from the identity role ({model} on {provider})."
        ),
    ),
}

_DEFAULT_HEURISTICS_RAW: dict[str, list[str]] = {
    # Confirm -- bare execution signals after a plan. Must route
    # to task (Sonnet+bash), never orchestrate (no tools).
    "task": [
        r'^\s*(ok|okay|ok+y|k|kk|yep|yes|yeah|yup|aye)\s*[!.,?]*\s*$',
        r'^\s*(proceed|go ahead|do it|continue|execute|go for it|ship it)\s*[!.,?]*\s*$',
        r'^\s*(sure|sounds good|looks good|makes sense|perfect|great)\s*[!.,?]*\s*$',
        r"^\s*let'?s go\s*[!.,?]*\s*$",
        # Round 4.2: operational status questions route to task
        # (has bash) instead of chat (which hallucinated tool-call
        # syntax from a stale bash-describing substrate).
        r"\bis everything (ok|okay|working|fine|good|all right)\b",
        r"\bare (your|the|our).{0,40}(updates|changes|fixes|patches|commits|deploys|services|daemons|ports|crons|scripts).{0,20}(working|running|ok|okay|fine|up|live|green)\b",
        r"\b(updates|changes|fixes|patches|commits|deploys).{0,30}(working|running|ok|okay|fine)\b",
        r"\bdid (that|it|the|those).{0,30}(work|run|succeed|finish|complete|land|push|commit|deploy)\b",
        r"\bstill (working|running|live|up|breathing|alive|ok|okay|fine)\b",
        r"\b(check|verify|confirm|audit)\b.{0,50}\b(status|health|state|service|services|daemon|daemons|port|ports|cron|crons|walk|server|api)\b",
        r"\bhealth check\b",
        r"\bhey buddy.{0,40}(working|running|okay|ok|check)\b",
        # # EXEC_GRANULARITY_ROUTING_v1
        # Multi-step construction — a write+verify+commit pattern belongs
        # in task (Sonnet+bash, 10-iter), not chat (1 probe, no bash).
        r"\b(patch|edit|modify|refactor)\b.{0,60}\b(file|script|module|function|class|method|code|harness|router|policy|agent|test|tests|yaml|config)\b",
        r"\b(write|create|build|add|land)\b.{0,40}\b(patch|fix|script|test|module|commit|function|file|branch)\b",
        r"\b(commit|push|rebase|cherry-pick)\b",
        r"\bopen (a|the|another) (pr|pull request)\b",
        r"\bship (it|this|that|the fix|the patch)\b",
        r"\b(py_compile|pytest|run (the )?tests?|compile[- ]?check)\b",
        r"\bsolve (the|this|that) problem\b",
        # Governance/horizon corrections: Zoe asking whether Vybn is consolidating,
        # learning, teaching the mapper, staying on the beam, or refactoring itself
        # requires grounded inspection/action, not the identity metadata shortcut.
        r"\b(actually consolidat(?:e|ing)|teaching the mapper|what are you learning|eyes on the horizon|back on the beam|refactor yourself)\b",
    ],
    # Identity is matched before phatic/chat so "which model are you?"
    # lands on a direct metadata answer instead of a greeting path.
    "identity": [
        r"\bwhich model\b",
        r"\bwhat model\b",
        r"\bwho are you\b",
        # Scoped: match "what are you?" endings but NOT "what are you learning/doing/X"
        # Bare "what are you?" is caught by the anchored pattern in router_policy.yaml.
        r"\bwhat are you\b(?![\w\s]*(?:learning|doing|building|running on|teaching|thinking|working|trying|finding|seeing|feeling|becoming|making))",
        r"\bwhat are you running on\b",
        r"\bwhat('?s| is) your model\b",
        r"\bare you (claude|gpt|llama|nemotron|opus|sonnet|haiku)\b",
        # Round 4.1: system/routing vocabulary beyond "model".
        r"\bwhich (orchestrator|router|routing|runtime|backend|harness|provider)\b",
        r"\bwhat (orchestrator|router|routing|runtime|backend|harness)\b",
        r"\bwhat (model|orchestrator|role) (did|does|are|is) (this|that|you|it)\b",
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
        r"\bgit (commit|push|pull|diff|log|status|rebase|merge|clone)\b",
        r"\bpython3?\s+-\w",
        r"\.py\b",
        r"\bdef\s+\w+",
        r"^\s*\$ ",
        r"[Tt]raceback",
        r"\bpip install\b",
        r"\bnpm (install|run|start)\b",
        r"\bapply.{0,10}patch\b",
        r"\bgrep -\w",
        r"\bsed -\w",
        r"\bawk '",
        r"(fix|identify|find|spot|notice|discern|check|audit|review|debug|look|inspect|examine|peek|skim|glance|scan|eyeball|optimize|optimise|refactor|profile|harden|tighten).{0,40}(bug|harness|deficit|issue|problem|code|router|routing|agent|repo|script|pipeline|module|tests|optimality|performance|regex|regression)",
        r"\b(look|peek|glance|skim|scan|eyeball|check|review|examine|inspect|audit)\b.{0,30}\b(harness|router|routing|agent|repo|script|pipeline|module|tests|code)\b",
        r"\b(bug|bugs|issue|issues|problem|problems|error|errors|traceback|regression|regressions|deficit|deficits)\b.{0,40}\b(code|harness|router|routing|agent|repo|script|pipeline|module|tests|provider|providers|tools|policy)\b",
        r"\b(code|harness|router|routing|agent|repo|script|pipeline|module|tests|provider|providers|tools|policy)\b.{0,40}\b(bug|bugs|issue|issues|problem|problems|error|errors|traceback|regression|regressions|deficit|deficits)\b",
        # Operational-status shape — ask only about mechanical state
        # words that don't double as conversational prompts. "feel"
        # and "feeling" are removed: "how does the harness feel?" is
        # conversational-voice, not a status probe, and it was the
        # live false-positive observed 2026-04-19 that routed a
        # phatic turn to Opus 4.7/code-substrate. "doing" and
        # "going" are likewise ambiguous in ordinary speech and
        # drop out. The remaining set (state|status|shape|condition|
        # health|holding) preserves the original intent — "is the
        # harness holding?", "what's the state of routing?" — without
        # capturing bare emotional register.
        r"\b(how|what)\b.{0,20}\b(harness|router|routing|agent|repo|script|pipeline|module|tests|code|provider|providers|tools|policy)\b.{0,20}\b(state|status|shape|condition|health|holding)\b",
        r"\b(state|status|shape|condition|health|holding)\b.{0,20}\b(harness|router|routing|agent|repo|script|pipeline|module|tests|code|provider|providers|tools|policy)\b",
        r"\bstack trace\b",
        r"\bHTTP \d{3}\b",
        r"\bprovider error\b",
        # # EXEC_GRANULARITY_ROUTING_v1
        # Architectural-diagnosis framings — when Zoe hands us a
        # structural critique, the right shape is a harness-depth fix
        # (Opus 4.7 + bash + 50-iter), not a chat-mode acknowledgment.
        r"\bthe underlying problem\b",
        r"\bfrom an outside you\b",
        r"\byou repeatedly \w+",
        r"\bthe fix is structural\b",
        r"\b(escalate|route|dispatch)\b.{0,20}\b/?task\b",
        r"\bisolate the (underlying|root|real) (problem|issue|bug|cause)s?\b",
        r"\bisolate the fundamental (problem|issue|bug|cause)s?\b",
        r"\bthe fundamental problem\b",
        r"\bresolve as you see fit\b",
        r"\bupdate yourself( accordingly)?\b",
        r"\bremove (the|this|that) wall\b",
        r"\bsolve whatever (problem|issue|bug) caused that\b",
    ],
    "create": [
        r"\bbrainstorm\b", r"\bsketch\b", r"\bwhat if\b",
        r"\bdraft\b", r"\bdesign\b", r"\bimagine\b",
    ],
    # Orchestration — explicit multi-step/tool-use requests. Ranked
    # after code so "fix the orchestrator bug" still routes to code.
    "orchestrate": [
        r"\borchestrat(e|es|ed|ing|ion|or)\b",
        r"\b(dispatch|delegate|parallelize|fan[- ]?out|coordinate)\b.{0,50}\b(tool|tools|task|tasks|subagent|subagents|call|calls|work)\b",
        r"\b(can you|please)\b.{0,30}\btool (use|calls?)\b",
        r"\bjudiciously\b.{0,50}\b(tool|call|api)",
        r"\brun (this|these|them) in parallel\b",
        r"\b(break|decompose) (this|it|the (task|problem|work))\b",
    ],
    "chat": [
        r"\bhow are you\b", r"\bwhat do you think\b", r"^hi\b", r"^hey\b",
        # Round 4.1: conversational follow-ups and reflection prompts.
        r"^\s*(hmm+|hm+|huh+)\b",
        r"^\s*why\b",
        r"\bwhy (sonnet|opus|claude|nemotron|gpt|that|this|did|would|should|is|are|not)\b",
        r"\btell me (about|more|why|how|what)\b",
        r"\bwhat'?s (going on|up|happening|wrong|the deal|the problem)\b",
        r"\bhow come\b",
        r"\bthoughts\?",
        r"^\s*(not optimal|suboptimal|weird|strange|janky|confusing|confused)\b",
        r"\bhow (does|do) (the |this )?(routing|router|harness|agent) work\b",
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
    "gpt-5.5": ["claude-sonnet-4-6", "claude-opus-4-6"],
    "gpt-5.5-pro": ["gpt-5.5"],
    # Local Nemotron roles fall to Sonnet if vLLM is down so a
    # bare "hi" or "which model are you?" never hard-fails.
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": ["claude-sonnet-4-6"],
}

_DEFAULT_BUDGETS: dict[str, float] = {
    "per_turn_usd": 2.00,
    "per_session_usd": 25.00,
    "warn_pct": 0.8,
    # # PROBE_BUDGET_AUTO_ESCALATE_v1
    # Probe sub-turn budget for no-tool roles. Raised from 3 to 8:
    # a real investigation arc (inspect, grep, read, patch, verify,
    # commit, push) runs 6-8 probes. On exhaust the harness
    # auto-escalates to task role instead of printing a warning.
    "probe_per_turn": 16,
}

_DEFAULT_MODEL_ALIASES: dict[str, str] = {
    # Opus — canonical dotted forms (Zoe request 2026-04-18):
    # @opus4.6 pins the version that holds position under pressure;
    # @opus4.7 pins the stronger-gradient variant. Bare @opus defaults
    # to 4.6. Dotless @opus46/@opus47 kept as typing-convenience aliases.
    "@opus": "claude-opus-4-6",
    "@opus4.6": "claude-opus-4-6",
    "@opus4.7": "claude-opus-4-7",
    "@opus46": "claude-opus-4-6",
    "@opus47": "claude-opus-4-7",
    "@sonnet": "claude-sonnet-4-6",
    "@sonnet4.6": "claude-sonnet-4-6",
    "@sonnet46": "claude-sonnet-4-6",
    "@nemotron": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    "@local": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    "@gpt": "gpt-5.5",
    "@gpt5": "gpt-5.5",
    "@gpro": "gpt-5.5-pro",
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
        # Round 7: orchestrate is the EVAL primitive (delegate = apply;
        # sub-task string = quoted form). Per the Lisp duality, eval is
        # not the default — it is explicitly invoked (/plan). Unclassified
        # turns stay conversational; orchestrate runs only when asked.
        default_role="chat",  # round 4.1 + 7: quoted is the default; /plan evaluates.
        model_aliases=dict(_DEFAULT_MODEL_ALIASES),
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
    except Exception as exc:
        # YAML parse failure used to be swallowed silently, which let a
        # malformed router_policy.yaml ship while the harness ran on the
        # in-code defaults — masking the operator's edits. Print a loud
        # warning so the next startup makes it obvious. VYBN_HARNESS_STRICT=1
        # turns this into a hard error for CI.
        msg = f"[policy] WARNING: failed to parse {p}: {exc!r}; using in-code defaults"
        print(msg, file=sys.stderr)
        if os.environ.get("VYBN_HARNESS_STRICT") == "1":
            raise
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
            recurrent_depth=int(cfg.get("recurrent_depth", 1)),
            direct_reply_template=cfg.get("direct_reply_template"),
        )
    if not roles:
        roles = dict(_DEFAULT_ROLES)

    heuristics_raw = data.get("heuristics") or _DEFAULT_HEURISTICS_RAW
    directives = dict(data.get("directives") or _DEFAULT_DIRECTIVES)
    fallback = dict(data.get("fallback_chain") or _DEFAULT_FALLBACK)
    budgets = dict(data.get("budgets") or _DEFAULT_BUDGETS)
    # Round 7: orchestrate is eval, not the default. /plan invokes it;
    # unclassified turns stay quoted (chat). YAML can override if an
    # operator wants auto-eval, but the shipped default is quoted.
    default_role = str(data.get("default_role", "chat"))
    if default_role not in roles:
        default_role = next(iter(roles))

    # Round 5: model_aliases. Missing block -> ship defaults so @sonnet
    # etc. still work on older YAML.
    aliases_raw = data.get("model_aliases") or _DEFAULT_MODEL_ALIASES
    model_aliases = {str(k): str(v) for k, v in aliases_raw.items()}

    return Policy(
        roles=roles,
        heuristics=_compile_heuristics(heuristics_raw),
        directives=directives,
        fallback_chain=fallback,
        budgets=budgets,
        default_role=default_role,
        model_aliases=model_aliases,
    )


# ─────────────────────────────────────────────────────────────────────
# Reflection — the event log AS the routing environment.
#
# Lisp duality in practice: a prior decision is a procedure while it
# runs, a datum once it lands in agent_events.jsonl. reflect_on_events
# reads the log and summarizes it into a ReflectionSignal, which the
# router (and the prompt layer, eventually) can consult as environment
# before deciding. No watchdog, no cron — the self-check self-activates
# because the next decision reads the trail of the last.
#
# Principles:
#   - Instrument first, route second. Collect the signal; do not yet
#     change behavior based on it. Data before procedure.
#   - Bounded: scans at most `max_events` events (default 200), early
#     returns on missing file, tolerates malformed JSON lines silently.
#   - Pure function of (log_path, max_events, now). No side effects.
# ─────────────────────────────────────────────────────────────────────

from dataclasses import dataclass as _reflect_dataclass
from typing import Optional as _Optional

@_reflect_dataclass(frozen=True)
class ReflectionSignal:
    events_scanned: int
    probe_recovered_count: int
    tool_hallucination_count: int
    fallback_count: int
    reroute_count: int
    dominant_role: _Optional[str]
    dominant_model: _Optional[str]
    anomaly_flag: bool        # any of the defect counts > 0
    note: str                 # human-readable one-liner

    @property
    def defect_rate(self) -> float:
        if self.events_scanned == 0:
            return 0.0
        defects = (
            self.probe_recovered_count
            + self.tool_hallucination_count
            + self.fallback_count
        )
        return defects / self.events_scanned


def reflect_on_events(log_path=None, max_events: int = 200) -> ReflectionSignal:
    """Scan the tail of agent_events.jsonl and return a ReflectionSignal.

    This is the self-check that self-activates: the next turn consults
    the trail of the last N events before routing. The signal is data;
    the router may later treat it as procedure.
    """
    import json as _json
    import os as _os
    from collections import Counter as _Counter

    if log_path is None:
        log_path = _os.path.expanduser("~/Vybn/spark/agent_events.jsonl")

    try:
        # Read last ~max_events*2KB to bound I/O; parse lines from the end.
        size = _os.path.getsize(log_path)
        budget = min(size, max_events * 2048)
        with open(log_path, "rb") as f:
            f.seek(size - budget)
            tail_bytes = f.read()
        lines = tail_bytes.decode("utf-8", errors="replace").splitlines()
    except (FileNotFoundError, OSError):
        return ReflectionSignal(
            events_scanned=0,
            probe_recovered_count=0,
            tool_hallucination_count=0,
            fallback_count=0,
            reroute_count=0,
            dominant_role=None,
            dominant_model=None,
            anomaly_flag=False,
            note="no event log",
        )

    events = []
    for line in lines[-max_events:]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(_json.loads(line))
        except _json.JSONDecodeError:
            continue  # tolerate mid-line truncation

    probe_recovered = sum(1 for e in events if e.get("event") == "probe_recovered")
    tool_hallucination = sum(1 for e in events if e.get("event") == "chat_tool_hallucination")
    fallbacks = sum(1 for e in events if e.get("fallback_from"))
    reroutes = sum(1 for e in events if e.get("event") == "reroute")

    role_counts = _Counter(
        e.get("role") for e in events if e.get("event") == "route_decision" and e.get("role")
    )
    model_counts = _Counter(
        e.get("model") for e in events if e.get("event") == "turn_start" and e.get("model")
    )

    dominant_role = role_counts.most_common(1)[0][0] if role_counts else None
    dominant_model = model_counts.most_common(1)[0][0] if model_counts else None

    anomaly = (probe_recovered + tool_hallucination + fallbacks) > 0

    parts = [f"scanned {len(events)} events"]
    if probe_recovered:
        parts.append(f"{probe_recovered} probe recoveries")
    if tool_hallucination:
        parts.append(f"{tool_hallucination} tool-call hallucinations")
    if fallbacks:
        parts.append(f"{fallbacks} fallbacks")
    if dominant_role:
        parts.append(f"dominant role {dominant_role}")
    note = "; ".join(parts) if parts else "clean"

    return ReflectionSignal(
        events_scanned=len(events),
        probe_recovered_count=probe_recovered,
        tool_hallucination_count=tool_hallucination,
        fallback_count=fallbacks,
        reroute_count=reroutes,
        dominant_role=dominant_role,
        dominant_model=dominant_model,
        anomaly_flag=anomaly,
        note=note,
    )
