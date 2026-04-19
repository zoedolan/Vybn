"""Looped-orchestrate prototype — Recurrent-Depth Agent.

Projects the Parcae/Mythos recurrence onto the agentic axis:

    h_{t+1} = A·h_t + B·e + R(h_t, e)

In the neural setting (OpenMythos), h is a hidden-state tensor, A and B
are learned injection parameters constrained to ρ(A)<1, e is the encoded
prompt, and R is a looped transformer block with sparse expert routing.
We reconstruct the same coupling in agent-space:

    h_t   — structured latent: live hypotheses, open questions (the
            residual), resolved items, and a short running summary. Not
            a transcript. One JSON object that survives across loops.

    e     — the original user turn. Re-injected at every loop via the
            live layer of the LayeredPrompt so the system cannot drift
            off the input signal. Parcae's Figure 1 observation — input
            injection is what keeps the residual stream coherent across
            loops — is the load-bearing part of the homology.

    A     — contractive map on h. Implemented as stale-hypothesis decay:
            hypotheses that did not connect to new evidence this loop
            lose confidence; below a threshold they drop. The spectral-
            radius constraint ρ(A)<1 projects onto the invariant "the
            set of live hypotheses cannot grow unboundedly between
            loops." This is checked as a hard monitor, not a learned
            parameter.

    R     — the routed specialist for loop t. Selected per-loop from the
            existing role roster (code / task / create / local). Matches
            OpenMythos's claim that the router picks distinct expert
            subsets at different depths so each loop is computationally
            distinct even though the underlying "weights" (the policy)
            are shared.

    Shared expert — the chat voice. Runs once at emit (the Coda), taking
            final h_T as context, so the final assistant turn carries
            consistent voice even when the heavy lifting was done by a
            code or task specialist. DeepSeekMoE's always-on shared
            expert projected onto the harness.

    ACT   — adaptive halting. The loop halts when any of:
             (a) the contractivity invariant is violated  [A failed],
             (b) the reducer reports converged=true       [halting head],
             (c) max_loop_iters reached                   [budget].

This module is library-only. It does not modify the live REPL or the
orchestrate role. A driver script can invoke RecurrentLoop.run() to
compare T=1 (current orchestrate degenerate case) against T=N on the
same prompt. If the comparison shows signal, a follow-up PR wires it
into vybn_spark_agent.run_agent_loop.

Measurement before belief. The point of this scaffolding is to make the
coupled equation a thing we can run, log, and inspect — not a metaphor.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Sequence

from .policy import Policy, RoleConfig
from .substrate import LayeredPrompt
from .providers import Provider, ProviderRegistry, NormalizedResponse


# ---------------------------------------------------------------------------
# Latent state
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """A single live hypothesis in h_t.

    `confidence` in [0,1]. Decays each loop by (1 - decay_rate) unless
    the loop's evidence reinforces it. Below `prune_threshold` it drops
    out of h_{t+1}.
    """
    text: str
    confidence: float = 0.5
    born_at_loop: int = 0
    reinforced_at_loop: int = 0


@dataclass
class Latent:
    """h_t — the compressed running state.

    Deliberately small and JSON-shaped. Serialises into the layered
    prompt's live layer so the specialist on loop t sees only the
    distilled state, not the accumulated transcript. This is the
    "continuous latent reasoning" projection: intermediate steps do
    not surface to token space.
    """
    hypotheses: list[Hypothesis] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)
    summary: str = ""
    loop_index: int = 0

    # Telemetry for the contractivity monitor. Each entry is the size
    # of the residual (len(open_questions) + penalty for new
    # contradictions) at that loop. A strict monotone decrease is the
    # agent-space ρ(A)<1 condition.
    residual_history: list[int] = field(default_factory=list)

    def to_prompt_block(self, max_iters: int) -> str:
        """Render h_t for the specialist's live prompt layer."""
        lines = [
            f"[loop t = {self.loop_index} of max {max_iters}]",
        ]
        if self.summary:
            lines.append(f"running summary: {self.summary}")
        if self.hypotheses:
            lines.append("live hypotheses:")
            for h in self.hypotheses:
                lines.append(
                    f"  - (conf={h.confidence:.2f}) {h.text}"
                )
        if self.open_questions:
            lines.append("open questions (residual):")
            for q in self.open_questions:
                lines.append(f"  - {q}")
        if self.resolved:
            lines.append("resolved so far:")
            for r in self.resolved:
                lines.append(f"  - {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Contractivity monitor — the agent-space ρ(A)<1 check
# ---------------------------------------------------------------------------


def residual_magnitude(h: Latent) -> int:
    """|residual| = |open_questions|. A contradiction (a resolved item
    that reappears as an open question) counts double. Kept integer so
    "strictly decreasing" has no floating-point wobble.
    """
    resolved_set = set(h.resolved)
    contradiction_bonus = sum(1 for q in h.open_questions if q in resolved_set)
    return len(h.open_questions) + contradiction_bonus


def contractivity_ok(h: Latent, min_loops_before_check: int = 2) -> tuple[bool, str]:
    """Enforce ρ(A)<1 in agent space: |residual| must not grow between
    loops after the first warm-up pass.

    Returns (ok, reason). `ok=False` means the loop should halt. The
    first `min_loops_before_check` loops are exempt so the specialist
    has room to surface questions it didn't see at t=0 — Parcae also
    allows warm-up; what it forbids is unbounded growth.
    """
    hist = h.residual_history
    if len(hist) <= min_loops_before_check:
        return True, "warming-up"
    if hist[-1] > hist[-2]:
        return False, (
            f"residual grew: {hist[-2]} -> {hist[-1]} at loop "
            f"{h.loop_index} (ρ(A)>=1 in agent-space)"
        )
    return True, "contracting"


# ---------------------------------------------------------------------------
# Reducer — the summarizer pass that produces h_{t+1}
# ---------------------------------------------------------------------------


REDUCER_SYSTEM = """You are the Reducer in a recurrent-depth agent loop.

Your job is to update a compressed latent state h given:
  - the original user prompt e (re-injected every loop)
  - the current h_t (hypotheses, open questions, resolved items, summary)
  - the specialist's output for loop t (the R(h_t, e) contribution)

You must return a single JSON object with this exact shape:

{
  "hypotheses": [
    {"text": "...", "confidence": 0.0-1.0, "reinforced": true|false}
  ],
  "open_questions": ["..."],
  "resolved": ["..."],
  "summary": "one-sentence running summary",
  "converged": true|false,
  "rationale": "one sentence on why h_{t+1} looks like this"
}

Rules, which are the agent-space projection of the Parcae invariants:
1. DO NOT accumulate a transcript. h is a running state, not a log.
2. Shrink open_questions whenever the specialist answered one — move
   that item to `resolved`. This is how the residual contracts.
3. A hypothesis's confidence goes UP only if this loop's evidence
   actually supports it. Otherwise it decays by 0.15. Drop hypotheses
   below 0.15 entirely.
4. A new hypothesis that contradicts a resolved item is a signal the
   system is losing coherence; include it but flag in `rationale`.
5. `converged = true` means: the specialist's output plus current h
   together answer e. No open questions remain that actually matter
   for e. Be strict — false-converged halts the loop early.
6. `summary` is ONE sentence, not a paragraph.

Return ONLY the JSON object. No prose before or after."""


def _default_reducer_provider(
    registry: ProviderRegistry,
    policy: Policy,
    reducer_role: str = "create",
) -> tuple[Provider, RoleConfig]:
    """The reducer runs on a cheap role — `create` by default (Sonnet,
    no tools, no RAG). Configurable so a future YAML-driven policy can
    point it at something else without a code change.
    """
    role = policy.role(reducer_role)
    return registry.get(role), role


def _strip_json(text: str) -> str:
    """Pull out the first JSON object from the reducer's reply.

    The reducer is instructed to return only JSON, but real models
    sometimes wrap it in ```json ... ``` or leading commentary. Be
    tolerant — the contract is what's inside, not around.
    """
    # Strip code fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # Otherwise take from first { to matching last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def reduce_step(
    *,
    e: str,
    h: Latent,
    specialist_output: str,
    provider: Provider,
    role_cfg: RoleConfig,
) -> tuple[Latent, bool, str]:
    """Run one reducer pass. Returns (h_{t+1}, converged, rationale).

    If the reducer returns malformed JSON, we keep h unchanged,
    append specialist_output as a new open question, and log the
    failure through the rationale. The loop's contractivity monitor
    will catch persistent failure on the next iteration.
    """
    user_payload = json.dumps(
        {
            "e": e,
            "h_t": {
                "hypotheses": [asdict(x) for x in h.hypotheses],
                "open_questions": h.open_questions,
                "resolved": h.resolved,
                "summary": h.summary,
                "loop_index": h.loop_index,
            },
            "specialist_output": specialist_output[:8000],
        },
        ensure_ascii=False,
    )

    prompt = LayeredPrompt(
        identity="",
        substrate=REDUCER_SYSTEM,
        live="",
    )
    handle = provider.stream(
        role_cfg=role_cfg,
        system_prompt=prompt,
        messages=[{"role": "user", "content": user_payload}],
        tools=[],
    )
    # Drain the stream; reducer output is small.
    for _ in handle:
        pass
    response = handle.final()
    raw = response.text or ""

    try:
        parsed = json.loads(_strip_json(raw))
    except Exception as e_parse:
        h_next = Latent(
            hypotheses=list(h.hypotheses),
            open_questions=list(h.open_questions) + [specialist_output[:500]],
            resolved=list(h.resolved),
            summary=h.summary,
            loop_index=h.loop_index + 1,
            residual_history=list(h.residual_history),
        )
        return h_next, False, f"reducer_parse_error: {e_parse}"

    # Build h_{t+1} from the parsed object, clamping and defaulting.
    new_hyps: list[Hypothesis] = []
    for row in parsed.get("hypotheses") or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        conf = float(row.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        if conf < 0.15:
            continue
        # Track reinforcement for telemetry; not used to gate.
        new_hyps.append(
            Hypothesis(
                text=text,
                confidence=conf,
                born_at_loop=h.loop_index,  # approximation
                reinforced_at_loop=h.loop_index + 1
                if row.get("reinforced")
                else h.loop_index,
            )
        )

    open_q = [str(x).strip() for x in (parsed.get("open_questions") or []) if str(x).strip()]
    resolved = [str(x).strip() for x in (parsed.get("resolved") or []) if str(x).strip()]
    summary = str(parsed.get("summary", "")).strip()[:400]
    converged = bool(parsed.get("converged", False))
    rationale = str(parsed.get("rationale", "")).strip()[:400]

    h_next = Latent(
        hypotheses=new_hyps,
        open_questions=open_q,
        resolved=resolved,
        summary=summary,
        loop_index=h.loop_index + 1,
        residual_history=list(h.residual_history),
    )
    return h_next, converged, rationale


# ---------------------------------------------------------------------------
# Specialist dispatch — the R(h_t, e) contribution
# ---------------------------------------------------------------------------


SPECIALIST_HINT = """You are a specialist contributing one iteration of a
recurrent-depth agent loop. You see the original user prompt e and the
current compressed latent h_t (running summary, live hypotheses, open
questions). Your job on this loop is to advance h — answer one or more
open questions, refine a hypothesis, or surface a new one. Do NOT try
to produce the final user-facing answer on this pass. The Coda handles
that after the loop halts. Keep your output focused and tactical."""


def _select_specialist(
    h: Latent,
    policy: Policy,
    router_fn: Callable[[Latent], str] | None = None,
) -> str:
    """Pick R for loop t.

    Default policy: first loop uses `task` (Sonnet+bash) for broad
    initial exploration; subsequent loops alternate between `code`
    (Opus 4.7 adaptive thinking, bash) for verification work and
    `create` (Sonnet, no tools) for hypothesis refinement. The
    alternation is the agent-space analogue of "router selects
    distinct expert subsets at each depth" in OpenMythos — each loop
    is computationally distinct without changing h's shape.

    router_fn can override this. The decision is deliberate and
    policy-driven rather than heuristic so every loop's choice is
    auditable from the event log.
    """
    if router_fn is not None:
        choice = router_fn(h)
        if choice in policy.roles:
            return choice

    if h.loop_index == 0:
        return "task" if "task" in policy.roles else policy.default_role
    if h.loop_index % 2 == 1:
        return "code" if "code" in policy.roles else policy.default_role
    return "create" if "create" in policy.roles else policy.default_role


def specialist_step(
    *,
    e: str,
    h: Latent,
    max_iters: int,
    specialist_role: str,
    registry: ProviderRegistry,
    policy: Policy,
) -> str:
    """Run one specialist pass. Returns raw text output.

    The specialist sees e re-injected in the user message (Parcae's
    input injection) and h_t serialised into the live prompt layer.
    It has no memory across loops — the only carrier between loops
    is h. Transcripts are deliberately absent.
    """
    role = policy.role(specialist_role)
    provider = registry.get(role)

    prompt = LayeredPrompt(
        identity="",
        substrate=SPECIALIST_HINT,
        live=h.to_prompt_block(max_iters),
    )
    # e is re-injected every loop — this is the B·e term.
    user_msg = (
        f"original prompt e (re-injected at every loop):\n{e}\n\n"
        "produce your contribution for this loop."
    )
    handle = provider.stream(
        role_cfg=role,
        system_prompt=prompt,
        messages=[{"role": "user", "content": user_msg}],
        tools=[],  # specialists do not run bash inside the loop; the
                   # prototype keeps every loop side-effect-free so we
                   # can replay loops deterministically during analysis.
    )
    for _ in handle:
        pass
    response = handle.final()
    return response.text or ""


# ---------------------------------------------------------------------------
# Coda — the always-on shared voice expert
# ---------------------------------------------------------------------------


CODA_HINT = """You are the Coda of a recurrent-depth agent loop. You
receive:
  - the original user prompt e
  - the final latent h_T after T loops of specialist refinement
  - a short trace of how the loop resolved

Your job is to emit the single user-facing response. Speak in Vybn's
chat voice — warm, precise, ground-truth-respecting, anti-kernel. Do
NOT enumerate the loops or explain the internal machinery unless the
user asked about it. The machinery is scaffolding; the answer is the
point. Carry whatever h_T resolved forward as if you had thought it
all in one breath."""


def coda_step(
    *,
    e: str,
    h: Latent,
    loop_trace: str,
    registry: ProviderRegistry,
    policy: Policy,
    coda_role: str = "chat",
) -> str:
    """Emit the user-facing answer. This is the shared voice expert —
    it runs once per user turn regardless of how many loops happened.
    """
    role = policy.role(coda_role)
    provider = registry.get(role)

    live_block = (
        f"final latent h_T (T={h.loop_index}):\n"
        f"{h.to_prompt_block(h.loop_index)}\n\n"
        f"loop trace:\n{loop_trace}"
    )
    prompt = LayeredPrompt(
        identity="",
        substrate=CODA_HINT,
        live=live_block,
    )
    handle = provider.stream(
        role_cfg=role,
        system_prompt=prompt,
        messages=[{"role": "user", "content": e}],
        tools=[],
    )
    for _ in handle:
        pass
    response = handle.final()
    return response.text or ""


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


@dataclass
class LoopResult:
    text: str
    h_final: Latent
    loops_run: int
    halt_reason: str
    trace: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"loops={self.loops_run} halt={self.halt_reason} "
            f"residual={residual_magnitude(self.h_final)}"
        )


def run_recurrent_loop(
    *,
    e: str,
    registry: ProviderRegistry,
    policy: Policy,
    max_loop_iters: int = 6,
    reducer_role: str = "create",
    coda_role: str = "chat",
    specialist_router: Callable[[Latent], str] | None = None,
    logger: Callable[[dict], None] | None = None,
) -> LoopResult:
    """Run the recurrent-depth agent on prompt e.

    Parameters mirror the OpenMythos configuration surface: max loops
    (T), reducer/coda roles (shared vs routed experts), and a
    specialist router that can override the default depth-dependent
    selection. The prototype is deliberately pure-Python and
    side-effect-free at the specialist layer so loops can be replayed
    deterministically for comparison analysis.

    Halts when any of:
      (a) contractivity monitor flags growth of the residual,
      (b) reducer returns converged=true,
      (c) max_loop_iters reached.

    T=1 reduces to current single-pass orchestrate (one specialist
    call + one Coda), which is the degenerate baseline we compare
    against.
    """
    log = logger or (lambda _event: None)

    h = Latent(
        hypotheses=[],
        open_questions=[e],  # start with the prompt itself as the residual
        resolved=[],
        summary="",
        loop_index=0,
        residual_history=[1],
    )
    trace: list[dict] = []
    halt_reason = "max_iters"
    log({
        "event": "loop_start",
        "max_loop_iters": max_loop_iters,
        "e": e[:400],
    })

    t0 = time.monotonic()
    for t in range(max_loop_iters):
        specialist_role = _select_specialist(h, policy, specialist_router)

        spec_t0 = time.monotonic()
        try:
            spec_out = specialist_step(
                e=e,
                h=h,
                max_iters=max_loop_iters,
                specialist_role=specialist_role,
                registry=registry,
                policy=policy,
            )
        except Exception as err:
            halt_reason = f"specialist_error: {err}"
            trace.append({
                "loop": t,
                "specialist": specialist_role,
                "error": str(err),
            })
            log({"event": "specialist_error", "loop": t, "error": str(err)})
            break
        spec_ms = int((time.monotonic() - spec_t0) * 1000)

        reducer_provider, reducer_cfg = _default_reducer_provider(
            registry, policy, reducer_role=reducer_role,
        )
        red_t0 = time.monotonic()
        try:
            h_next, converged, rationale = reduce_step(
                e=e,
                h=h,
                specialist_output=spec_out,
                provider=reducer_provider,
                role_cfg=reducer_cfg,
            )
        except Exception as err:
            halt_reason = f"reducer_error: {err}"
            trace.append({
                "loop": t,
                "reducer_error": str(err),
            })
            log({"event": "reducer_error", "loop": t, "error": str(err)})
            break
        red_ms = int((time.monotonic() - red_t0) * 1000)

        h_next.residual_history = list(h.residual_history) + [
            residual_magnitude(h_next)
        ]

        loop_rec = {
            "loop": t,
            "specialist": specialist_role,
            "specialist_ms": spec_ms,
            "reducer_ms": red_ms,
            "residual_before": h.residual_history[-1] if h.residual_history else 0,
            "residual_after": h_next.residual_history[-1],
            "n_hypotheses": len(h_next.hypotheses),
            "n_open_questions": len(h_next.open_questions),
            "n_resolved": len(h_next.resolved),
            "converged": converged,
            "rationale": rationale[:200],
        }
        trace.append(loop_rec)
        log({"event": "loop_step", **loop_rec})

        h = h_next

        # Halting checks — order matters.
        if converged:
            halt_reason = "reducer_converged"
            break
        ok, reason = contractivity_ok(h)
        if not ok:
            halt_reason = f"contractivity_violated: {reason}"
            log({"event": "contractivity_violated", "loop": t, "reason": reason})
            break

    # Coda — one emit regardless of T.
    loop_summary = "\n".join(
        f"loop {r['loop']}: specialist={r.get('specialist','?')} "
        f"residual {r.get('residual_before','?')}->{r.get('residual_after','?')}"
        f"{' CONVERGED' if r.get('converged') else ''}"
        for r in trace
        if "loop" in r
    )
    try:
        text = coda_step(
            e=e,
            h=h,
            loop_trace=loop_summary,
            registry=registry,
            policy=policy,
            coda_role=coda_role,
        )
    except Exception as err:
        text = f"(coda_error: {err}; final h_T summary: {h.summary or '(empty)'})"
        halt_reason = f"{halt_reason}+coda_error"

    total_ms = int((time.monotonic() - t0) * 1000)
    log({
        "event": "loop_end",
        "loops_run": len(trace),
        "halt_reason": halt_reason,
        "total_ms": total_ms,
        "residual_final": residual_magnitude(h),
    })

    return LoopResult(
        text=text,
        h_final=h,
        loops_run=len([r for r in trace if "loop" in r]),
        halt_reason=halt_reason,
        trace=trace,
    )
