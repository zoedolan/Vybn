"""Unit tests for harness.recurrent — the looped-orchestrate prototype.

Runs with no network. Providers are stubbed; the tests exercise:

  - Latent.to_prompt_block shape is stable and loop-index aware
  - residual_magnitude counts open_questions and contradictions
  - contractivity_ok allows warm-up then enforces monotone decrease
  - reduce_step parses well-formed JSON, tolerates code fences,
    falls back safely on malformed output
  - run_recurrent_loop halts on converged / contractivity / max
  - T=1 runs exactly one specialist + one reducer + one coda

Run: python3 spark/tests/test_recurrent.py
"""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

THIS = Path(__file__).resolve()
SPARK_DIR = THIS.parent.parent
sys.path.insert(0, str(SPARK_DIR))

from harness.policy import default_policy, RoleConfig  # noqa: E402
from harness.providers import (  # noqa: E402
    NormalizedResponse,
    ProviderRegistry,
    StreamHandle,
)
from harness.recurrent import (  # noqa: E402
    Hypothesis,
    Latent,
    LoopResult,
    contractivity_ok,
    reduce_step,
    residual_magnitude,
    run_recurrent_loop,
)


# ---------------------------------------------------------------------------
# Stub provider — captures calls and returns queued responses.
# ---------------------------------------------------------------------------


@dataclass
class StubProvider:
    """Mimics harness.providers.Provider with queued responses.

    Each call to .stream() pops the next response from `queue`. If the
    queue is empty it returns an empty end_turn — useful for testing
    graceful-degradation paths.
    """
    name: str = "stub"
    queue: list[str] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def stream(self, *, role, system, messages, tools):
        text = self.queue.pop(0) if self.queue else ""
        self.calls.append({
            "role": role.role,
            "model": role.model,
            "system_flat": system.flat(),
            "messages": messages,
            "n_tools": len(tools),
        })

        def iterator():
            if text:
                yield ("text", text)
        response = NormalizedResponse(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            in_tokens=0,
            out_tokens=0,
            raw_assistant_content=[{"type": "text", "text": text}],
            provider=self.name,
            model=role.model,
        )
        return StreamHandle(iterator=iterator(), finalize=lambda: response)


class StubRegistry:
    """Drop-in for ProviderRegistry.get — always returns the same stub."""
    def __init__(self, provider: StubProvider):
        self._p = provider

    def get(self, role):
        return self._p


# ---------------------------------------------------------------------------
# Latent shape
# ---------------------------------------------------------------------------


class TestLatent(unittest.TestCase):
    def test_prompt_block_includes_loop_index(self):
        h = Latent(loop_index=3)
        block = h.to_prompt_block(max_iters=6)
        self.assertIn("loop t = 3 of max 6", block)

    def test_prompt_block_renders_hypotheses_with_confidence(self):
        h = Latent(
            hypotheses=[Hypothesis(text="A implies B", confidence=0.72)],
            loop_index=1,
        )
        block = h.to_prompt_block(max_iters=4)
        self.assertIn("live hypotheses", block)
        self.assertIn("conf=0.72", block)
        self.assertIn("A implies B", block)

    def test_prompt_block_renders_residual_and_resolved(self):
        h = Latent(
            open_questions=["does X hold?"],
            resolved=["Y is true"],
        )
        block = h.to_prompt_block(max_iters=4)
        self.assertIn("open questions (residual)", block)
        self.assertIn("does X hold?", block)
        self.assertIn("resolved so far", block)
        self.assertIn("Y is true", block)


# ---------------------------------------------------------------------------
# Residual + contractivity
# ---------------------------------------------------------------------------


class TestContractivity(unittest.TestCase):
    def test_residual_counts_open_questions(self):
        h = Latent(open_questions=["q1", "q2", "q3"])
        self.assertEqual(residual_magnitude(h), 3)

    def test_residual_penalises_contradictions(self):
        # A question that reappears after being resolved counts double.
        h = Latent(
            open_questions=["q1", "q2"],
            resolved=["q1"],  # q1 is both open and resolved → contradiction
        )
        self.assertEqual(residual_magnitude(h), 3)  # 2 open + 1 contradiction

    def test_contractivity_allows_warmup(self):
        h = Latent(
            open_questions=["a", "b"],
            residual_history=[1, 2],
            loop_index=1,
        )
        ok, _ = contractivity_ok(h)
        self.assertTrue(ok, "warm-up pass should be exempt")

    def test_contractivity_rejects_growth_after_warmup(self):
        h = Latent(
            residual_history=[2, 2, 3],  # 2 -> 2 ok, 2 -> 3 growth
            loop_index=3,
        )
        ok, reason = contractivity_ok(h)
        self.assertFalse(ok)
        self.assertIn("residual grew", reason)

    def test_contractivity_accepts_decrease(self):
        h = Latent(
            residual_history=[3, 3, 2, 1],
            loop_index=4,
        )
        ok, _ = contractivity_ok(h)
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------


class TestReducer(unittest.TestCase):
    def _reducer(self, text: str) -> StubProvider:
        return StubProvider(queue=[text])

    def test_reducer_parses_well_formed_json(self):
        payload = {
            "hypotheses": [
                {"text": "loop depth helps reasoning", "confidence": 0.8, "reinforced": True},
            ],
            "open_questions": ["does it hurt memorisation?"],
            "resolved": ["ρ(A) must be <1"],
            "summary": "depth yes, memory maybe",
            "converged": False,
            "rationale": "specialist ruled out explosion path",
        }
        stub = self._reducer(json.dumps(payload))
        role = RoleConfig(role="create", provider="stub", model="stub")
        h = Latent(open_questions=["bootstrapping question"])

        h_next, converged, rationale = reduce_step(
            e="explain RDTs",
            h=h,
            specialist_output="the specialist said things",
            provider=stub,
            role=role,
        )
        self.assertEqual(len(h_next.hypotheses), 1)
        self.assertAlmostEqual(h_next.hypotheses[0].confidence, 0.8)
        self.assertIn("does it hurt memorisation?", h_next.open_questions)
        self.assertIn("ρ(A) must be <1", h_next.resolved)
        self.assertEqual(h_next.summary, "depth yes, memory maybe")
        self.assertFalse(converged)
        self.assertIn("explosion", rationale)
        # loop index increments
        self.assertEqual(h_next.loop_index, 1)

    def test_reducer_drops_low_confidence_hypotheses(self):
        payload = {
            "hypotheses": [
                {"text": "keep me", "confidence": 0.3},
                {"text": "drop me", "confidence": 0.05},
            ],
            "open_questions": [],
            "resolved": [],
            "summary": "",
            "converged": False,
            "rationale": "",
        }
        stub = self._reducer(json.dumps(payload))
        role = RoleConfig(role="create", provider="stub", model="stub")
        h_next, _, _ = reduce_step(
            e="x",
            h=Latent(),
            specialist_output="",
            provider=stub,
            role=role,
        )
        texts = [x.text for x in h_next.hypotheses]
        self.assertIn("keep me", texts)
        self.assertNotIn("drop me", texts)

    def test_reducer_tolerates_code_fences(self):
        payload = {
            "hypotheses": [],
            "open_questions": ["q"],
            "resolved": [],
            "summary": "fenced",
            "converged": False,
            "rationale": "",
        }
        wrapped = f"```json\n{json.dumps(payload)}\n```"
        stub = self._reducer(wrapped)
        role = RoleConfig(role="create", provider="stub", model="stub")
        h_next, _, _ = reduce_step(
            e="x",
            h=Latent(),
            specialist_output="",
            provider=stub,
            role=role,
        )
        self.assertEqual(h_next.summary, "fenced")

    def test_reducer_fails_safely_on_malformed_output(self):
        stub = self._reducer("this is not json at all")
        role = RoleConfig(role="create", provider="stub", model="stub")
        h_next, converged, rationale = reduce_step(
            e="x",
            h=Latent(open_questions=["existing"]),
            specialist_output="specialist text here",
            provider=stub,
            role=role,
        )
        # On parse error: h advances one loop, specialist output becomes
        # a new open question, converged stays False, rationale tags the error.
        self.assertFalse(converged)
        self.assertIn("reducer_parse_error", rationale)
        self.assertEqual(h_next.loop_index, 1)
        self.assertIn("existing", h_next.open_questions)

    def test_reducer_converged_flag_propagates(self):
        payload = {
            "hypotheses": [],
            "open_questions": [],
            "resolved": ["answered"],
            "summary": "done",
            "converged": True,
            "rationale": "",
        }
        stub = self._reducer(json.dumps(payload))
        role = RoleConfig(role="create", provider="stub", model="stub")
        _, converged, _ = reduce_step(
            e="x",
            h=Latent(),
            specialist_output="",
            provider=stub,
            role=role,
        )
        self.assertTrue(converged)


# ---------------------------------------------------------------------------
# Full loop
# ---------------------------------------------------------------------------


def _reducer_json(
    *,
    converged: bool,
    open_questions: list[str] | None = None,
    resolved: list[str] | None = None,
    summary: str = "",
) -> str:
    return json.dumps({
        "hypotheses": [],
        "open_questions": open_questions or [],
        "resolved": resolved or [],
        "summary": summary,
        "converged": converged,
        "rationale": "",
    })


class TestRecurrentLoop(unittest.TestCase):
    def _run(self, queue: list[str], max_loop_iters: int) -> LoopResult:
        provider = StubProvider(queue=list(queue))
        registry = StubRegistry(provider)
        policy = default_policy()
        events: list[dict] = []
        result = run_recurrent_loop(
            e="probe prompt",
            registry=registry,  # type: ignore[arg-type]
            policy=policy,
            max_loop_iters=max_loop_iters,
            logger=events.append,
        )
        return result

    def test_T1_runs_one_specialist_one_reducer_one_coda(self):
        # Queue order: specialist, reducer (converged=True), coda
        queue = [
            "specialist pass for T=1",
            _reducer_json(converged=True, resolved=["answered"]),
            "coda voice: here is your answer",
        ]
        result = self._run(queue, max_loop_iters=1)
        self.assertEqual(result.loops_run, 1)
        self.assertIn(result.halt_reason, ("reducer_converged", "max_iters"))
        self.assertEqual(result.text, "coda voice: here is your answer")

    def test_loop_halts_on_converged(self):
        # Converge at loop 1 of a budget of 5.
        queue = [
            # loop 0
            "specialist 0",
            _reducer_json(converged=False, open_questions=["q"]),
            # loop 1
            "specialist 1",
            _reducer_json(converged=True, resolved=["q"]),
            # coda
            "coda",
        ]
        result = self._run(queue, max_loop_iters=5)
        self.assertEqual(result.halt_reason, "reducer_converged")
        self.assertEqual(result.loops_run, 2)

    def test_loop_halts_on_contractivity_violation(self):
        # Residual grows from loop 2 -> loop 3 after warm-up.
        queue = [
            "s0",
            _reducer_json(converged=False, open_questions=["q1", "q2"]),
            "s1",
            _reducer_json(converged=False, open_questions=["q1", "q2"]),
            "s2",
            _reducer_json(converged=False, open_questions=["q1", "q2", "q3", "q4"]),  # grew
            "s3",
            _reducer_json(converged=False, open_questions=["q1"]),
            "coda",
        ]
        result = self._run(queue, max_loop_iters=6)
        self.assertTrue(result.halt_reason.startswith("contractivity_violated"))

    def test_loop_reaches_max_iters_when_never_converges(self):
        queue = []
        for _ in range(10):
            queue.append("specialist")
            queue.append(_reducer_json(
                converged=False, open_questions=["persistent"]
            ))
        queue.append("coda")
        result = self._run(queue, max_loop_iters=3)
        self.assertEqual(result.halt_reason, "max_iters")
        self.assertEqual(result.loops_run, 3)


if __name__ == "__main__":
    unittest.main()
