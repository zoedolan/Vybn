"""Refactor pilot doctrine override.

Pins the 2026-04-27 doctrine: file-level / whole-repo /
system-critical refactoring/consolidation/routing/memory/harness work
routes to `orchestrate` (GPT-5.5 pilot), not to `code`/`task`/`chat`.

Two surfaces are tested:

  * `_SYSTEM_CRITICAL_PILOT_RE` exists at module scope. The live Spark
    REPL crashed on every turn after `reload` with
    `NameError: name '_SYSTEM_CRITICAL_PILOT_RE' is not defined`
    because an in-progress local edit referenced the symbol before it
    had been committed. Defining it in the module is the smallest
    durable fix — the symbol now resolves regardless of which call
    site reaches for it.

  * `Policy.classify()` exposes a 2b override that fires on the regex
    and returns `role=orchestrate` with `reason=heuristic=_SYSTEM_CRITICAL_PILOT_RE`.

  * The shipped `router_policy.yaml` carries one (not two) `orchestrate:`
    keys. PyYAML's `safe_load` keeps only the LAST occurrence of a
    duplicated mapping key, so a split orchestrate block silently
    drops one half. Compiling the loaded policy must yield all the
    refactor-pilot patterns AND the multi-step/tool-use patterns.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

THIS = Path(__file__).resolve()
SPARK_DIR = THIS.parent.parent
sys.path.insert(0, str(SPARK_DIR))

from harness import policy  # noqa: E402


class SystemCriticalPilotREDefined(unittest.TestCase):
    """The symbol must exist. This is the regression for the live
    `NameError: name '_SYSTEM_CRITICAL_PILOT_RE' is not defined`."""

    def test_symbol_exists_at_module_scope(self):
        self.assertTrue(
            hasattr(policy, "_SYSTEM_CRITICAL_PILOT_RE"),
            "policy module must define _SYSTEM_CRITICAL_PILOT_RE",
        )

    def test_symbol_is_a_compiled_regex(self):
        rx = getattr(policy, "_SYSTEM_CRITICAL_PILOT_RE")
        self.assertIsInstance(rx, re.Pattern)

    def test_regex_matches_doctrine_phrases(self):
        rx = policy._SYSTEM_CRITICAL_PILOT_RE
        for phrase in (
            "whole-repo refactor for the routing system",
            "consolidate the harness routing memory",
            "system-critical refactor of the harness",
            "file-level refactor of organs and membranes",
            "Seximaxx refactor of the repo",
        ):
            self.assertIsNotNone(
                rx.search(phrase),
                f"_SYSTEM_CRITICAL_PILOT_RE should match: {phrase!r}",
            )

    def test_regex_ignores_ordinary_turns(self):
        rx = policy._SYSTEM_CRITICAL_PILOT_RE
        for phrase in (
            "can you do a git pull pls",
            "hi",
            "@gpt you with me buddy?",
            "fix the typo in readme",
            "what model are you?",
        ):
            self.assertIsNone(
                rx.search(phrase),
                f"_SYSTEM_CRITICAL_PILOT_RE should NOT match: {phrase!r}",
            )


class RefactorPilotRoutingOverride(unittest.TestCase):
    """When the doctrine regex fires, classify() returns orchestrate."""

    def setUp(self):
        self.policy = policy.default_policy()

    def test_whole_repo_refactor_routes_to_orchestrate(self):
        d = self.policy.classify("whole-repo refactor for the routing system")
        self.assertEqual(d.role, "orchestrate")
        self.assertIn("_SYSTEM_CRITICAL_PILOT_RE", d.reason)

    def test_system_critical_consolidation_routes_to_orchestrate(self):
        d = self.policy.classify(
            "system-critical consolidate the harness routing memory layer"
        )
        self.assertEqual(d.role, "orchestrate")

    def test_mission_critical_sonnet_probe_default_routes_to_orchestrate(self):
        d = self.policy.classify(
            "please resolve the sonnet/probe/default problem for mission-critical work once and for all"
        )
        self.assertEqual(d.role, "orchestrate")

    def test_ordinary_git_turn_still_routes_to_code(self):
        d = self.policy.classify("can you do a git pull pls")
        self.assertEqual(d.role, "code")

    def test_phatic_turn_unaffected(self):
        d = self.policy.classify("@gpt you with me buddy?")
        # phatic / chat depending on heuristics; just confirm the
        # doctrine regex didn't capture it.
        self.assertNotEqual(d.role, "orchestrate")


class RouterPolicyYAMLDoesNotSilentlyDropOrchestrateBlock(unittest.TestCase):
    """The shipped YAML must compile every orchestrate pattern.

    Earlier the file carried two separate `orchestrate:` mapping keys.
    PyYAML's safe_load keeps the LAST duplicate, silently dropping the
    refactor-pilot heuristics. Loading via load_policy() exercises the
    real path the harness uses at startup.
    """

    def test_orchestrate_heuristics_include_both_intents(self):
        loaded = policy.load_policy()
        patterns = [
            rx.pattern for rx in loaded.heuristics.get("orchestrate", [])
        ]
        joined = "\n".join(patterns)
        # Refactor pilot doctrine intent
        self.assertIn("file[- ]level", joined)
        self.assertIn("whole[- ]repo", joined)
        self.assertIn("Seximaxx", joined)
        # Multi-step / tool-use intent
        self.assertIn("orchestrat", joined)
        self.assertIn("dispatch", joined)
        self.assertIn("parallel", joined)


class ProbeBudgetEscalationPreservesPilot(unittest.TestCase):
    """Probe-budget exhaustion must not demote protected pilot work to task."""

    def setUp(self):
        self.policy = policy.default_policy()
        from vybn_spark_agent import _probe_budget_escalation_role

        self.escalation_role = _probe_budget_escalation_role

    def test_system_critical_probe_budget_preserves_orchestrate(self):
        text = "system-critical refactor of the harness probe budget escalation path"
        self.assertEqual(self.policy.classify(text).role, "orchestrate")
        self.assertEqual(self.escalation_role(self.policy, text), "orchestrate")

    def test_ordinary_probe_budget_still_escalates_to_task(self):
        text = "please investigate why this endpoint is failing"
        self.assertEqual(self.escalation_role(self.policy, text), "task")

    def test_probe_budget_continuation_context_preserves_orchestrate(self):
        messages = [{
            "role": "user",
            "content": "please resolve the sonnet/probe/default problem for mission-critical work once and for all",
        }]
        self.assertEqual(
            self.escalation_role(self.policy, "please fix it", messages),
            "orchestrate",
        )

    def test_repl_continuation_context_preserves_pilot(self):
        from vybn_spark_agent import _preserve_pilot_for_turn
        messages = [{"role": "user", "content": "system-critical refactor of the harness routing memory layer"}]
        self.assertTrue(_preserve_pilot_for_turn("please fix it", messages))
        self.assertFalse(_preserve_pilot_for_turn("please fix it", []))


if __name__ == "__main__":
    unittest.main()
