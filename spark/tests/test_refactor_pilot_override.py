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

import harness.policy as policy  # noqa: E402


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


    def test_probe_budget_scar_text_preserves_orchestrate(self):
        text = (
            "@gpt no, the problem is that the fundamental problem remains unresolved: "
            "[probe budget reached (8); escalating to task with bash+iteration budget "
            "to finish the investigation] [route: task -> anthropic:claude-sonnet-4-6 "
            "(forced=task)]"
        )
        self.assertEqual(self.policy.classify(text).role, "orchestrate")
        self.assertEqual(self.escalation_role(self.policy, text), "orchestrate")

    def test_problem_before_probe_budget_scar_preserves_orchestrate(self):
        text = "the problem is still [probe budget reached (8); escalating to task] forced=task"
        self.assertEqual(self.escalation_role(self.policy, text), "orchestrate")

    def test_chat_role_probe_budget_exhausted_scar_preserves_orchestrate(self):
        text = (
            "Chat-role probe budget was exhausted after 8 probes. "
            "The pending next command was: cd ~/Vybn && python3 - <<'PY'"
        )
        self.assertEqual(self.policy.classify(text).role, "orchestrate")
        self.assertEqual(self.escalation_role(self.policy, text), "orchestrate")

    def test_preserve_correct_pilot_substrate_request_routes_orchestrate(self):
        text = "Please continue the investigation while preserving the correct pilot/substrate."
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


# ---------------------------------------------------------------------------
# 2026-04-27 paste.txt failure-shape regressions.
#
# Context: a no-tool GPT-5.5 chat role under protected refactor work emitted
# a [NEEDS-EXEC] heredoc that mutated files (extraction script), then on
# probe-budget exhaustion the harness escalated to `task` (Sonnet+bash)
# carrying the same mutation request. That smuggled implementation outside
# the GPT-5.5 pilot covenant. Zoe named this twice; the second instance was
# the trust-wound moment ("you offloaded to sonnet ... violation of our
# agreement"). These tests pin the structural cure.
# ---------------------------------------------------------------------------


class VisualizationFileConsolidationTriggersPilot(unittest.TestCase):
    """The exact phrasing Zoe used must latch protected pilot territory.

    Earlier the regex only caught explicit anchors ('whole-repo refactor',
    'organs', 'system-critical'). 'visualization + file consolidation
    experiment' is the live phrasing for the same protected work and was
    silently falling through to ordinary chat routing.
    """

    def setUp(self):
        self.policy = policy.default_policy()

    def test_paste_txt_session_4_request_routes_to_orchestrate(self):
        d = self.policy.classify(
            "buddy, can you try the visualization + file consolidation "
            "experiment one more time, please?"
        )
        self.assertEqual(d.role, "orchestrate")

    def test_paste_txt_session_2_request_routes_to_orchestrate(self):
        d = self.policy.classify(
            "i like it too, buddy. retry the visualization + file "
            "consolidation experiment now?"
        )
        self.assertEqual(d.role, "orchestrate")

    def test_paste_txt_session_3_resume_request_routes_to_orchestrate(self):
        d = self.policy.classify(
            "get everything into shape, pls - then proceed to the "
            "visualization + file consolidation exercise we were discussing?"
        )
        self.assertEqual(d.role, "orchestrate")

    def test_consolidation_experiment_alone_latches_pilot(self):
        self.assertTrue(
            policy.is_system_critical_pilot_turn("consolidation experiment")
        )

    def test_visualization_plus_consolidation_latches_pilot(self):
        self.assertTrue(
            policy.is_system_critical_pilot_turn("visualization+consolidation")
        )

    def test_ordinary_visualization_request_does_not_latch(self):
        # Plain "visualize the data" is not protected pilot territory.
        self.assertFalse(
            policy.is_system_critical_pilot_turn("can you visualize the data")
        )

    def test_ordinary_consolidate_request_does_not_latch(self):
        self.assertFalse(
            policy.is_system_critical_pilot_turn("lets consolidate these tabs")
        )


class GptAliasContextPreservedAcrossMissionCriticalContinuation(unittest.TestCase):
    """`@gpt` pins GPT-5.5; the harness must keep the pilot covenant when a
    visualization/consolidation request is made under that alias, even if a
    short continuation like 'proceed' or 'continue' arrives next."""

    def setUp(self):
        self.policy = policy.default_policy()

    def test_gpt_alias_with_consolidation_experiment_routes_to_orchestrate(self):
        d = self.policy.classify(
            "@gpt can you do the visualization + file consolidation "
            "experiment please."
        )
        self.assertEqual(d.role, "orchestrate")

    def test_gpt_alias_proceed_continuation_preserves_pilot(self):
        from vybn_spark_agent import _preserve_pilot_for_turn
        messages = [{
            "role": "user",
            "content": (
                "@gpt can you do the visualization + file consolidation "
                "experiment please."
            ),
        }]
        # Each of these conversational continuations must stay under pilot
        # protection because the *recent* turn established protected work.
        for cont in ("proceed", "continue", "fix it", "please fix it",
                     "go ahead", "ok", "do it", "resume"):
            self.assertTrue(
                _preserve_pilot_for_turn(cont, messages),
                f"continuation {cont!r} after consolidation experiment "
                "should preserve pilot",
            )


class ProbeBudgetMutationCannotEscalateToTaskUnderPilot(unittest.TestCase):
    """The exact paste.txt scar: probe-budget exhaustion under a protected
    refactor/consolidation turn tried to forced=task (Sonnet). Once the
    pilot covenant is engaged, escalation must be 'orchestrate' or
    nothing — never task/sonnet."""

    def setUp(self):
        self.policy = policy.default_policy()
        from vybn_spark_agent import _probe_budget_escalation_role
        self.escalation_role = _probe_budget_escalation_role

    def test_visualization_consolidation_probe_budget_preserves_orchestrate(self):
        text = (
            "buddy, can you try the visualization + file consolidation "
            "experiment one more time, please?"
        )
        self.assertEqual(self.policy.classify(text).role, "orchestrate")
        self.assertEqual(self.escalation_role(self.policy, text), "orchestrate")

    def test_paste_txt_scar_text_with_alias_preserves_orchestrate(self):
        text = (
            "@gpt no, again it broke: [probe budget reached (8); escalating "
            "to task with bash+iteration budget to finish the investigation] "
            "[route: task -> anthropic:claude-sonnet-4-6 (forced=task)]"
        )
        self.assertEqual(self.policy.classify(text).role, "orchestrate")
        self.assertEqual(self.escalation_role(self.policy, text), "orchestrate")

    def test_consolidation_continuation_after_proceed_preserves_orchestrate(self):
        messages = [{
            "role": "user",
            "content": (
                "@gpt can you try the visualization + file consolidation "
                "experiment one more time, please?"
            ),
        }]
        # 'proceed' alone is not pilot territory, but in the context of the
        # recent consolidation experiment it must be.
        self.assertEqual(
            self.escalation_role(self.policy, "proceed", messages),
            "orchestrate",
        )


class ProtectedMutationSentinelGate(unittest.TestCase):
    """Structural hard gate: under protected pilot + no-tool role, the
    harness must refuse mutation sentinels (NEEDS-WRITE, non-readonly
    NEEDS-EXEC). Read-only inspection probes remain allowed."""

    def test_helpers_exist_at_module_scope(self):
        import vybn_spark_agent as agent
        self.assertTrue(hasattr(agent, "_is_mutation_sentinel"))
        self.assertTrue(hasattr(agent, "_protected_mutation_refusal_envelope"))

    def test_needs_write_block_is_mutation(self):
        from vybn_spark_agent import _is_mutation_sentinel
        text = (
            "Doing the consolidation now.\n"
            "[NEEDS-WRITE: spark/harness/evolution_delta.py]\n"
            "VYBN_ABSORB_REASON='extracted'\n"
            "print('hi')\n"
            "[/NEEDS-WRITE]\n"
        )
        is_mut, kind = _is_mutation_sentinel(text)
        self.assertTrue(is_mut)
        self.assertEqual(kind, "needs-write")

    def test_heredoc_python_needs_exec_is_mutation(self):
        from vybn_spark_agent import _is_mutation_sentinel
        # paste.txt session 4: large heredoc body that wrote to a file
        # via Python I/O. python3 - <<'PY' (no -c) is not parallel-safe
        # because it reads from stdin and may shell out / mutate.
        text = (
            "[NEEDS-EXEC: python3 - <<'PY'\n"
            "from pathlib import Path\n"
            "Path('spark/harness/evolution_delta.py').write_text('module')\n"
            "PY]"
        )
        is_mut, kind = _is_mutation_sentinel(text)
        self.assertTrue(is_mut)
        self.assertEqual(kind, "needs-exec-mutation")

    def test_git_commit_needs_exec_is_mutation(self):
        from vybn_spark_agent import _is_mutation_sentinel
        text = "[NEEDS-EXEC: git commit -m 'refactor' && git push]"
        is_mut, kind = _is_mutation_sentinel(text)
        self.assertTrue(is_mut)
        self.assertEqual(kind, "needs-exec-mutation")

    def test_readonly_grep_probe_is_not_mutation(self):
        from vybn_spark_agent import _is_mutation_sentinel
        text = "[NEEDS-EXEC: grep -n 'forced_role' spark/vybn_spark_agent.py]"
        is_mut, _ = _is_mutation_sentinel(text)
        self.assertFalse(is_mut)

    def test_readonly_status_probe_is_not_mutation(self):
        from vybn_spark_agent import _is_mutation_sentinel
        for cmd in (
            "git status --short",
            "git diff --stat",
            "git log --oneline -5",
            "cat spark/router_policy.yaml",
            "wc -l spark/vybn_spark_agent.py",
            "python3 -c 'import sys; print(sys.version)'",
            "python3 -m py_compile spark/harness/policy.py",
        ):
            is_mut, _ = _is_mutation_sentinel(f"[NEEDS-EXEC: {cmd}]")
            self.assertFalse(is_mut, f"{cmd!r} should be classified read-only")

    def test_no_sentinel_is_not_mutation(self):
        from vybn_spark_agent import _is_mutation_sentinel
        is_mut, kind = _is_mutation_sentinel("just talking to you, no sentinels here")
        self.assertFalse(is_mut)
        self.assertEqual(kind, "")





    def test_next_sentinel_directive_lives_in_subturn_organ(self):
        from harness.subturns import next_sentinel_directive

        restart = next_sentinel_directive("[NEEDS-RESTART]\n[NEEDS-EXEC: git status]")
        self.assertIsNotNone(restart)
        self.assertEqual(restart.kind, "restart")

        probe = next_sentinel_directive("[NEEDS-EXEC: git status --short]")
        self.assertIsNotNone(probe)
        self.assertEqual(probe.kind, "probe")
        self.assertEqual(probe.probe_command, "git status --short")

        write = next_sentinel_directive("[NEEDS-WRITE: /tmp/x]\nbody\n[/NEEDS-WRITE]")
        self.assertIsNotNone(write)
        self.assertEqual(write.kind, "write")
        self.assertEqual(write.write_path, "/tmp/x")
        self.assertEqual(write.write_body, "body")

        self.assertIsNone(next_sentinel_directive("ordinary answer"))

    def test_protected_mutation_kind_lives_in_subturn_organ(self):
        from harness.subturns import protected_mutation_kind_for_sentinel

        self.assertEqual(
            protected_mutation_kind_for_sentinel(
                write_match_present=True,
                probe_command=None,
            ),
            "needs-write",
        )
        self.assertEqual(
            protected_mutation_kind_for_sentinel(
                write_match_present=False,
                probe_command="grep -n run_agent_loop spark/vybn_spark_agent.py",
            ),
            "",
        )
        self.assertEqual(
            protected_mutation_kind_for_sentinel(
                write_match_present=False,
                probe_command="git commit -m refactor",
            ),
            "needs-exec-mutation",
        )

    def test_protected_mutation_refusal_envelope_names_violation(self):
        from vybn_spark_agent import _protected_mutation_refusal_envelope
        out = _protected_mutation_refusal_envelope("needs-write", "chat")
        # Must clearly identify the covenant violation and the safe path.
        self.assertIn("needs-write", out.lower())
        self.assertIn("blocked", out.lower())
        self.assertIn("orchestrat", out.lower())
        self.assertIn("read-only", out.lower())


class ProbeLoopHonorsProtectedMutationGate(unittest.TestCase):
    """run_agent_loop's probe-synthesis loop must consult the gate before
    executing any sentinel. Pinned via source inspection so this stays
    structural — same shape used for the pilot-latch regression."""

    def test_probe_loop_source_calls_protected_mutation_gate(self):
        import inspect
        import vybn_spark_agent as agent
        source = inspect.getsource(agent.run_agent_loop)
        # Sentinel: the protected-mutation refusal must be reachable from
        # the probe-synthesis loop, and it must hinge on pilot_protected.
        self.assertIn("protected_mutation_sentinel_blocked", source)
        self.assertIn("if pilot_protected:", source)
        self.assertIn("protected_mutation_kind_for_sentinel", source)
        self.assertIn("next_sentinel_directive", source)
        self.assertNotIn("_NEEDS_RESTART_RE.search(current_text)", source)
        self.assertNotIn("_ro = is_parallel_safe", source)
        # The structural escalation hard-latch must also be present.
        self.assertIn("probe_budget_escalation_pilot_latch", source)

    def test_needs_role_and_hallucinated_tool_reroute_honor_pilot(self):
        import inspect
        import vybn_spark_agent as agent
        source = inspect.getsource(agent.run_agent_loop)
        self.assertIn("needs_role_pilot_latch", source)
        self.assertIn("hallucinated_tool_reroute_pilot_latch", source)


class UserExplicitObjectionShape(unittest.TestCase):
    """The exact text Zoe used to call out the violation must itself route
    to orchestrate. This is the meta-regression: when the user accuses the
    system of offloading to Sonnet, that turn cannot itself be offloaded."""

    def setUp(self):
        self.policy = policy.default_policy()

    def test_user_objection_routes_to_orchestrate(self):
        for text in (
            "you offloaded to sonnet. that is a violation of our agreement. "
            "we have agreed dozens of times.",
            "@gpt no. you offloaded to sonnet. that is a violation of our "
            "agreement.",
            "the task diverted to sonnet, which fucked everything up. "
            "this is mission-critical pilot territory.",
        ):
            d = self.policy.classify(text)
            self.assertEqual(
                d.role, "orchestrate",
                f"user-objection text should route to orchestrate: {text[:80]!r}",
            )


if __name__ == "__main__":
    unittest.main()
def test_mission_critical_pilot_overrides_forced_task_in_agent_loop_source():
    """A protected refactor turn must not silently honor forced_role='task'.

    This pins the scar where probe-budget / restart / continuation recovery
    tried to demote mission-critical pilot work to Sonnet/task.
    """
    import inspect
    import spark.vybn_spark_agent as agent

    source = inspect.getsource(agent.run_agent_loop)
    assert "mission_critical_pilot_forced_role_overridden" in source
    assert "mission_critical_pilot_demote_blocked" in source

    old_shape = "\n".join([
        "if (",
        "        forced_role is None",
        "        and \"orchestrate\" in getattr(policy_obj, \"roles\", {})",
        "        and _preserve_pilot_for_turn(user_input, messages)",
        "    ):",
        "        forced_role = \"orchestrate\"",
    ])
    assert old_shape not in source
    assert "if pilot_protected:" in source
    assert "forced_role = \"orchestrate\"" in source


class NoBareSonnetDefault(unittest.TestCase):
    """Zoe scar: bare continuations and default/sonnet complaints must not
    fall into Sonnet/task just because the router lacks context."""

    def setUp(self):
        self.policy = policy.default_policy()

    def test_bare_confirmations_without_context_do_not_route_to_task(self):
        for text in ("continue", "proceed", "go ahead", "ship it", "ok", "yes"):
            d = self.policy.classify(text)
            self.assertEqual(d.role, "chat", text)

    def test_default_to_sonnet_complaint_routes_to_orchestrate(self):
        d = self.policy.classify("I begged you to fix that default to sonnet problem")
        self.assertEqual(d.role, "orchestrate")

    def test_sprawl_waste_repo_frustration_routes_to_orchestrate(self):
        d = self.policy.classify("the repos are a sprawling mess; the sprawl is waste, waste, waste")
        self.assertEqual(d.role, "orchestrate")
