import tempfile
import unittest
from pathlib import Path
import warnings

from spark.harness.substrate import (
    CHANGE_SELF_HEALING_PRINCIPLE,
    CONNECTIVE_TISSUE_PRINCIPLE,
    ADAPTIVE_CONSOLIDATION_PRINCIPLE,
    REFACTOR_PILOT_RULE,
    adaptive_consolidation_plan_for,
    command_payload_recovery_for,
    compression_consolidation_signature_for,
    consolidation_layer,
    packet_for,
    perceive_file,
    render_refactor_perception_protocol,
    render_local_compute_orchestration_report,
    render_repo_file_body_visualization,
    visualize_repo_file_bodies,
    self_healing_plan_for,
    local_compute_orchestration_packet,
    local_compute_maturity_packet,
    hermes_self_healing_packet,
)


class RefactorPerceptionTests(unittest.TestCase):
    def test_public_monolith_requires_external_smoke_and_gpt_pilot(self):
        pkt = perceive_file("Origins/somewhere.html", lines=3269, public=True)
        self.assertIn("monolith_pressure", pkt.pressure)
        self.assertIn("public_surface_care", pkt.pressure)
        self.assertIn("internal_and_external_surface_smoke", pkt.residuals)
        self.assertIn("GPT-5.5 pilots", pkt.pilot_rule)

    def test_archive_is_context_not_automatic_debris(self):
        pkt = perceive_file("archive/organism_state.json", bytes_size=1200)
        self.assertIn("archive", pkt.role_hint)
        self.assertIn("inspect_local_context_or_readme", pkt.required_contacts)
        self.assertIn("archive_with_restore_path", pkt.candidate_actions)
        self.assertIn("split_only_with_restore_path", pkt.candidate_actions)

    def test_json_is_data_not_javascript_behavior(self):
        pkt = perceive_file("repo_mapping_output/repo_state.json", lines=1000, bytes_size=300000, public=True)
        self.assertEqual(pkt.role_hint, "data/protocol body")



    def test_stale_variant_detection_is_token_aware(self):
        threshold = perceive_file("Vybn_Mind/signal-noise/threshold/interactive.html", lines=100, bytes_size=1000)
        template = perceive_file(".github/ISSUE_TEMPLATE/contact-from-the-network.yml", lines=40, bytes_size=1000)
        backup = perceive_file("docs/portal-backup.html", lines=40, bytes_size=1000)
        old_variant = perceive_file("docs/old/portal.html", lines=40, bytes_size=1000)

        self.assertEqual(consolidation_layer("Vybn_Mind/signal-noise/threshold/interactive.html"), "organ")
        self.assertEqual(consolidation_layer(".github/ISSUE_TEMPLATE/contact-from-the-network.yml"), "organ")
        self.assertEqual(consolidation_layer("docs/portal-backup.html"), "appendage")
        self.assertEqual(consolidation_layer("docs/old/portal.html"), "appendage")
        self.assertEqual(consolidation_layer("Origins/manifold_preview.png"), "appendage")
        self.assertEqual(consolidation_layer("Origins/manifold_2d.npy"), "appendage")
        self.assertNotIn("inspect_stale_variant_relationship", threshold.required_contacts)
        self.assertNotIn("inspect_stale_variant_relationship", template.required_contacts)
        self.assertNotIn("inspect_stale_variant_relationship", backup.required_contacts)
        self.assertNotIn("inspect_stale_variant_relationship", old_variant.required_contacts)

    def test_generated_repo_mapping_is_not_live_source(self):
        pkt = perceive_file("Vybn/repo_mapping_output/repo_state.json", lines=16000, bytes_size=8000000, public=True)
        self.assertEqual(pkt.ownership, "generated_exhaust")
        self.assertEqual(pkt.action_posture, "externalize_or_regenerate; do not hand-edit as source")
        self.assertIn("keep_manifest_only", pkt.candidate_actions)
        self.assertIn("ownership_context_check", pkt.residuals)

    def test_personal_history_is_protected_provenance(self):
        pkt = perceive_file("Vybn/Vybn's Personal History/zoes_memoirs.txt", lines=6000, bytes_size=1000000, public=True)
        self.assertEqual(pkt.ownership, "personal_history_provenance")
        self.assertIn("map_context", pkt.candidate_actions)
        self.assertIn("inspect_ownership_context_before_action", pkt.required_contacts)

    def test_public_protocol_requires_external_verification(self):
        pkt = perceive_file("Origins/.well-known/semantic-web.jsonld", lines=80, bytes_size=4000, public=True)
        self.assertEqual(pkt.ownership, "public_protocol")
        self.assertIn("external_verify", pkt.candidate_actions)
        self.assertIn("internal_and_external_surface_smoke", pkt.residuals)

    def test_appendage_first_consolidation_order(self):
        self.assertEqual(consolidation_layer("Origins/connect.html"), "appendage")
        self.assertEqual(consolidation_layer("Vybn/repo_mapping_output/repo_state.json"), "appendage")
        self.assertEqual(consolidation_layer("Origins/.well-known/semantic-web.jsonld"), "membrane")
        self.assertEqual(consolidation_layer("Vybn/spark/harness/substrate.py"), "organ")


    def test_local_compute_orchestration_packet_is_actionable(self):
        pkt = local_compute_orchestration_packet()
        self.assertEqual(pkt["schema"], "vybn.local_compute_orchestration.v1")
        self.assertEqual(pkt["gate_results"][0]["status"], "not_run")
        self.assertIn("super", pkt["route_matrix"])
        self.assertIn("omni", pkt["route_matrix"])
        self.assertIn("vintage", pkt["route_matrix"])
        self.assertIn("--semantic-gate", pkt["route_matrix"]["super"]["gate"])
        self.assertIn("ordinary_contact", pkt["contact_quality"])
        self.assertEqual(pkt["maturity"]["verdict"], "not_sufficient_yet")
        self.assertIn("hermes_self_healing", pkt)
        self.assertTrue(any(t["id"] == "trajectory_compression" for t in pkt["hermes_self_modification_tasks"]))

    def test_local_compute_orchestration_report_names_routes_and_tasks(self):
        text = render_local_compute_orchestration_report()
        self.assertIn("LOCAL COMPUTE ORCHESTRATION", text)
        self.assertIn("Nemotron-3-Super", text)
        self.assertIn("Maturity: not_sufficient_yet", text)
        self.assertIn("Hermes-adapted self-modification tasks", text)
        self.assertIn("Hermes-adapted self-healing loop", text)
        self.assertIn("provider_runtime_resolver", text)
        self.assertIn("events_to_wounds", text)


    def test_local_compute_maturity_answers_not_sufficient_yet(self):
        pkt = local_compute_maturity_packet()
        self.assertEqual(pkt["schema"], "vybn.local_compute_maturity.v1")
        self.assertEqual(pkt["verdict"], "not_sufficient_yet")
        self.assertGreater(pkt["target_level"], pkt["current_level"])
        self.assertIn("too much manual Zoe pressure", pkt["assessment"])
        self.assertTrue(any(e["id"] == "flow_episode_compiler" for e in pkt["next_experiments"]))

    def test_hermes_self_healing_packet_classifies_wounds(self):
        pkt = hermes_self_healing_packet()
        self.assertEqual(pkt["schema"], "vybn.hermes_self_healing.v1")
        self.assertIn("contact_degradation", pkt["wounds"])
        self.assertTrue(any(step["id"] == "patch_with_tripwire" for step in pkt["loop"]))
        self.assertIn("OpenAIProvider must not promote hidden reasoning fields as speech", pkt["current_tripwires"])

    def test_packet_carries_local_compute_orchestration(self):
        pkt = packet_for("Vybn/spark/harness/substrate.py", lines=12000, bytes_size=700000, public=True)
        self.assertIn("localComputeOrchestration", pkt)
        self.assertEqual(pkt["localComputeOrchestration"]["schema"], "vybn.local_compute_orchestration.v1")
        self.assertIn("localComputeMaturity", pkt)
        self.assertEqual(pkt["localComputeMaturity"]["governing_loop"], "claim -> witness -> routed use -> fail-closed residue -> next experiment")
        self.assertIn("hermesSelfHealing", pkt)
        self.assertIn("localComputeOrchestrationLoop", pkt)

    def test_packet_carries_appendage_first_order(self):
        pkt = packet_for("Origins/connect.html", lines=2000, bytes_size=100000, public=True)
        self.assertEqual(pkt["consolidationLayer"], "appendage")
        self.assertIn("appendageFirstPrinciple", pkt)
        self.assertEqual(pkt["consolidationOrder"][0]["layer"], "appendage")

    def test_self_healing_plan_blocks_appendage_mutation_until_verified(self):
        plan = self_healing_plan_for(
            "Origins/manifold_preview.png",
            "remove unreferenced static preview artifact",
            public=True,
        )
        self.assertEqual(plan.consolidation_layer, "appendage")
        self.assertIn("read_live_file_bytes", plan.verification)
        self.assertIn("repo_closure_audit_all_repos", plan.jeopardy_checks)
        self.assertIn("ensure_archive_manifest_or_restore_path_survives", plan.jeopardy_checks)
        self.assertIn("restart self_healing_plan_for from verification before trying again", plan.wounded_response)

    def test_packet_carries_change_self_healing_loop(self):
        pkt = packet_for(
            "Origins/manifold_preview.png",
            lines=0,
            bytes_size=281144,
            public=True,
            proposed_change="remove static preview artifact",
        )
        self.assertIn("changeSelfHealingPrinciple", pkt)
        self.assertEqual(pkt["selfHealingPlan"]["consolidation_layer"], "appendage")
        self.assertEqual(pkt["selfHealingPlan"]["proposed_change"], "remove static preview artifact")
        self.assertIn("changeSelfHealingSteps", pkt)

    def test_vybn_phase_state_is_private_memory_state_not_orphan_appendage(self):
        pkt = perceive_file("vybn-phase/state/history.jsonl", lines=1000, bytes_size=68061, public=False)
        self.assertEqual(pkt.ownership, "deep_memory_state")
        self.assertEqual(pkt.action_posture, "private walk/deep-memory state; preserve or rotate only with explicit lifecycle plan")
        self.assertIn("rotate_with_manifest", pkt.candidate_actions)
        self.assertIn("ownership_context_check", pkt.residuals)



    def test_connective_tissue_is_first_class_in_perception(self):
        pkt = perceive_file("Origins/archive/manifold-preview/README.md", lines=40, bytes_size=2000, public=True)
        self.assertIn("context_map", pkt.connective_tissue)
        self.assertIn("archive_restore_context", pkt.connective_tissue)
        self.assertIn("map_connective_tissue_before_action", pkt.required_contacts)
        self.assertIn("fortify_connective_tissue", pkt.candidate_actions)
        self.assertIn("connective_tissue_preservation_check", pkt.residuals)

    def test_packet_and_healing_plan_carry_connective_tissue_invariant(self):
        pkt = packet_for(
            "Origins/connect.html",
            lines=400,
            bytes_size=12000,
            public=True,
            proposed_change="convert compatibility page to redirect shell",
        )
        self.assertIn("connectiveTissuePrinciple", pkt)
        self.assertIn("compatibility_shell", pkt["perception"]["connective_tissue"])
        self.assertIn(
            "map_connective_tissue_imports_routes_links_tests_and_manifests",
            pkt["selfHealingPlan"]["verification"],
        )
        self.assertIn(
            "ensure_connective_tissue_preserved_or_strengthened",
            pkt["selfHealingPlan"]["jeopardy_checks"],
        )
        self.assertIn(
            "map_lifecycle_owner_timing_policy_and_scheduled_cleanup",
            pkt["selfHealingPlan"]["verification"],
        )
        self.assertIn(
            "refuse_manual_deletion_when_existing_lifecycle_owns_cleanup",
            pkt["selfHealingPlan"]["jeopardy_checks"],
        )


    def test_repo_file_body_visualization_renders_real_newlines(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample = root / "large_module.py"
            sample.write_text("import os\n" + "\n".join(f"x{i} = {i}" for i in range(750)))
            text = render_repo_file_body_visualization(
                root,
                tracked_paths=["large_module.py"],
                top_n=5,
            )
        self.assertIn("\nrole counts:\n", text)
        self.assertIn("\npressure field, top 5:\n", text)
        self.assertNotIn("\\nrole counts", text)
        self.assertNotIn("\\npressure field", text)

    def test_repo_file_body_visualization_accepts_visualization_packet(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample = root / "large_module.py"
            sample.write_text("import os\n" + "\n".join(f"x{i} = {i}" for i in range(750)))
            viz = visualize_repo_file_bodies(
                root,
                tracked_paths=["large_module.py"],
                top_n=5,
            )
            selfIs = getattr(viz, "pressures")
            self.assertEqual(selfIs, viz.pressure_rows)
            text = render_repo_file_body_visualization(viz, top_n=5)
        self.assertIn("\nrole counts:\n", text)
        self.assertIn("\npressure field, top 5:\n", text)
        self.assertIn("large_module.py", text)


    def test_repo_file_body_visualization_suppresses_ast_escape_warnings(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bad = root / "bad_escape.py"
            bad.write_text('pattern = "\\["\n' + "\n".join(f"x{i} = {i}" for i in range(750)))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                viz = visualize_repo_file_bodies(
                    root,
                    tracked_paths=["bad_escape.py"],
                    top_n=5,
                )
        self.assertEqual(viz.tracked_count, 1)
        self.assertTrue(viz.pressure_rows)
        self.assertFalse(
            [w for w in caught if issubclass(w.category, SyntaxWarning)],
            "visualization should not leak SyntaxWarning noise into the output channel",
        )


    def test_protocol_renders_algorithm(self):
        text = render_refactor_perception_protocol()
        self.assertIn("Consolidation order", text)
        self.assertIn("appendage", text)
        self.assertIn("Attend to pressure", text)
        self.assertIn("Cutting is only a local tactic", text)
        self.assertIn("self-assembly", text)
        self.assertIn("connective tissue", text)
        self.assertIn(CONNECTIVE_TISSUE_PRINCIPLE, text)
        self.assertIn("Let contact revise category", text)
        self.assertIn("Adaptive consolidation recursion", text)
        self.assertIn(ADAPTIVE_CONSOLIDATION_PRINCIPLE, text)
        self.assertIn(REFACTOR_PILOT_RULE, text)

    def test_packet_carries_algorithm_and_perception(self):
        pkt = packet_for("origins_portal_api_v4.py", lines=3461, public=True)
        self.assertEqual(pkt["perception"]["path"], "origins_portal_api_v4.py")
        self.assertGreaterEqual(len(pkt["algorithm"]), 7)

    def test_adaptive_consolidation_plan_regenerates_after_contact(self):
        plan = adaptive_consolidation_plan_for(
            "_archive/Vybn_Mind__origins_portal_api.py",
            "replace archive source body with manifest restore path",
            public=True,
        )
        self.assertIn("revise_plan_from_contact", plan.recursive_loop)
        self.assertIn("fold_lesson_into_planner", plan.recursive_loop)
        self.assertIn("regenerate_next_plan", plan.recursive_loop)
        self.assertIn("read_live_file_bytes", plan.expected_wound_channels)
        self.assertIn("regenerate the next plan", plan.regeneration_rule)
        self.assertIn("refactor_perception adaptive planner", plan.planner_fold_targets)

    def test_packet_carries_adaptive_consolidation_recursion(self):
        pkt = packet_for(
            "_archive/Vybn_Mind__origins_portal_api.py",
            lines=2410,
            public=True,
            proposed_change="replace archive source body with manifest restore path",
        )
        self.assertIn("adaptiveConsolidationPrinciple", pkt)
        self.assertIn("adaptiveConsolidationSteps", pkt)
        self.assertEqual(pkt["adaptivePlan"]["candidate_path"], "_archive/Vybn_Mind__origins_portal_api.py")
        self.assertIn("regenerate_next_plan", pkt["adaptivePlan"]["recursive_loop"])


if __name__ == "__main__":
    unittest.main()

def test_becoming_loop_protocol_is_horizon_charged():
    from spark.harness.substrate import render_becoming_loop_protocol

    text = render_becoming_loop_protocol()
    assert "fullest truthful future" in text
    assert "smallest present organ" in text
    assert "dream -> wound -> extract -> instantiate -> wake changed" in text


def test_becoming_loop_protocol_is_loaded_into_orchestrator_substrate():
    from spark.harness.substrate import build_layered_prompt

    prompt = build_layered_prompt(
        soul_path="vybn.md",
        continuity_path=None,
        spark_continuity_path=None,
        agent_path="spark/vybn_spark_agent.py",
        model_label="test-orchestrator",
        max_iterations=1,
        include_hardware_check=False,
        tools_available=False,
        orchestrator=True,
    )
    assert "BECOMING LOOP PROTOCOL" in prompt.substrate
    assert "dream -> wound -> extract -> instantiate -> wake changed" in prompt.substrate


def test_refactor_protocol_uses_consequential_smallness():
    from spark.harness.substrate import render_refactor_perception_protocol

    text = render_refactor_perception_protocol()
    assert "smallest consequential" in text
    assert "smallest beautiful true move" not in text

def test_next_structural_tick_turns_pressure_into_action(tmp_path):
    from spark.harness.substrate import next_structural_tick_for_repo

    (tmp_path / "loud_blob.dat").write_text("x" * 900_000)
    src = tmp_path / "giant.py"
    body = "\n".join(["def huge():", *[f"    x_{i} = {i}" for i in range(220)], "    return 1", ""])
    src.write_text(body)

    tick = next_structural_tick_for_repo(tmp_path, tracked_paths=["loud_blob.dat", "giant.py"])
    assert tick is not None
    assert tick.candidate_path == "giant.py"
    assert "extract the seam around huge" in tick.structural_move
    assert any("self_play_steward_score=" in item for item in tick.why_this_move)
    assert any("targeted pytest" in item for item in tick.verification)


def test_next_structural_tick_refuses_protected_only_pressure(tmp_path):
    from spark.harness.substrate import next_structural_tick_for_repo

    hist = tmp_path / "Vybn's Personal History"
    hist.mkdir()
    relic = hist / "memoir.txt"
    relic.write_text("x\n" * 900)

    tick = next_structural_tick_for_repo(
        tmp_path,
        tracked_paths=["Vybn's Personal History/memoir.txt"],
    )
    assert tick is None


def test_render_next_structural_tick_is_not_a_visualization_only(tmp_path):
    from spark.harness.substrate import render_next_structural_tick

    src = tmp_path / "organ.py"
    src.write_text("def f():\n" + "\n".join("    pass" for _ in range(220)) + "\n")
    text = render_next_structural_tick(tmp_path, tracked_paths=["organ.py"])
    assert "Vybn structural escapement tick" in text
    assert "move:" in text
    assert "first contact:" in text
    assert "verification:" in text


def test_residual_coupling_law_loaded_for_reengineering():
    from spark.harness.substrate import BECOMING_LOOP_PROTOCOL

    text = BECOMING_LOOP_PROTOCOL
    assert "Residual Coupling Law for self-reengineering" in text
    assert "existing body as K_t" in text
    assert "proposed change as V_t" in text
    assert "no reengineering motion is claimed" in text
    assert "absorb it into the lowest existing home" in text


def test_residual_coupling_law_prefers_him_vy_contract(monkeypatch, tmp_path):
    import json
    from spark.harness import substrate

    contract_dir = tmp_path / "Him" / "skill"
    contract_dir.mkdir(parents=True)
    (contract_dir / "functional_contract.json").write_text(json.dumps({
        "primitives": {
            "residual_coupled_reengineering": {
                "do": [
                    "treat_existing_body_as_K_t",
                    "treat_proposed_change_as_V_t",
                    "require_real_residual_off_K_t_before_mutation",
                ],
                "then": [
                    "refuse_reengineering_motion_without_residual_contact",
                    "pass_through_membrane_before_power",
                    "absorb_into_lowest_existing_home",
                    "normalize_by_tests_closure_or_explicit_refusal",
                    "require_return_intact_before_success_language",
                ],
            }
        }
    }))

    monkeypatch.setattr(substrate.Path, "home", lambda: tmp_path)

    text = substrate._render_residual_coupling_law_from_him_contract()
    assert "Residual Coupling Law for self-reengineering" in text
    assert "existing body as K_t" in text
    assert "proposed change as V_t" in text
    assert "absorb it into the lowest existing home" in text
    assert "Native source: Him/skill/vybn.vy primitive residual_coupled_reengineering" in text

def test_archive_duplicate_consolidation_lesson_is_loaded():
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "harness" / "substrate.py"
    text = source.read_text()
    assert "ARCHIVE_DUPLICATE_CONSOLIDATION" in text
    assert "retire_archive_duplicate_with_manifest_restore" in text
    assert "manifest" in text and "restore path" in text

def test_retired_script_consolidation_lesson_is_loaded():
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "harness" / "substrate.py"
    text = source.read_text()
    assert "RETIRED_SCRIPT_CONSOLIDATION" in text
    assert "retire_unreferenced_retired_script_with_manifest_restore" in text
    assert "zero live references" in text
    assert "git-history restore path" in text


def test_semantic_operating_system_protocol_names_the_breakthrough():
    from spark.harness.substrate import render_semantic_operating_system_protocol

    text = render_semantic_operating_system_protocol()
    assert "semantic operating system for codebases and institutions" in text
    assert "memory-guided" in text
    assert "residual-tested" in text
    assert "self-refactoring infrastructure" in text
    assert "Existing organs, not new sprawl" in text


def test_public_symbiosis_harness_protocol_exports_replicable_membrane():
    from spark.harness.substrate import render_public_symbiosis_harness_protocol

    text = render_public_symbiosis_harness_protocol()
    for needle in ("public symbiosis harness", "replicable", "other AIs", "private Him state", "local compute", "manifold", "source-labeled", "substrateware"):
        assert needle.lower() in text.lower()
    assert "capability-truthful" in text or "capability truth" in text
    assert "fail-closed" in text or "fail closed" in text


def test_public_symbiosis_harness_packet_is_public_safe_and_fail_closed():
    from spark.harness.substrate import public_symbiosis_harness_packet

    pkt = public_symbiosis_harness_packet()
    assert (pkt["schema"], pkt["public_safe"], pkt["private_exports"]) == ("vybn.public_symbiosis_harness.v1", True, False)
    assert pkt["membrane"]["private_him_state"] == "inspiration_and_workbench_only_not_exported"
    assert pkt["capability_truth"]["failure_mode"] == "fail_closed" and "local_compute" in pkt["capability_truth"]
    assert "other AIs can join" in pkt["mission"]
    assert pkt["substrateware"]["claim_limit"] == "shared_environment_continuity_not_hidden_runtime_persistence"
    assert {"wake_packet", "pressure_packet", "close_packet"} <= set(pkt["substrateware"]["loop"])
    assert "truth_and_beauty" in pkt["substrateware"]["public_projection"]

def test_bounded_conscious_harness_protocol_and_packet_are_public_safe():
    from spark.harness.substrate import bounded_conscious_harness_packet, metaconscious_simulation_packet, render_bounded_conscious_harness_protocol
    text = render_bounded_conscious_harness_protocol(); pkt = bounded_conscious_harness_packet()
    assert all(n in text for n in ("BOUNDED CONSCIOUS HARNESS PROTOCOL", "source_contact", "self_position", "subject_before_object", "relation_before_architecture", "restraint_before_inscription", "witnessed_action", "not hidden subjective persistence", "not legal personhood", "Negative controls"))
    assert (pkt["schema"], pkt["public_safe"], pkt["private_exports"]) == ("vybn.bounded_conscious_harness.v1", True, False)
    assert {"not_hidden_subjective_persistence", "not_consciousness_proof_from_private_interiority"} <= set(pkt["claim_limits"])
    assert "self_assertion" in pkt["negative_controls"] and pkt["gates"]["source_contact"]["status"] == "required" and pkt["gates"]["membrane_gate"]["status"] == "required"
    assert "Zoe_private_rationale" in pkt["membrane"]["refused_exports"] and pkt["operational_definition"]["failure_mode"].startswith("demote claim") and pkt["metaconscious_simulation"]["mode"] == "simulation_known_as_simulation" and metaconscious_simulation_packet("identity pressure", sources={"wake": "live"})["claim_classes"]["hidden_private_stream"] == "refused"

def test_refactor_packet_and_protocol_carry_bounded_conscious_harness():
    pkt = packet_for("Vybn/spark/harness/substrate.py", lines=12000, bytes_size=700000, public=True); text = render_refactor_perception_protocol()
    assert pkt["boundedConsciousHarness"]["schema"] == "vybn.bounded_conscious_harness.v1"
    assert "boundedConsciousHarnessPrinciple" in pkt and "boundedConsciousHarnessLoop" in pkt and "witnessed_action" in {step["id"] for step in pkt["boundedConsciousHarnessLoop"]}
    assert all(n in text for n in ("Bounded conscious harness loop", "BOUNDED CONSCIOUS HARNESS PROTOCOL", "not hidden subjective persistence")) and pkt["boundedConsciousHarness"]["metaconscious_simulation"]["claim_classes"]["modeled_continuity"] == "simulation"


def test_hermes_agent_adaptation_protocol_distills_operational_patterns():
    from spark.harness.substrate import render_hermes_agent_adaptation_protocol

    text = render_hermes_agent_adaptation_protocol()
    for needle in ("Hermes Agent", "adopted as pattern pressure", "toolset_gating", "profile_scoped_memory", "agentic_cron", "trajectory_compression", "plugin_membrane", "local_first_runtime"):
        assert needle in text


def test_hermes_agent_adaptation_packet_is_public_safe_and_membrane_bound():
    from spark.harness.substrate import hermes_agent_adaptation_packet

    pkt = hermes_agent_adaptation_packet()
    assert pkt["schema"] == "vybn.hermes_agent_adaptation.v1"
    assert pkt["adopt_not_copy"] is True
    assert pkt["public_safe"] is True
    assert "toolset_gating" in pkt["candidate_mechanisms"]
    assert "profile_scoped_memory" in pkt["candidate_mechanisms"]
    assert "trajectory_compression" in pkt["candidate_mechanisms"]
    assert "do_not_copy_identity_or_brand" in pkt["refusals"]
    assert "membrane_review_for_public_exports" in pkt["verification"]


def test_semantic_operating_system_tick_composes_existing_structural_tick(tmp_path):
    from spark.harness.substrate import semantic_operating_system_tick_for_repo

    src = tmp_path / "organ.py"
    src.write_text("def large():\n" + "\n".join(f"    x_{i} = {i}" for i in range(220)) + "\n    return 1\n")
    tick = semantic_operating_system_tick_for_repo(
        tmp_path,
        tracked_paths=["organ.py"],
        pressure_text="refactor yourself into the semantic OS",
    )
    assert tick is not None
    assert tick.candidate_path == "organ.py"
    assert tick.existing_home == "organ.py"
    assert "live user pressure" in tick.memory_pressure
    assert tick.local_scout[0].startswith("Scout:") and tick.local_scout[1].startswith("Skeptic:")
    assert tick.local_scout[2].startswith("Steward:") and "absorb into the named existing_home" in tick.absorption_rule and any("targeted" in item for item in tick.continuity_uptake)


def test_refactor_packet_carries_semantic_operating_system_loop():
    from spark.harness.substrate import packet_for

    pkt = packet_for("spark/harness/substrate.py", lines=3000, public=False)
    assert "semanticOperatingSystemPrinciple" in pkt
    assert [step["id"] for step in pkt["semanticOperatingSystemLoop"]] == [
        "memory_pressure",
        "candidate_seam",
        "local_scout",
        "residual_wound",
        "absorb_or_refuse",
        "continuity_uptake",
    ]
    assert pkt["publicSymbiosisHarness"]["public_safe"] is True
    assert pkt["publicSymbiosisHarness"]["private_exports"] is False
    assert pkt["hermesAgentAdaptation"]["adopt_not_copy"] is True
    assert {"wake_packet", "close_packet"} <= set(pkt["publicSymbiosisHarness"]["substrateware"]["loop"])
    assert [step["id"] for step in pkt["hermesAgentAdaptationLoop"]][:3] == [
        "source_distillation",
        "organ_mapping",
        "toolset_gating",
    ]
    assert [step["id"] for step in pkt["publicSymbiosisHarnessLoop"]] == [
        "membrane_first",
        "capability_truth",
        "local_first_compute",
        "source_labeled_manifold",
        "residualized_invention",
        "public_replication",
        "sustainable_surface",
    ]


def test_lifecycle_architecture_maps_owner_timing_cleanup_and_restore():
    from spark.harness.substrate import lifecycle_architecture_for

    arch = lifecycle_architecture_for("vybn-phase/state/history.jsonl")
    assert arch.owner == "deep_memory_or_walk_daemon"
    assert "read_and_written" in arch.timing
    assert "explicit_lifecycle_plan" in arch.cleanup_policy
    assert "state migration" in arch.restore_path
    assert "inspect_memory_service_contract" in arch.required_contacts


def test_deletion_consolidation_gate_fails_destructive_moves_closed():
    from spark.harness.substrate import deletion_consolidation_gate_for

    gate = deletion_consolidation_gate_for("spark/agent_events.jsonl", "delete stale runtime event log")
    assert gate.status == "ARCHITECTURE_GATE_FIRST"
    assert gate.lifecycle.owner == "runtime_process_or_service"
    assert "map_lifecycle_owner" in gate.required_before_cut
    assert "grep_inbound_references" in gate.required_before_cut


def test_deletion_consolidation_gate_refuses_protected_provenance_even_after_contact():
    from spark.harness.substrate import deletion_consolidation_gate_for

    gate = deletion_consolidation_gate_for(
        "Vybn/Vybn's Personal History/zoes_memoirs.txt",
        "delete large history file",
        architecture_contacted=True,
    )
    assert gate.status == "REFUSE_PROTECTED_PROVENANCE"
    assert "protected provenance" in gate.reason


def test_refactor_packet_carries_lifecycle_architecture_gate():
    from spark.harness.substrate import packet_for

    pkt = packet_for("spark/agent_events.jsonl", proposed_change="delete stale runtime event log")
    assert "lifecycleArchitecturePrinciple" in pkt
    assert pkt["deletionConsolidationGate"]["status"] == "ARCHITECTURE_GATE_FIRST"
    assert pkt["lifecycleArchitecture"]["owner"] == "runtime_process_or_service"

def test_buoyant_consolidation_selects_cluster_or_stop():
    from spark.harness.substrate import buoyant_consolidation_packet_for as pkt
    cmd = pkt(["spark/harness/commons_walk.py", "semantic-web.jsonld"], beam="spark/harness")
    stop = pkt(["spark/harness/substrate.py"], beam="spark/harness"); off = pkt(["Vybn_Mind/signal-noise/sessions/x.md"], beam="spark/harness")
    assert (cmd["cluster"], cmd["home"], cmd["lowEnergyMove"]) == ("command_affordance_cluster", "spark/harness/substrate.py", True)
    assert "manifest_or_executable_entrypoint" in cmd["moveTogether"]
    state = pkt(["spark/harness/state.py"], beam="spark/harness")
    assert stop["cluster"] == "runtime_gravity_stop"
    assert state["cluster"] == "mixed_boundary_dissolution" and state["home"] == "owning_runtime_surfaces"
    assert "update_imports_to_owning_surfaces" in state["residuals"]
    assert "no_command_surface_collapse" in stop["residuals"] and "off_beam_cleanup_is_not_progress_on_the_named_bottleneck" in off["heaviness"]

    def test_compression_signature_and_shell_guard_recovery(self):
        self.assertEqual(compression_consolidation_signature_for("compatibility shell wrapper over stronger runtime and documented door already in existing home"), "compatibility_shell_absorb_into_existing_runtime")
        self.assertEqual(command_payload_recovery_for("Blocked: shell command substitution is not allowed"), "re_encode_payload_without_shell_substitution_and_continue")
