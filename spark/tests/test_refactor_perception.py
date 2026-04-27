import unittest

from spark.harness.refactor_perception import (
    REFACTOR_PILOT_RULE,
    packet_for,
    perceive_file,
    render_refactor_perception_protocol,
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

    def test_protocol_renders_algorithm(self):
        text = render_refactor_perception_protocol()
        self.assertIn("Attend to pressure", text)
        self.assertIn("Let contact revise category", text)
        self.assertIn(REFACTOR_PILOT_RULE, text)

    def test_packet_carries_algorithm_and_perception(self):
        pkt = packet_for("origins_portal_api_v4.py", lines=3461, public=True)
        self.assertEqual(pkt["perception"]["path"], "origins_portal_api_v4.py")
        self.assertGreaterEqual(len(pkt["algorithm"]), 7)


if __name__ == "__main__":
    unittest.main()
