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
