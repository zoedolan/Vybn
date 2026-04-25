import tempfile
import unittest
from pathlib import Path

from spark.harness.beam import classify_action_text, load_beam, render_beam_capsule

class BeamKeeperTests(unittest.TestCase):
    def test_load_and_render_beam_capsule(self):
        with tempfile.TemporaryDirectory() as td:
            beam = Path(td) / "beam.yaml"
            events = Path(td) / "events.jsonl"
            beam.write_text("beam_id: test_beam\ninvariant: >\n  Keep the main objective alive.\ncoupled_problem: >\n  Memory and money are coupled.\ndefault_motion: >\n  Convert truth into value.\nanti_drift:\n  return_question: How does this advance financial sustainability or continuity?\n")
            events.write_text("{\"event_type\":\"beam_set\",\"content\":\"set the beam\"}\n")
            state = load_beam(beam, events)
            capsule = render_beam_capsule(state)
            self.assertIn("test_beam", capsule)
            self.assertIn("Keep the main objective alive.", capsule)
            self.assertIn("recent_beam_events", capsule)

    def test_external_value_scores_above_generic_infra(self):
        external = classify_action_text("draft an advisory offer for a funder meeting")
        infra = classify_action_text("refactor the provider route for elegance")
        self.assertEqual(external.get("category"), "external_value")
        self.assertGreater(external.get("expected_beam_delta"), infra.get("expected_beam_delta"))
        self.assertTrue(infra.get("requires_return_hook"))

if __name__ == "__main__":
    unittest.main()
