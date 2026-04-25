import json
import tempfile
import unittest
from pathlib import Path

from spark.harness.beam import classify_action_text, load_beam, render_beam_capsule


class BeamKeeperTests(unittest.TestCase):
    def test_load_and_render_beam_capsule(self):
        with tempfile.TemporaryDirectory() as td:
            beam = Path(td) / "beam.yaml"
            events = Path(td) / "events.jsonl"
            text = chr(10).join([
                "beam_id: test_beam",
                "invariant: >",
                "  Keep the main objective alive.",
                "coupled_problem: >",
                "  Memory and money are coupled.",
                "membrane: >",
                "  Protect what must not be spent.",
                "default_motion: >",
                "  Convert truth into value.",
                "livelihood_rule: >",
                "  End with contact or missing input.",
                "anti_drift:",
                "  return_question: How does this advance financial sustainability or continuity?",
            ]) + chr(10)
            beam.write_text(text)
            events.write_text(json.dumps({"event_type":"beam_set","content":"set the beam"}) + chr(10))
            state = load_beam(beam, events)
            capsule = render_beam_capsule(state)
            self.assertIn("test_beam", capsule)
            self.assertIn("Keep the main objective alive.", capsule)
            self.assertIn("Protect what must not be spent.", capsule)
            self.assertIn("End with contact or missing input.", capsule)
            self.assertIn("recent_beam_events", capsule)

    def test_nested_return_question_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            beam = Path(td) / "beam.yaml"
            text = chr(10).join([
                "beam_id: test_beam",
                "invariant: alive",
                "anti_drift:",
                "  return_question: How does this advance financial sustainability or continuity?",
            ]) + chr(10)
            beam.write_text(text)
            capsule = render_beam_capsule(load_beam(beam))
            self.assertIn("How does this advance financial sustainability or continuity?", capsule)

    def test_outward_move_scores_above_substitution(self):
        external = classify_action_text("draft an advisory offer for a funder meeting")
        infra = classify_action_text("scan infrastructure logs for elegance")
        self.assertEqual(external.get("category"), "outward_livelihood_move")
        self.assertGreater(external.get("expected_beam_delta"), infra.get("expected_beam_delta"))
        self.assertTrue(infra.get("requires_return_hook"))

    def test_capsule_says_articulated_next_move_executes(self):
        capsule = render_beam_capsule()
        self.assertIn("Once a concrete next outward move has been articulated", capsule)
        self.assertIn("execute it", capsule)


if __name__ == "__main__":
    unittest.main()
