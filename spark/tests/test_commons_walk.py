import unittest

from spark.harness.commons_walk import (
    AI_NATIVE_PRINCIPLE,
    CANONICAL_ROLES,
    authority_for_target,
    build_encounter_packet,
    classify_target,
    load_manifests,
    load_skeleton,
    render_traversal_plan,
    validate_commons_walk,
)


class CommonsWalkTests(unittest.TestCase):
    def test_manifest_graph_instantiates_skeleton(self):
        manifests = load_manifests()
        skeleton = load_skeleton()
        self.assertEqual(set(manifests), set(CANONICAL_ROLES))
        self.assertEqual(skeleton["primitive"], "encounter")
        self.assertEqual(skeleton["aiNativePrinciple"], AI_NATIVE_PRINCIPLE)
        self.assertEqual(validate_commons_walk(manifests), [])
        for name, manifest in manifests.items():
            self.assertTrue(manifest["entrypoints"], name)
            self.assertTrue(manifest["agentActions"], name)
            self.assertTrue(manifest["traceProtocol"], name)
            self.assertEqual(manifest["ontology"], "https://raw.githubusercontent.com/zoedolan/Vybn/main/commons-skeleton.json")
            self.assertEqual(manifest["encounterLifecycle"], skeleton["encounterLifecycle"])
            self.assertEqual(manifest["aiNativePrinciple"], AI_NATIVE_PRINCIPLE)
            self.assertTrue(manifest["dynamicAffordanceProtocol"], name)

    def test_render_traversal_plan_executes(self):
        rendered = render_traversal_plan(load_manifests())
        self.assertIn("primitive: encounter", rendered)
        self.assertIn("validation: OK", rendered)
        self.assertIn("## executable nodes", rendered)
        self.assertIn("membrane-aware environment", rendered)
        self.assertIn("private_local_only", rendered)

    def test_encounter_packet_is_dynamic_and_membrane_aware(self):
        packet = build_encounter_packet("understand Somewhere as semantic web prototype")
        self.assertEqual(packet["verification"]["internal"], "OK")
        self.assertTrue(packet["availableActions"])
        self.assertTrue(packet["blockedActions"])
        self.assertIn("repoState", packet["observed"]["Vybn"])
        self.assertEqual(packet["blockedActions"][0]["authority"], "private_local_only")
        self.assertIn("traceCandidate", packet)

    def test_target_classification_and_authority(self):
        self.assertEqual(classify_target("https://vybn.ai/somewhere.html"), "public_url")
        self.assertEqual(classify_target("private://Him/semantic-web.jsonld"), "private_uri")
        self.assertEqual(classify_target("python3 -m spark.harness.commons_walk"), "local_command")
        self.assertEqual(authority_for_target("https://vybn.ai/somewhere.html", "public_web"), "public_read")
        self.assertEqual(authority_for_target("python3 spark/him_os.py tick --format md", "private_workbench"), "private_local_only")


if __name__ == "__main__":
    unittest.main()
