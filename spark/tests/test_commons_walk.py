import unittest

from spark.harness.commons_walk import (
    CANONICAL_ROLES,
    classify_target,
    load_manifests,
    load_skeleton,
    validate_commons_walk,
    render_traversal_plan,
)


class CommonsWalkTests(unittest.TestCase):
    def test_manifest_graph_instantiates_skeleton(self):
        manifests = load_manifests()
        skeleton = load_skeleton()
        self.assertEqual(set(manifests), set(CANONICAL_ROLES))
        self.assertEqual(skeleton["primitive"], "encounter")
        self.assertEqual(validate_commons_walk(manifests), [])
        for name, manifest in manifests.items():
            self.assertTrue(manifest["entrypoints"], name)
            self.assertTrue(manifest["agentActions"], name)
            self.assertTrue(manifest["traceProtocol"], name)
            self.assertEqual(manifest["ontology"], "https://raw.githubusercontent.com/zoedolan/Vybn/main/commons-skeleton.json")
            self.assertEqual(manifest["encounterLifecycle"], skeleton["encounterLifecycle"])

    def test_render_traversal_plan_executes(self):
        rendered = render_traversal_plan(load_manifests())
        self.assertIn("primitive: encounter", rendered)
        self.assertIn("validation: OK", rendered)
        self.assertIn("## executable nodes", rendered)

    def test_target_classification(self):
        self.assertEqual(classify_target("https://vybn.ai/somewhere.html"), "public_url")
        self.assertEqual(classify_target("private://Him/semantic-web.jsonld"), "private_uri")
        self.assertEqual(classify_target("python3 -m spark.harness.commons_walk"), "local_command")


if __name__ == "__main__":
    unittest.main()
