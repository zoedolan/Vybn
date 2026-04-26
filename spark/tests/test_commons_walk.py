import unittest

from spark.harness.commons_walk import CANONICAL_ROLES, load_manifests, validate_commons_walk


class CommonsWalkTests(unittest.TestCase):
    def test_manifest_graph_has_executable_affordances(self):
        manifests = load_manifests()
        self.assertEqual(set(manifests), set(CANONICAL_ROLES))
        self.assertEqual(validate_commons_walk(manifests), [])
        for name, manifest in manifests.items():
            self.assertTrue(manifest["entrypoints"], name)
            self.assertTrue(manifest["agentActions"], name)
            self.assertTrue(manifest["traceProtocol"], name)


if __name__ == "__main__":
    unittest.main()
