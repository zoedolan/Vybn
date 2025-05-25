import os
import json
import unittest

class TestMindVisualization(unittest.TestCase):
    def test_overlay_map_exists_and_fields(self):
        overlay_path = os.path.join('Mind Visualization', 'overlay_map.jsonl')
        self.assertTrue(os.path.exists(overlay_path))
        with open(overlay_path, 'r', encoding='utf-8') as f:
            line = f.readline()
        self.assertTrue(line)
        rec = json.loads(line)
        for key in ('p', 'f', 's', 'e', 'o'):
            self.assertIn(key, rec)

    def test_concept_map_exists_and_fields(self):
        concept_path = os.path.join('Mind Visualization', 'concept_map.jsonl')
        self.assertTrue(os.path.exists(concept_path))
        with open(concept_path, 'r', encoding='utf-8') as f:
            line = f.readline()
        self.assertTrue(line)
        rec = json.loads(line)
        for key in ('w', 'c', 'f', 's', 'e'):
            self.assertIn(key, rec)

if __name__ == '__main__':
    unittest.main()
