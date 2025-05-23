import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.cognitive_structures.graph_summary import graph_stats


class TestGraphSummary(unittest.TestCase):
    def test_graph_stats(self):
        graph = {
            'memory_nodes': [1, 2],
            'memoir_nodes': [3],
            'repo_nodes': ['a', 'b', 'c'],
            'edges': [
                {'source': 1, 'target': 'a'},
                {'source': 2, 'target': 'b'},
            ],
        }
        expected = {
            'memory_nodes': 2,
            'memoir_nodes': 1,
            'repo_nodes': 3,
            'edges': 2,
        }
        self.assertEqual(graph_stats(graph), expected)


if __name__ == '__main__':
    unittest.main()
