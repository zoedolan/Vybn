import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.cognitive_structures.graph_centrality import compute_degree_centrality

class TestGraphCentrality(unittest.TestCase):
    def test_compute_degree_centrality(self):
        graph = {
            'edges': [
                {'source': 'A', 'target': 'B'},
                {'source': 'B', 'target': 'C'}
            ]
        }
        centrality = compute_degree_centrality(graph)
        expected = {'A': 1, 'B': 2, 'C': 1}
        self.assertEqual(centrality, expected)

if __name__ == '__main__':
    unittest.main()
