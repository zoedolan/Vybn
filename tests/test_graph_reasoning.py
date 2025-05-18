import json
import unittest
import os
from cognitive_structures.graph_reasoning import find_path

class TestGraphReasoning(unittest.TestCase):
    def test_find_simple_path(self):
        graph = {
            'memory_nodes': [],
            'memoir_nodes': [],
            'repo_nodes': ['nodeA', 'nodeB', 'nodeC'],
            'edges': [
                {'source': 'nodeA', 'target': 'nodeB'},
                {'source': 'nodeB', 'target': 'nodeC'},
            ]
        }
        path_file = 'tmp_graph.json'
        with open(path_file, 'w') as f:
            json.dump(graph, f)
        path = find_path(path_file, 'nodeA', 'nodeC')
        os.remove(path_file)
        self.assertEqual(path, ['nodeA', 'nodeB', 'nodeC'])

if __name__ == '__main__':
    unittest.main()
