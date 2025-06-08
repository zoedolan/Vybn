import unittest
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vybn.co_emergence import log_score


class TestEmergenceJournal(unittest.TestCase):
    def test_log_score(self):
        graph = {
            'memory_nodes': [1],
            'memoir_nodes': [],
            'repo_nodes': ['a', 'b'],
            'edges': [{'source': 1, 'target': 'a'}],
        }
        gpath = 'tmp_graph.json'
        jpath = 'tmp_journal.jsonl'
        with open(gpath, 'w') as f:
            json.dump(graph, f)
        if os.path.exists(jpath):
            os.remove(jpath)
        entry = log_score(gpath, jpath)
        with open(jpath, 'r') as f:
            line = f.readline()
        os.remove(gpath)
        os.remove(jpath)
        logged = json.loads(line)
        self.assertEqual(logged['score'], entry['score'])
        self.assertIn('timestamp', logged)


if __name__ == '__main__':
    unittest.main()
