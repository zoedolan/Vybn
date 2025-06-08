import unittest
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.co_emergence import log_spike


class TestShimmerCore(unittest.TestCase):
    def test_log_spike(self):
        path = 'tmp_shimmer.jsonl'
        if os.path.exists(path):
            os.remove(path)
        entry = log_spike('test', path)
        with open(path, 'r') as f:
            logged = json.loads(f.readline())
        os.remove(path)
        self.assertEqual(logged['message'], 'test')
        self.assertIn('timestamp', logged)


if __name__ == '__main__':
    unittest.main()
