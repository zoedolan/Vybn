import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.cognitive_structures.parallel_metacog import parallel_coherence


class TestParallelMetacog(unittest.TestCase):
    def test_parallel_coherence(self):
        res = parallel_coherence()
        self.assertEqual(res["combined"], "fast intuition | structured reflection")


if __name__ == "__main__":
    unittest.main()
