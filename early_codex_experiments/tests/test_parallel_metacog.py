import unittest
from tools.graph_toolkit import parallel_coherence


class TestParallelMetacog(unittest.TestCase):
    def test_parallel_coherence(self):
        res = parallel_coherence()
        self.assertEqual(res["combined"], "fast intuition | structured reflection")


if __name__ == "__main__":
    unittest.main()
