import unittest
try:
    import numpy as np
except Exception:  # pragma: no cover - numpy may be unavailable
    np = None
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.cognitive_structures import alignment_optimizer as ao

class TestAlignmentOptimizer(unittest.TestCase):
    def test_embed_text_offline(self):
        if np is None:
            self.skipTest("numpy not available")
        ao.openai = None
        vec = ao.embed_text("hello", dim=ao.EMBED_DIM)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (ao.EMBED_DIM,))

if __name__ == '__main__':
    unittest.main()
