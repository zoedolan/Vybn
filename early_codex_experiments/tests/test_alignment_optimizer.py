import unittest
import numpy as np
from scripts.cognitive_structures import alignment_optimizer as ao

class TestAlignmentOptimizer(unittest.TestCase):
    def test_embed_text_offline(self):
        ao.openai = None
        vec = ao.embed_text("hello", dim=ao.EMBED_DIM)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (ao.EMBED_DIM,))

if __name__ == '__main__':
    unittest.main()
