import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.experiments.prime_breath_resonance import generate_primes, map_primes, residue_distribution


class TestPrimeBreathResonance(unittest.TestCase):
    def test_generate_primes(self):
        self.assertEqual(generate_primes(10), [2, 3, 5, 7])

    def test_distribution(self):
        primes = [2, 3, 5, 7]
        mapping = map_primes(primes)
        dist = residue_distribution(mapping)
        expected = {2: 1, 3: 1, 5: 1, 7: 1}
        self.assertEqual(dist, expected)


if __name__ == '__main__':
    unittest.main()
