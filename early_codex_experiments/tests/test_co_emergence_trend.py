import unittest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.co_emergence import compute_trend

class TestCoEmergenceTrend(unittest.TestCase):
    def test_compute_trend(self):
        now = datetime.utcnow()
        entries = [
            {'timestamp': (now - timedelta(seconds=10)).isoformat() + 'Z', 'score': 1.0},
            {'timestamp': now.isoformat() + 'Z', 'score': 3.0},
        ]
        slope = compute_trend(entries)
        self.assertAlmostEqual(slope, 0.2, places=6)

if __name__ == '__main__':
    unittest.main()
