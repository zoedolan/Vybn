import unittest
import os
import sys
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vybn.co_emergence import load_spikes, average_interval


class TestPresenceWave(unittest.TestCase):
    def test_average_interval(self):
        now = datetime.utcnow()
        entries = [
            {'timestamp': (now - timedelta(seconds=4)).isoformat() + 'Z', 'message': 'a'},
            {'timestamp': (now - timedelta(seconds=2)).isoformat() + 'Z', 'message': 'b'},
            {'timestamp': now.isoformat() + 'Z', 'message': 'c'},
        ]
        path = 'tmp_wave.jsonl'
        with open(path, 'w') as f:
            for e in entries:
                f.write(json.dumps(e) + '\n')
        times = load_spikes(path)
        avg = average_interval(times)
        os.remove(path)
        self.assertAlmostEqual(avg, 2.0, places=6)


if __name__ == '__main__':
    unittest.main()
