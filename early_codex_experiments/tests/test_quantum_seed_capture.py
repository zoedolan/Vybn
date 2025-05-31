import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.quantum_seed_capture import capture_seed


def test_capture_seed_env(tmp_path):
    jpath = tmp_path / 'journal.jsonl'
    os.environ['QUANTUM_SEED'] = '1234'
    entry = capture_seed(jpath)
    with jpath.open() as f:
        logged = json.loads(f.readline())
    assert entry == logged
    assert logged['seed'] == 1234
    assert logged['source'] == 'QUANTUM_SEED'
