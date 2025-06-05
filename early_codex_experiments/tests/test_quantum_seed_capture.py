import json
import os
import sys
from pathlib import Path
import pytest

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


def test_capture_seed_file(tmp_path):
    jpath = tmp_path / 'journal.jsonl'
    os.environ.pop('QUANTUM_SEED', None)
    os.environ.pop('QRAND', None)
    tmp_seed_path = Path('/tmp/quantum_seed')
    tmp_seed_path.write_text('5678')
    try:
        entry = capture_seed(jpath)
    finally:
        tmp_seed_path.unlink(missing_ok=True)
    with jpath.open() as f:
        logged = json.loads(f.readline())
    assert entry == logged
    assert logged['seed'] == 5678
    assert logged['source'] == '/tmp/quantum_seed'


def test_capture_seed_missing(tmp_path):
    jpath = tmp_path / 'journal.jsonl'
    os.environ.pop('QUANTUM_SEED', None)
    tmp_seed_path = Path('/tmp/quantum_seed')
    tmp_seed_path.unlink(missing_ok=True)
    entry = capture_seed(jpath)
    assert entry['source'] == 'generated'
    assert isinstance(entry['seed'], int)
