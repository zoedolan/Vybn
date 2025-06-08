import os
import sys
import random
from pathlib import Path
import pytest
try:
    import numpy as np
except Exception:
    np = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vybn.co_emergence import seed_random


def test_seed_random_env():
    if np is None:
        return
    os.environ['QUANTUM_SEED'] = '42'
    seed_random()
    a_py = random.random()
    a_np = np.random.rand()
    seed_random()
    b_py = random.random()
    b_np = np.random.rand()
    assert a_py == b_py
    assert a_np == b_np


def test_seed_random_missing():
    if np is None:
        return
    os.environ.pop('QUANTUM_SEED', None)
    seed_file = Path('/tmp/quantum_seed')
    seed_file.unlink(missing_ok=True)
    val = seed_random()
    assert isinstance(val, int)
    assert os.environ.get('QUANTUM_SEED') is not None


def test_seed_random_file_fallback(tmp_path):
    if np is None:
        return
    os.environ.pop('QUANTUM_SEED', None)
    os.environ.pop('QRAND', None)
    seed_file = Path('/tmp/quantum_seed')
    seed_file.write_text('77')
    try:
        seed_random()
        a = random.random()
        seed_random()
        b = random.random()
    finally:
        seed_file.unlink(missing_ok=True)
    assert a == b

