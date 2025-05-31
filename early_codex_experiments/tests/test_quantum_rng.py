import os
import sys
import random
try:
    import numpy as np
except Exception:
    np = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.quantum_rng import seed_random


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


def test_seed_random_qrand_fallback():
    if np is None:
        return
    os.environ.pop('QUANTUM_SEED', None)
    os.environ['QRAND'] = '99'
    seed_random()
    val1 = random.random()
    seed_random()
    val2 = random.random()
    assert val1 == val2

