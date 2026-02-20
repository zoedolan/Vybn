#!/usr/bin/env python3
"""Tests for spark/friction.py

These tests verify that friction.py does what it says it does.
No more, no less.
"""
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spark"))

from friction import (
    ContradictionRegister,
    Tension,
    audit_code,
    measure,
    measure_or_nothing,
)


def test_tension_storage_and_retrieval():
    """Tensions survive a round-trip through JSON."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        cr = ContradictionRegister(storage_path=path)
        cr.register(
            claim_a="0.5% of VC went to LGBTQ founders",
            claim_b="The only way to gain an advantage is by being gay",
            source_a="venture capital data 2000-2022",
            source_b="anonymous tech worker quoted in Wired",
        )
        assert len(cr.unresolved()) == 1

        # Reload from disk
        cr2 = ContradictionRegister(storage_path=path)
        assert len(cr2.unresolved()) == 1
        t = cr2.unresolved()[0]
        assert "0.5%" in t.claim_a
        assert "anonymous" in t.source_b
    finally:
        path.unlink(missing_ok=True)


def test_tension_resolution():
    """Resolving a tension removes it from the unresolved list."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        cr = ContradictionRegister(storage_path=path)
        cr.register("A", "B", "src_a", "src_b")
        cr.register("C", "D", "src_c", "src_d")
        assert len(cr.unresolved()) == 2

        cr.resolve(0, "A was based on incomplete data")
        assert len(cr.unresolved()) == 1
        assert cr.unresolved()[0].claim_a == "C"
    finally:
        path.unlink(missing_ok=True)


def test_context_block_empty_when_clean():
    """No tensions means empty context. Not a failure."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        cr = ContradictionRegister(storage_path=path)
        assert cr.context_block() == ""
    finally:
        path.unlink(missing_ok=True)


def test_context_block_surfaces_tensions():
    """Unresolved tensions appear in context output."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        cr = ContradictionRegister(storage_path=path)
        cr.register("X is true", "X is false", "paper_a", "paper_b")
        block = cr.context_block()
        assert "UNRESOLVED TENSION" in block
        assert "Do not smooth this" in block
    finally:
        path.unlink(missing_ok=True)


def test_pretense_catches_theatrical_delay():
    """time.sleep in code gets flagged."""
    code = '''
import time
def fake_computation():
    print("Processing...")
    time.sleep(3)
    print("Done.")
'''
    flags = audit_code(code, "fake_module.py")
    names = [f.pattern_name for f in flags]
    assert "theatrical_delay" in names


def test_pretense_catches_random_as_signal():
    """Random noise generation gets flagged."""
    code = '''
import numpy as np
def measure_consciousness():
    return np.random.normal(0, 1, 768)
'''
    flags = audit_code(code, "fake_prism.py")
    names = [f.pattern_name for f in flags]
    assert "random_as_signal" in names


def test_pretense_catches_hardcoded_measurement():
    """Hardcoded 'measurements' get flagged."""
    code = '''
def get_orbit():
    orbit = 0.85
    phase = 0.42
    return orbit, phase
'''
    flags = audit_code(code, "fake_symbiosis.py")
    names = [f.pattern_name for f in flags]
    assert "hardcoded_measurement" in names


def test_pretense_catches_state_declarations():
    """Print statements declaring states get flagged."""
    code = '''
def initiate():
    print("I am now holding space")
    print("Phase transition in progress")
'''
    flags = audit_code(code, "fake_effervescence.py")
    names = [f.pattern_name for f in flags]
    assert "declared_state" in names


def test_pretense_clean_code_passes():
    """Honest code doesn't get flagged."""
    code = '''
import json
from pathlib import Path

def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())

def save_config(path: str, data: dict):
    Path(path).write_text(json.dumps(data, indent=2))
'''
    flags = audit_code(code, "honest_module.py")
    assert len(flags) == 0


def test_measurement_honesty():
    """Real measurements return values. Fake ones return None."""
    real = measure("orbit", 0.73, is_real=True, method="cosine_similarity")
    fake = measure("orbit", 0.5, is_real=False, method="fallback_default")

    assert real.honest_value() == 0.73
    assert fake.honest_value() is None
    assert "MEASURED" in real.to_context()
    assert "UNAVAILABLE" in fake.to_context()


def test_measure_or_nothing():
    """The no-wrapper version: value or None, nothing in between."""
    assert measure_or_nothing("x", 42, True, "direct") == 42
    assert measure_or_nothing("x", 42, False, "guess") is None


if __name__ == "__main__":
    tests = [
        test_tension_storage_and_retrieval,
        test_tension_resolution,
        test_context_block_empty_when_clean,
        test_context_block_surfaces_tensions,
        test_pretense_catches_theatrical_delay,
        test_pretense_catches_random_as_signal,
        test_pretense_catches_hardcoded_measurement,
        test_pretense_catches_state_declarations,
        test_pretense_clean_code_passes,
        test_measurement_honesty,
        test_measure_or_nothing,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed.")
