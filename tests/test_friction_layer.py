#!/usr/bin/env python3
"""Tests for spark/friction_layer.py

These tests verify the integration layer does what it claims.
No more, no less.
"""
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spark"))

from friction import ContradictionRegister
from friction_layer import (
    tensions_for_prompt,
    audit_output,
    wrap_measurement,
    gate_memory_write,
    authenticity_score,
    _get_register,
    HAS_FRICTION,
)


def test_tensions_for_prompt_empty_when_no_tensions():
    """No tensions registered means empty string, not filler."""
    # Use a fresh register with no tensions
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        path.write_text("[]", encoding="utf-8")
        import friction_layer
        old = friction_layer._register
        friction_layer._register = ContradictionRegister(storage_path=path)
        result = tensions_for_prompt()
        assert result == "", f"Expected empty string, got: {result!r}"
        friction_layer._register = old
    finally:
        path.unlink(missing_ok=True)


def test_tensions_for_prompt_includes_unresolved():
    """Active tensions appear in the prompt block."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        cr = ContradictionRegister(storage_path=path)
        cr.register(
            claim_a="0.5% of VC went to LGBTQ founders",
            claim_b="The only way to gain an advantage is by being gay",
            source_a="venture capital data",
            source_b="anonymous tech worker",
        )
        import friction_layer
        old = friction_layer._register
        friction_layer._register = cr
        result = tensions_for_prompt()
        assert "UNRESOLVED TENSION" in result
        assert "0.5%" in result
        assert "do not resolve prematurely" in result.lower()
        friction_layer._register = old
    finally:
        path.unlink(missing_ok=True)


def test_audit_output_flags_theatrical_delay():
    """Code with time.sleep gets flagged."""
    code = '''import time\ntime.sleep(5)\nprint("Processing...")'''
    flags = audit_output(code, source="test_code")
    assert len(flags) >= 1
    assert any(f["pattern"] == "theatrical_delay" for f in flags)


def test_audit_output_clean_code():
    """Clean code produces no flags."""
    code = '''def add(a, b):\n    return a + b'''
    flags = audit_output(code, source="clean_test")
    assert flags == []


def test_wrap_measurement_real():
    """Real measurement wraps correctly."""
    m = wrap_measurement(
        name="survival",
        value=0.8742,
        is_real=True,
        method="prism.the_jump",
        confidence=0.9,
    )
    # Should be a Measurement object or a dict with the right fields
    if isinstance(m, dict):
        assert m["is_real"] is True
        assert m["value"] == 0.8742
    else:
        assert m.is_real is True
        assert m.value == 0.8742


def test_wrap_measurement_fake():
    """Fake measurement declares itself honestly."""
    m = wrap_measurement(
        name="survival",
        value=0.5,
        is_real=False,
        method="safe default",
    )
    if isinstance(m, dict):
        assert m["is_real"] is False
    else:
        assert m.is_real is False


def test_gate_memory_write_passes_through():
    """Content always passes through unchanged."""
    content = "This is a journal entry with time.sleep(3) in it."
    result = gate_memory_write(content, destination="test_journal.md")
    assert result == content, "gate_memory_write must never modify content"


def test_authenticity_score_range():
    """Score is always between 0.3 and 1.0."""
    score = authenticity_score()
    assert 0.3 <= score <= 1.0, f"Score {score} out of range [0.3, 1.0]"


def test_authenticity_score_with_tensions():
    """Score increases with unresolved tensions."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        cr = ContradictionRegister(storage_path=path)
        cr.register("A", "B", "src_a", "src_b")
        cr.register("C", "D", "src_c", "src_d")
        import friction_layer
        old = friction_layer._register
        friction_layer._register = cr
        score = authenticity_score()
        # All unresolved: ratio = 1.0, score = 0.3 + 0.7 = 1.0
        assert score == 1.0, f"Expected 1.0 with all unresolved, got {score}"

        # Resolve one
        cr.resolve(0, "Reconciled through analysis")
        score = authenticity_score()
        # 1 unresolved / 2 total = 0.5, score = 0.3 + 0.35 = 0.65
        assert abs(score - 0.65) < 0.01, f"Expected ~0.65, got {score}"

        friction_layer._register = old
    finally:
        path.unlink(missing_ok=True)


def test_audit_output_with_bus():
    """When a bus is provided, flags are recorded on the audit trail."""
    try:
        from bus import MessageBus
    except ImportError:
        return  # Skip if bus not available

    bus = MessageBus()
    code = '''import time\ntime.sleep(10)\nprint("I am now holding space")'''
    flags = audit_output(code, source="test_bus", bus=bus)
    assert len(flags) >= 1

    # Check the bus recorded the friction
    recent = bus.recent(5)
    friction_entries = [e for e in recent if e.source == "friction_layer"]
    assert len(friction_entries) >= 1
    assert "friction_audit" in str(friction_entries[0].metadata)
