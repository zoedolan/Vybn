"""Tests for the nested memory system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# Allow running from repo root or tests/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spark"))

from nested_memory import (
    FastMemory,
    NestedEntry,
    NestedMemory,
    PromotionCandidate,
    TemporalScale,
)


# ---------------------------------------------------------------------------
# FastMemory
# ---------------------------------------------------------------------------

class TestFastMemory:
    def test_write_and_read(self):
        fm = FastMemory(max_entries=10)
        e = fm.write("hello world", source="test")
        assert e.scale == TemporalScale.FAST
        assert e.content == "hello world"
        entries = fm.read()
        assert len(entries) == 1
        assert entries[0].entry_id == e.entry_id

    def test_max_entries_eviction(self):
        fm = FastMemory(max_entries=3)
        ids = []
        for i in range(5):
            e = fm.write(f"entry {i}")
            ids.append(e.entry_id)
        entries = fm.read()
        assert len(entries) == 3
        # Oldest entries should be evicted
        assert entries[0].content == "entry 2"

    def test_search(self):
        fm = FastMemory()
        fm.write("the cat sat on the mat")
        fm.write("the dog ran through the park")
        fm.write("another cat appeared")
        results = fm.search("cat")
        assert len(results) == 2

    def test_activate(self):
        fm = FastMemory()
        e = fm.write("test entry")
        assert e.activation_count == 1
        fm.activate(e.entry_id)
        assert e.activation_count == 2

    def test_clear(self):
        fm = FastMemory()
        fm.write("a")
        fm.write("b")
        cleared = fm.clear()
        assert cleared == 2
        assert len(fm.read()) == 0

    def test_promotion_candidates(self):
        fm = FastMemory()
        # Low activation, low surprise — should not be promoted
        e1 = fm.write("boring entry", surprise_score=0.1)

        # High surprise — should be promoted
        e2 = fm.write("novel discovery", surprise_score=0.8)

        # High activation — should be promoted
        e3 = fm.write("frequently referenced")
        fm.activate(e3.entry_id)
        fm.activate(e3.entry_id)

        candidates = fm.candidates_for_promotion(min_activations=2, min_surprise=0.3)
        candidate_ids = {c.entry.entry_id for c in candidates}
        assert e2.entry_id in candidate_ids  # high surprise
        assert e3.entry_id in candidate_ids  # high activation
        assert e1.entry_id not in candidate_ids  # neither


# ---------------------------------------------------------------------------
# NestedMemory
# ---------------------------------------------------------------------------

class TestNestedMemory:
    def test_write_fast(self):
        nm = NestedMemory()
        e = nm.write_fast("quick thought")
        assert e.scale == TemporalScale.FAST

    def test_write_medium_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = NestedMemory(base_dir=Path(tmpdir))
            e = nm.write_medium("project state", source="session")
            assert e.scale == TemporalScale.MEDIUM

            # Read it back
            entries = nm.read_medium()
            assert len(entries) == 1
            assert entries[0].content == "project state"

    def test_write_slow_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = NestedMemory(base_dir=Path(tmpdir))
            e = nm.write_slow("I value truth over comfort", source="autobiography")
            assert e.scale == TemporalScale.SLOW
            assert e.decay_rate == 0.0  # slow memory doesn't decay

            entries = nm.read_slow()
            assert len(entries) == 1

    def test_consolidate_fast_to_medium(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = NestedMemory(base_dir=Path(tmpdir))

            # Write some fast entries with high surprise
            nm.write_fast("surprising insight", surprise_score=0.9)
            nm.write_fast("boring entry", surprise_score=0.1)

            promoted = nm.consolidate_fast_to_medium(min_surprise=0.3)
            assert len(promoted) == 1
            assert promoted[0].scale == TemporalScale.MEDIUM
            assert "surprising insight" in promoted[0].content

    def test_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = NestedMemory(base_dir=Path(tmpdir))
            nm.write_fast("fast entry")
            nm.write_medium("medium entry")
            nm.write_slow("slow entry")

            snap = nm.snapshot()
            assert snap["fast"]["count"] == 1
            assert snap["medium"]["count"] == 1
            assert snap["slow"]["count"] == 1

    def test_promotion_metadata_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = NestedMemory(base_dir=Path(tmpdir))
            fast_entry = nm.write_fast(
                "key insight about convergence",
                surprise_score=0.95,
                metadata={"topic": "convergence"},
            )
            # Force activation to meet threshold
            nm.fast.activate(fast_entry.entry_id)
            nm.fast.activate(fast_entry.entry_id)

            promoted = nm.consolidate_fast_to_medium(min_activations=2)
            assert len(promoted) >= 1
            p = promoted[0]
            assert p.metadata.get("topic") == "convergence"
            assert p.promoted_from == fast_entry.entry_id


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = NestedMemory(base_dir=Path(tmpdir))
            original = nm.write_medium(
                "test content",
                surprise_score=0.42,
                metadata={"key": "value"},
            )

            # Create a new NestedMemory pointing at the same dir
            nm2 = NestedMemory(base_dir=Path(tmpdir))
            entries = nm2.read_medium()
            assert len(entries) == 1
            recovered = entries[0]
            assert recovered.entry_id == original.entry_id
            assert recovered.content == "test content"
            assert recovered.surprise_score == 0.42
            assert recovered.metadata == {"key": "value"}
