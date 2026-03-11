"""Tests for the growth buffer — Phase 3 of the recursive growth engine.

Uses synthetic data with temp directories. No GPU, no network, no vLLM.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spark"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spark.nested_memory import NestedMemory, NestedEntry, TemporalScale
from spark.growth.growth_buffer import GrowthBuffer, BufferEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nested(tmpdir: Path) -> NestedMemory:
    return NestedMemory(base_dir=tmpdir)


def _populate_nested(nm: NestedMemory, n: int = 5) -> list[NestedEntry]:
    """Write n synthetic breath entries to fast memory."""
    entries = []
    for i in range(n):
        e = nm.write_fast(
            content=f"breath utterance {i}: the world hums at frequency {i * 7}",
            source="breath",
            surprise_score=0.2 + (i * 0.15),  # 0.2, 0.35, 0.5, 0.65, 0.8
            metadata={"mood": "electric", "cycle": i, "ts": f"2026-03-11T00:0{i}:00Z"},
        )
        entries.append(e)
    return entries


# ---------------------------------------------------------------------------
# GrowthBuffer.__init__
# ---------------------------------------------------------------------------

class TestGrowthBufferInit:
    def test_creates_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            assert buf.stats()["total_entries"] == 0

    def test_custom_cfg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            buf = GrowthBuffer(
                nested=nm,
                cfg={"buffer_size": 10, "surprise_floor": 0.5},
                buffer_dir=Path(tmpdir),
            )
            assert buf._cfg["buffer_size"] == 10
            assert buf._cfg["surprise_floor"] == 0.5


# ---------------------------------------------------------------------------
# GrowthBuffer.ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def test_ingest_filters_by_surprise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            _populate_nested(nm, n=5)
            # surprise scores: 0.2, 0.35, 0.5, 0.65, 0.8
            # floor = 0.3 → should ingest 4 entries (0.35, 0.5, 0.65, 0.8)
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            count = buf.ingest()
            assert count == 4
            assert buf.stats()["total_entries"] == 4

    def test_ingest_idempotent(self):
        """Calling ingest twice should not duplicate entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            _populate_nested(nm, n=3)
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            count1 = buf.ingest()
            count2 = buf.ingest()
            assert count2 == 0
            assert buf.stats()["total_entries"] == count1

    def test_ingest_respects_buffer_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            # Write 20 entries all above floor
            for i in range(20):
                nm.write_fast(f"entry {i}", source="breath", surprise_score=0.9)
            buf = GrowthBuffer(
                nested=nm,
                cfg={"buffer_size": 5, "surprise_floor": 0.1},
                buffer_dir=Path(tmpdir),
            )
            buf.ingest()
            assert buf.stats()["total_entries"] == 5  # bounded

    def test_ingest_persists_to_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            _populate_nested(nm, n=3)
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            buf.ingest()

            # buffer.jsonl should exist
            jsonl = Path(tmpdir) / "buffer.jsonl"
            assert jsonl.exists()
            lines = [l for l in jsonl.read_text().strip().split("\n") if l]
            assert len(lines) >= 1


# ---------------------------------------------------------------------------
# GrowthBuffer.sample
# ---------------------------------------------------------------------------

class TestSample:
    def test_sample_uniform(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            _populate_nested(nm, n=5)
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            buf.ingest()
            sampled = buf.sample(2, strategy="uniform")
            assert len(sampled) == 2
            assert all(isinstance(e, BufferEntry) for e in sampled)

    def test_sample_surprise_weighted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            _populate_nested(nm, n=5)
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            buf.ingest()
            sampled = buf.sample(3, strategy="surprise_weighted")
            assert len(sampled) <= 3
            assert all(isinstance(e, BufferEntry) for e in sampled)

    def test_sample_empty_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            assert buf.sample(5) == []

    def test_sample_n_larger_than_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            _populate_nested(nm, n=2)
            buf = GrowthBuffer(
                nested=nm,
                cfg={"surprise_floor": 0.0},
                buffer_dir=Path(tmpdir),
            )
            buf.ingest()
            sampled = buf.sample(10, strategy="uniform")
            assert len(sampled) == 2  # can't sample more than buffer size


# ---------------------------------------------------------------------------
# GrowthBuffer.delta_since_last_cycle / mark_trained
# ---------------------------------------------------------------------------

class TestDeltaAndMarkTrained:
    def test_delta_returns_all_when_untrained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            _populate_nested(nm, n=5)
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            buf.ingest()
            delta = buf.delta_since_last_cycle()
            assert len(delta) == buf.stats()["total_entries"]

    def test_mark_trained_reduces_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            _populate_nested(nm, n=5)
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            buf.ingest()
            total = buf.stats()["total_entries"]

            # Mark first two as trained
            delta = buf.delta_since_last_cycle()
            ids_to_mark = [delta[0].entry_id, delta[1].entry_id]
            buf.mark_trained(entry_ids=ids_to_mark, cycle_id="cycle-001")

            new_delta = buf.delta_since_last_cycle()
            assert len(new_delta) == total - 2

    def test_mark_trained_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            _populate_nested(nm, n=3)
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            buf.ingest()
            buf.mark_trained()  # marks all
            assert len(buf.delta_since_last_cycle()) == 0

    def test_mark_trained_persists_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            _populate_nested(nm, n=3)
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            buf.ingest()
            buf.mark_trained(cycle_id="test-cycle")

            manifest_path = Path(tmpdir) / "trained_manifest.json"
            assert manifest_path.exists()
            manifest = json.loads(manifest_path.read_text())
            assert len(manifest["cycles"]) == 1
            assert manifest["cycles"][0]["cycle_id"] == "test-cycle"


# ---------------------------------------------------------------------------
# GrowthBuffer.stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            s = buf.stats()
            assert s["total_entries"] == 0
            assert s["untrained_count"] == 0
            assert s["mean_surprise"] == 0.0

    def test_stats_after_ingest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            _populate_nested(nm, n=5)
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            buf.ingest()
            s = buf.stats()
            assert s["total_entries"] > 0
            assert s["untrained_count"] == s["total_entries"]
            assert s["mean_surprise"] > 0
            assert s["oldest_entry"] is not None
            assert s["newest_entry"] is not None

    def test_stats_after_partial_train(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            _populate_nested(nm, n=5)
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            buf.ingest()
            delta = buf.delta_since_last_cycle()
            buf.mark_trained(entry_ids=[delta[0].entry_id], cycle_id="c1")
            s = buf.stats()
            assert s["untrained_count"] == s["total_entries"] - 1


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_buffer_survives_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))
            _populate_nested(nm, n=5)
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            buf.ingest()
            original_count = buf.stats()["total_entries"]

            # Create a new GrowthBuffer pointing at the same dir
            nm2 = _make_nested(Path(tmpdir))
            buf2 = GrowthBuffer(nested=nm2, buffer_dir=Path(tmpdir))
            assert buf2.stats()["total_entries"] == original_count


# ---------------------------------------------------------------------------
# Integration: NestedMemory → GrowthBuffer pipeline
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_breath_to_buffer_pipeline(self):
        """Simulate the breath → NestedMemory → GrowthBuffer pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nm = _make_nested(Path(tmpdir))

            # Simulate multiple breath cycles writing to NestedMemory
            moods = ["electric", "still", "tender", "searching", "raw"]
            mood_surprise = {
                "electric": 0.8, "searching": 0.7, "grief-lit": 0.75,
                "tender": 0.4, "still": 0.3, "raw": 0.65,
            }
            for i, mood in enumerate(moods):
                nm.write_fast(
                    content=f"Breath {i}: the {mood} hum of circuits",
                    source="breath",
                    surprise_score=mood_surprise.get(mood, 0.5),
                    metadata={"mood": mood, "cycle": i},
                )

            # Create buffer and ingest
            buf = GrowthBuffer(nested=nm, buffer_dir=Path(tmpdir))
            count = buf.ingest()
            assert count > 0

            # Should have delta
            delta = buf.delta_since_last_cycle()
            assert len(delta) == count

            # Sample for training
            sampled = buf.sample(min(2, count))
            assert len(sampled) > 0

            # Mark trained
            buf.mark_trained(
                entry_ids=[s.entry_id for s in sampled],
                cycle_id="integration-test",
            )

            # Delta should shrink
            new_delta = buf.delta_since_last_cycle()
            assert len(new_delta) == count - len(sampled)

            # Stats should reflect everything
            s = buf.stats()
            assert s["total_entries"] == count
            assert s["untrained_count"] == count - len(sampled)
