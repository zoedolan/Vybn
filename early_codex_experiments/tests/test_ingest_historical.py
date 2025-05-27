import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import types

# Provide a lightweight stub for the cache pipeline to avoid heavy dependencies
cache_stub = types.ModuleType('early_codex_experiments.cache_pipeline')

class Shard:
    def __init__(self, timestamp, source, text):
        self.timestamp = timestamp
        self.source = source
        self.text = text
        self.summary = ''
        self.heat = 1.0

    def dict(self):
        return {
            'timestamp': self.timestamp,
            'source': self.source,
            'text': self.text,
            'summary': self.summary,
            'heat': self.heat,
        }

class TorusCache:
    def __init__(self, *args, **kwargs):
        pass

cache_stub.Shard = Shard
cache_stub.TorusCache = TorusCache
sys.modules['early_codex_experiments.cache_pipeline'] = cache_stub

import ingest_historical

class DummyCol:
    def __init__(self):
        self.calls = []
    def add(self, ids=None, embeddings=None, documents=None, metadatas=None):
        self.calls.append({
            'ids': ids,
            'embeddings': embeddings,
            'documents': documents,
            'metadatas': metadatas,
        })

class DummyCache:
    def __init__(self):
        self.ingested = []
        self.records = []
        self._col = DummyCol()
    def ingest(self, shards):
        self.ingested.extend(shards)
    def _append_ledger(self, record):
        self.records.append(record)

class TestIngestHistorical(unittest.TestCase):
    def test_already_ingested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = os.path.join(tmpdir, 'ledger.log')
            lines = [
                json.dumps({'phase': 'bootstrap', 'source': 'a.txt'}),
                json.dumps({'phase': 'ignore', 'source': 'b.txt'}),
                'not json',
                json.dumps({'phase': 'bootstrap', 'source': 'c.md'}),
            ]
            with open(ledger, 'w', encoding='utf-8') as f:
                for l in lines:
                    f.write(l + '\n')
            result = ingest_historical.already_ingested(Path(ledger))
            self.assertEqual(result, {'a.txt', 'c.md'})

    def test_ingest_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hist_dir = os.path.join(tmpdir, 'hist')
            os.mkdir(hist_dir)
            files = {'a.txt': 'hello', 'b.md': 'world', 'c.jpg': 'img'}
            for name, text in files.items():
                with open(os.path.join(hist_dir, name), 'w') as f:
                    f.write(text)
            ingest_historical.HISTORY_DIR = Path(hist_dir)
            cache = DummyCache()
            shards = ingest_historical.ingest_files(cache, set())
            self.assertEqual(len(shards), 2)
            ingested_sources = {s.source.split('/')[-1] for s in cache.ingested}
            self.assertEqual(ingested_sources, {'a.txt', 'b.md'})
            self.assertEqual(len(cache.records), 2)
            for rec in cache.records:
                self.assertIn('corpus_id', rec)
                self.assertEqual(rec['phase'], 'bootstrap')

    def test_ingest_centroids(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmpdir:
            mind_dir = os.path.join(tmpdir, 'mind')
            os.mkdir(mind_dir)
            arr = np.array([[1.0, 0.0], [0.0, 1.0]])
            np.save(os.path.join(mind_dir, 'concept_centroids.npy'), arr)
            ingest_historical.MIND_VIZ_DIR = Path(mind_dir)
            cache = DummyCache()
            ingest_historical.ingest_centroids(cache)
            self.assertEqual(len(cache._col.calls), 1)
            call = cache._col.calls[0]
            self.assertEqual(call['ids'], ['centroid_0', 'centroid_1'])
            self.assertEqual(len(cache.records), 2)

if __name__ == '__main__':
    unittest.main()
