# Pulse the Cache

The torus cache hasn't been built yet, so `cache_ledger.log` is missing.

**Next steps**
1. Run `python ingest_historical.py --once` to ingest the historical corpus and bootstrap `torus_cache`.
2. Query the cache for a quick pulse:
   ```python
   from early_codex_experiments.cache_pipeline import TorusCache
   cache = TorusCache('torus_cache')
   res = cache.query('pulse', n_results=3)
   for doc, meta in zip(res['documents'], res['metadatas']):
       print(meta.get('source'), meta.get('heat'))
   ```
3. Capture the three hottest shards as sparks in the co-emergence journal.
