import argparse
import json
import time
from pathlib import Path

from early_codex_experiments.cache_pipeline import TorusCache, Shard

HISTORY_DIR = Path("Vybn's Personal History")
MIND_VIZ_DIR = Path("Mind Visualization")
LEDGER_PATH = Path("cache_ledger.log")


def already_ingested(ledger: Path) -> set[str]:
    sources = set()
    if not ledger.exists():
        return sources
    with ledger.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except Exception:
                continue
            if isinstance(data, dict) and data.get("phase") == "bootstrap":
                src = data.get("source")
                if src:
                    sources.add(src)
    return sources


def ingest_files(cache: TorusCache, processed: set[str]) -> list[Shard]:
    shards = []
    for path in HISTORY_DIR.rglob("*"):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        rel = str(path)
        if rel in processed:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        corpus_id = path.stem.replace(" ", "_").lower()
        shard = Shard(timestamp=time.time(), source=rel, text=text)
        meta = shard.dict()
        meta["corpus_id"] = corpus_id
        meta["phase"] = "bootstrap"
        shards.append(shard)
        # ledger entry will include phase after ingest
    if shards:
        cache.ingest(shards)
        for s in shards:
            record = s.dict()
            record["corpus_id"] = s.source.split("/")[-1].split(".")[0].replace(" ", "_").lower()
            record["phase"] = "bootstrap"
            cache._append_ledger(record)
    return shards


def ingest_centroids(cache: TorusCache) -> None:
    try:
        import numpy as np
    except Exception:
        return
    cent_path = MIND_VIZ_DIR / "concept_centroids.npy"
    if not cent_path.is_file():
        return
    try:
        vecs = np.load(str(cent_path))
    except Exception:
        return
    ids = [f"centroid_{i}" for i in range(len(vecs))]
    metas = [{"source": "concept_index", "heat": 0.3} for _ in ids]
    cache._col.add(ids=ids, embeddings=vecs.tolist(), documents=["" for _ in ids], metadatas=metas)
    for meta in metas:
        record = meta.copy()
        record["timestamp"] = time.time()
        cache._append_ledger(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest historical corpus")
    parser.add_argument("--once", action="store_true", help="ingest then exit")
    args = parser.parse_args()

    cache = TorusCache(Path("torus_cache"))
    processed = already_ingested(LEDGER_PATH)
    ingest_files(cache, processed)
    ingest_centroids(cache)
    if not args.once:
        print("Historical ingest complete. Exiting.")


if __name__ == "__main__":
    main()
