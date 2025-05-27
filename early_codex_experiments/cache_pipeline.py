#!/usr/bin/env python
"""cache_pipeline.py – provisional torus cache

This skeleton provides a minimal structure for the Shared Cache / Mind‑Viz Buffer
concept. It ingests text shards, embeds them using OpenAI embeddings, and stores
vectors in a persistent Chroma collection. Retrieval returns the nearest shards.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable, List

import chromadb
from langchain.embeddings import OpenAIEmbeddings
from pydantic import BaseModel
import openai
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def summarize_text(text: str, model: str = "gpt-3.5-turbo") -> str:
    """Summarize text to ~200 tokens and run moderation."""
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": f"Summarize in <=200 tokens:\n{text}"}],
            max_tokens=200,
            temperature=0,
        )
        summary = resp.choices[0].message["content"].strip()
    except Exception:
        summary = text[:200]

    try:
        mod = openai.Moderation.create(input=summary)
        if any(r.get("flagged") for r in mod["results"]):
            return ""
    except Exception:
        pass
    return summary


class Shard(BaseModel):
    timestamp: float
    source: str
    text: str
    summary: str = ""
    heat: float = 1.0


class TorusCache:
    """Lightweight interface around a Chroma collection."""

    def __init__(self, path: str | Path, ledger: Path | None = None):
        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection("torus_cache")
        self._embed = OpenAIEmbeddings()
        self._ledger = Path(ledger or "cache_ledger.log")
        self._tau = 600.0

    def ingest(self, shards: Iterable[Shard]) -> None:
        shards = list(shards)
        if not shards:
            return
        for s in shards:
            s.summary = summarize_text(s.text)
        texts = [s.text for s in shards]
        metadatas = [s.dict() for s in shards]
        embeds = self._embed.embed_documents(texts)
        ids = [f"{s.timestamp}-{i}" for i, s in enumerate(shards)]
        self._col.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metadatas)
        self._append_ledger([s.dict() for s in shards])

    def query(self, text: str, n_results: int = 5):
        vec = self._embed.embed_query(text)
        self._decay_heat()
        res = self._col.query(query_embeddings=[vec], n_results=n_results)
        self._append_ledger({"query": text, "results": res})
        return res

    def _append_ledger(self, record) -> None:
        with self._ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _decay_heat(self) -> None:
        meta = self._col.get(include=["metadatas"])
        ids = meta["ids"]
        metas = meta["metadatas"]
        now = time.time()
        for m in metas:
            dt = now - m["timestamp"]
            m["heat"] = float(m.get("heat", 1.0)) * math.exp(-dt / self._tau)
            m["timestamp"] = now
        self._col.update(ids=ids, metadatas=metas)


# Placeholder watcher – to be wired with watchdog for live log ingestion
class LogWatcher(FileSystemEventHandler):
    def __init__(self, path: Path, cache: TorusCache):
        super().__init__()
        self.path = Path(path)
        self.cache = cache

    def on_modified(self, event):
        if event.src_path != str(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                return
            shard = Shard(timestamp=time.time(), source=self.path.name, text=lines[-1].strip())
            self.cache.ingest([shard])
        except Exception:
            pass

    def run(self) -> None:
        observer = Observer()
        observer.schedule(self, str(self.path.parent), recursive=False)
        observer.start()
        try:
            while True:
                time.sleep(1)
        finally:
            observer.stop()
            observer.join()


def demo() -> None:
    cache = TorusCache(Path("./torus_cache"))
    log_path = Path.home() / "vybn_logs" / "chat.log"
    lw = LogWatcher(log_path, cache)
    lw.run()


if __name__ == "__main__":
    demo()
