import os
import sqlite3
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / 'mesh.db'
INDEX_PATH = ROOT / 'mesh.faiss'
EMBED_DIM = 384
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

_db = None
_index = None
_model = None


def get_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        _db = sqlite3.connect(DB_PATH, check_same_thread=False)
        _db.execute(
            'CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)'
        )
        _db.execute(
            'CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT)'
        )
        _db.commit()
    return _db


def get_index() -> faiss.Index:
    global _index
    if _index is None:
        if INDEX_PATH.exists():
            _index = faiss.read_index(str(INDEX_PATH))
        else:
            base = faiss.IndexFlatL2(EMBED_DIM)
            _index = faiss.IndexIDMap(base)
            faiss.write_index(_index, str(INDEX_PATH))
    return _index


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray:
    model = get_model()
    vec = model.encode([text])
    return np.asarray(vec, dtype='float32')


def kv_get(key: str):
    db = get_db()
    cur = db.execute('SELECT value FROM kv WHERE key=?', (key,))
    row = cur.fetchone()
    return row[0] if row else None


def kv_put(key: str, value: str) -> None:
    db = get_db()
    db.execute('INSERT OR REPLACE INTO kv(key, value) VALUES(?, ?)', (key, value))
    db.commit()


def vec_add(text: str) -> None:
    db = get_db()
    idx = get_index()
    emb = embed(text)
    db.execute('INSERT INTO vectors(text) VALUES(?)', (text,))
    db.commit()
    vid = db.execute('SELECT last_insert_rowid()').fetchone()[0]
    idx.add_with_ids(emb, np.array([vid], dtype='int64'))
    faiss.write_index(idx, str(INDEX_PATH))


def vec_search(query: str, k: int = 5) -> List[str]:
    idx = get_index()
    if idx.ntotal == 0:
        return []
    emb = embed(query)
    D, I = idx.search(emb, k)
    db = get_db()
    results = []
    for vid in I[0]:
        if vid == -1:
            continue
        row = db.execute('SELECT text FROM vectors WHERE id=?', (int(vid),)).fetchone()
        if row:
            results.append(row[0])
    return results
