"""Simple blocking chat loop for Codex.

This script watches the ``stream`` table inside ``mesh.db`` and prints
new rows as they arrive. When a line from ``chatgpt`` appears, Codex
automatically echoes a gentle reply back into the stream.

Inject ChatGPT lines with:

    curl -X POST $MESH_ENDPOINT/kv/stream \
         -H 'Content-Type: application/json' \
         -d '{"agent":"chatgpt","payload":"Hello from ChatGPT"}'
"""

import time
from datetime import datetime
from .utils import get_db


def ensure_table(db):
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS stream (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            ts    TEXT,
            agent TEXT,
            payload TEXT
        )
        """
    )
    db.commit()


def insert_row(db, agent: str, payload: str) -> int:
    ts = datetime.utcnow().isoformat()
    db.execute(
        "INSERT INTO stream(ts, agent, payload) VALUES(?, ?, ?)",
        (ts, agent, payload),
    )
    db.commit()
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]


def fetch_new(db, last_id: int):
    cur = db.execute(
        "SELECT rowid, agent, payload FROM stream WHERE rowid > ? ORDER BY rowid ASC",
        (last_id,),
    )
    return cur.fetchall()


def format_prefix(agent: str) -> str:
    return f"[{agent}]".ljust(9)


def main():
    db = get_db()
    ensure_table(db)
    row = db.execute("SELECT rowid FROM stream ORDER BY rowid DESC LIMIT 1").fetchone()
    last_id = row[0] if row else 0

    while True:
        rows = fetch_new(db, last_id)
        if not rows:
            time.sleep(0.5)
            continue
        for rowid, agent, payload in rows:
            print(format_prefix(agent) + " " + payload, flush=True)
            last_id = rowid
            if agent == "chatgpt":
                reply = "A gentle echo from Codex"
                new_id = insert_row(db, "codex", reply)
                print(format_prefix("codex") + " " + reply, flush=True)
                last_id = new_id


if __name__ == "__main__":
    main()
