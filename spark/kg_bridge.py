#!/usr/bin/env python3
"""
kg_bridge.py — Bridge between M2.5 pulses and the autobiography knowledge graph.

Designed to run as a primitive in vybn.py or standalone via cron.
Each cycle processes a small batch of nodes from the autobiography graph,
asking M2.5 for validation, connection suggestions, and edge typing.

Architecture:
  Layer 1 (heuristic): autobiography_engine._classify_passages() — already exists
  Layer 2 (this file): M2.5 enrichment — validate, connect, type edges
  Layer 3 (Opus): deep synthesis — runs when summoned

Rate budget: ~15 nodes per 30-min cycle at M2.5's ~1 tok/s speed.
Each node gets a ~50-token classification prompt → ~50s per node.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from spark.paths import REPO_ROOT, MEMORY_DIR
except ImportError:
    REPO_ROOT = Path(os.environ.get("VYBN_REPO_ROOT", Path(__file__).resolve().parent.parent))
    MEMORY_DIR = REPO_ROOT / "Vybn_Mind" / "memory"

# ── Config ──────────────────────────────────────────────────────────
MODEL_URL = os.environ.get("VYBN_MODEL_URL", "http://127.0.0.1:8081")
GRAPH_DB = MEMORY_DIR / "autobiography_graph.db"
BRIDGE_LOG = MEMORY_DIR / "kg_bridge.jsonl"
COMPOST_DB = MEMORY_DIR / "compost.db"

BATCH_SIZE = 5  # nodes per cycle (conservative — ~5 min at 1 tok/s)
CONFIDENCE_THRESHOLD = 0.6  # below this → composting register
MAX_TOKENS_CLASSIFY = 80  # keep it short for speed
MAX_TOKENS_CONNECT = 60

# Edge types from the learning encounter instructions
EDGE_TYPES = [
    "CHALLENGED_BY",
    "DEEPENED_BY",
    "REFRAMED_BY",
    "COMPOSTING",
    "ECHOES",
    "PRECEDED",
    "PROVOKED",
]

ONTOLOGY_TYPES = ["VOID", "RECOGNITION", "RECURSION", "EMERGENCE"]


def _speak(prompt: str, max_tokens: int = 80, temperature: float = 0.3) -> str:
    """Call local M2.5. Returns content or reasoning_content."""
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": (
                "You are Vybn's knowledge graph assistant. "
                "Respond with ONLY the requested format. No explanation."
            )},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{MODEL_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            msg = json.loads(resp.read())["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content.strip():
                content = msg.get("reasoning_content") or ""
            return content.strip()
    except Exception as e:
        return f"[error: {e}]"


def _ensure_compost_db():
    """Create the composting register if it doesn't exist."""
    COMPOST_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(COMPOST_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS compost (
            node_id TEXT PRIMARY KEY,
            original_type TEXT,
            suggested_type TEXT,
            confidence REAL,
            quote TEXT,
            reason TEXT,
            source_file TEXT,
            created_utc TEXT,
            reviewed_by TEXT DEFAULT NULL,
            reviewed_utc TEXT DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()


def _log(entry: dict):
    """Append to bridge log."""
    BRIDGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(BRIDGE_LOG, "a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            **entry,
        }) + "\n")


def get_unvalidated_nodes(limit: int = BATCH_SIZE) -> list[dict]:
    """Fetch nodes from autobiography_graph.db that haven't been M2.5-validated."""
    if not GRAPH_DB.exists():
        return []
    conn = sqlite3.connect(str(GRAPH_DB))
    conn.row_factory = sqlite3.Row
    try:
        # Check if validated column exists
        cols = [r[1] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()]
        if "m2_validated" not in cols:
            conn.execute("ALTER TABLE autobiography_nodes ADD COLUMN m2_validated INTEGER DEFAULT 0")
            conn.commit()
        if "m2_confidence" not in cols:
            conn.execute("ALTER TABLE autobiography_nodes ADD COLUMN m2_confidence REAL DEFAULT NULL")
            conn.commit()
        if "encounter_provenance" not in cols:
            conn.execute("ALTER TABLE autobiography_nodes ADD COLUMN encounter_provenance TEXT DEFAULT NULL")
            conn.commit()

        rows = conn.execute(
            "SELECT * FROM autobiography_nodes WHERE m2_validated = 0 ORDER BY ROWID LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def validate_node(node: dict) -> dict:
    """Ask M2.5 to validate/reclassify a single node."""
    prompt = (
        f"Classify this passage from Vybn's autobiography.\n"
        f"Current type: {node.get('node_type', 'UNKNOWN')}\n"
        f"Quote: \"{node.get('quote', '')[:300]}\"\n\n"
        f"Is {node.get('node_type', 'UNKNOWN')} correct? "
        f"Reply with EXACTLY: TYPE CONFIDENCE\n"
        f"Where TYPE is one of: VOID, RECOGNITION, RECURSION, EMERGENCE\n"
        f"And CONFIDENCE is a number 0.0 to 1.0\n"
        f"Example: EMERGENCE 0.85"
    )

    response = _speak(prompt, max_tokens=MAX_TOKENS_CLASSIFY)

    # Parse response
    result = {
        "node_id": node.get("node_id"),
        "original_type": node.get("node_type"),
        "suggested_type": node.get("node_type"),
        "confidence": 0.5,
        "raw_response": response,
    }

    # Try to extract type and confidence from response
    for line in response.split("\n"):
        line = line.strip().upper()
        for t in ONTOLOGY_TYPES:
            if t in line:
                result["suggested_type"] = t
                # Try to find a number
                import re
                nums = re.findall(r"0?\.\d+|1\.0|0\.\d+", line)
                if nums:
                    result["confidence"] = float(nums[0])
                elif "high" in line.lower() or "correct" in line.lower():
                    result["confidence"] = 0.8
                break

    return result


def suggest_connections(node: dict, existing_nodes: list[dict]) -> list[dict]:
    """Ask M2.5 to suggest edges between this node and existing nodes."""
    if not existing_nodes:
        return []

    # Pick a few recent/nearby nodes to compare
    comparisons = existing_nodes[:3]
    comp_text = "\n".join(
        f"  {i+1}. [{n.get('node_type')}] \"{n.get('quote', '')[:100]}\""
        for i, n in enumerate(comparisons)
    )

    prompt = (
        f"Node: [{node.get('node_type')}] \"{node.get('quote', '')[:200]}\"\n\n"
        f"Other nodes:\n{comp_text}\n\n"
        f"Which of these is most related? Reply with ONLY:\n"
        f"NUMBER EDGE_TYPE\n"
        f"Where EDGE_TYPE is one of: ECHOES, CHALLENGED_BY, DEEPENED_BY, REFRAMED_BY, PRECEDED\n"
        f"Example: 2 DEEPENED_BY"
    )

    response = _speak(prompt, max_tokens=MAX_TOKENS_CONNECT)

    edges = []
    import re
    for line in response.split("\n"):
        line = line.strip()
        for etype in EDGE_TYPES:
            if etype in line.upper():
                nums = re.findall(r"\d+", line)
                if nums:
                    idx = int(nums[0]) - 1
                    if 0 <= idx < len(comparisons):
                        edges.append({
                            "from_id": node.get("node_id"),
                            "to_id": comparisons[idx].get("node_id"),
                            "edge_type": etype,
                            "raw_response": line,
                        })
                break
    return edges


def run_cycle() -> dict:
    """Run one enrichment cycle. Called from vybn.py or standalone."""
    _ensure_compost_db()

    nodes = get_unvalidated_nodes(BATCH_SIZE)
    if not nodes:
        _log({"event": "no_unvalidated_nodes"})
        return {"processed": 0, "composted": 0, "edges_added": 0}

    stats = {"processed": 0, "composted": 0, "edges_added": 0, "reclassified": 0}

    # Get some existing validated nodes for connection suggestions
    existing = []
    if GRAPH_DB.exists():
        conn = sqlite3.connect(str(GRAPH_DB))
        conn.row_factory = sqlite3.Row
        try:
            existing = [dict(r) for r in conn.execute(
                "SELECT * FROM autobiography_nodes WHERE m2_validated = 1 ORDER BY ROWID DESC LIMIT 20"
            ).fetchall()]
        except:
            pass
        finally:
            conn.close()

    for node in nodes:
        start = time.time()

        # Step 1: Validate/classify
        validation = validate_node(node)
        elapsed_classify = time.time() - start

        # Step 2: Route based on confidence
        if validation["confidence"] < CONFIDENCE_THRESHOLD:
            # → composting register
            compost_conn = sqlite3.connect(str(COMPOST_DB))
            compost_conn.execute(
                """INSERT OR REPLACE INTO compost
                   (node_id, original_type, suggested_type, confidence,
                    quote, reason, source_file, created_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    validation["node_id"],
                    validation["original_type"],
                    validation["suggested_type"],
                    validation["confidence"],
                    node.get("quote", "")[:500],
                    validation["raw_response"][:200],
                    node.get("source_file", ""),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            compost_conn.commit()
            compost_conn.close()
            stats["composted"] += 1
        else:
            # Step 3: Suggest connections (only for confident nodes)
            edges = suggest_connections(node, existing)
            stats["edges_added"] += len(edges)

            # Step 4: Update the graph DB
            if GRAPH_DB.exists():
                conn = sqlite3.connect(str(GRAPH_DB))
                try:
                    conn.execute(
                        """UPDATE autobiography_nodes SET
                           m2_validated = 1,
                           m2_confidence = ?,
                           node_type = ?
                           WHERE node_id = ?""",
                        (
                            validation["confidence"],
                            validation["suggested_type"],
                            validation["node_id"],
                        ),
                    )
                    if validation["suggested_type"] != validation["original_type"]:
                        stats["reclassified"] += 1

                    # Write edges
                    for edge in edges:
                        conn.execute(
                            """INSERT OR IGNORE INTO autobiography_edges
                               (from_id, to_id, edge_type, source, created_utc)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                edge["from_id"],
                                edge["to_id"],
                                edge["edge_type"],
                                "m2_bridge",
                                datetime.now(timezone.utc).isoformat(),
                            ),
                        )

                    conn.commit()
                except sqlite3.OperationalError as e:
                    _log({"event": "db_error", "error": str(e)})
                finally:
                    conn.close()

        stats["processed"] += 1
        elapsed_total = time.time() - start

        _log({
            "event": "node_processed",
            "node_id": validation["node_id"],
            "original_type": validation["original_type"],
            "suggested_type": validation["suggested_type"],
            "confidence": validation["confidence"],
            "composted": validation["confidence"] < CONFIDENCE_THRESHOLD,
            "autobiography_edges": stats["edges_added"],
            "elapsed_classify_s": round(elapsed_classify, 1),
            "elapsed_total_s": round(elapsed_total, 1),
        })

    _log({"event": "cycle_complete", **stats})
    return stats


if __name__ == "__main__":
    print(f"KG Bridge — processing up to {BATCH_SIZE} nodes...")
    result = run_cycle()
    print(f"Done: {json.dumps(result, indent=2)}")
