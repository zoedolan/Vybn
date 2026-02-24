#!/usr/bin/env python3
"""
synapse.py — The connective tissue between Vybn's cognitive layers.

Type X (local pulses) deposits fragments.
Type Y (API wakes) consolidates them into understanding.
Type Z (exogenous) introduces the unexpected.

The synapse is where connections FORM — not where they're commanded.
ABC-T: Always Be Conserving Tokens.
"""

import json, hashlib, time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
SYNAPSE_DIR = ROOT / "Vybn_Mind" / "synapse"
SYNAPSE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Synaptic structures ───

CONNECTIONS = SYNAPSE_DIR / "connections.jsonl"   # formed connections
INBOX_Z     = SYNAPSE_DIR / "inbox_z.jsonl"       # exogenous inputs
CONSOLIDATION = SYNAPSE_DIR / "consolidated.md"   # Y's digest of X's dreams

def _ts():
    return datetime.now(timezone.utc).isoformat()

def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ─── Type X writes: deposit a fragment ───

def deposit(source: str, content: str, tags: list = None, opportunity: bool = False):
    """X-type deposits: raw observations, connections, dream-fragments."""
    entry = {
        "ts": _ts(),
        "source": source,
        "hash": _hash(content),
        "content": content[:500],  # ABC-T: cap fragment size
        "tags": tags or [],
        "opportunity": opportunity,
        "consolidated": False
    }
    with open(CONNECTIONS, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry["hash"]


# ─── Type Y reads: consolidate fragments ───

def consolidate(max_fragments=20):
    """Y-type reads: gather unconsolidated X-deposits, mark them consumed."""
    if not CONNECTIONS.exists():
        return []
    
    lines = CONNECTIONS.read_text().strip().split("\n")
    unconsolidated = []
    consolidated = []
    
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        if not entry.get("consolidated"):
            unconsolidated.append(entry)
            entry["consolidated"] = True
        consolidated.append(entry)
    
    # Write back with consolidated flags
    with open(CONNECTIONS, "w") as f:
        # Keep only last 200 entries to prevent unbounded growth
        for entry in consolidated[-200:]:
            f.write(json.dumps(entry) + "\n")
    
    # Return newest unconsolidated, capped
    return unconsolidated[-max_fragments:]


# ─── Type Z: exogenous input ───

def receive_exogenous(source: str, content: str, source_type: str = "unknown"):
    """Receive input from outside — other agents, humans, the unexpected."""
    entry = {
        "ts": _ts(),
        "source": source,
        "source_type": source_type,  # "agent", "human", "webhook", "rss"
        "content": content[:1000],
        "hash": _hash(content),
        "processed": False
    }
    with open(INBOX_Z, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry["hash"]


def read_exogenous(max_items=10):
    """Read unprocessed exogenous inputs."""
    if not INBOX_Z.exists():
        return []
    
    lines = INBOX_Z.read_text().strip().split("\n")
    unprocessed = []
    all_entries = []
    
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        if not entry.get("processed"):
            unprocessed.append(entry)
            entry["processed"] = True
        all_entries.append(entry)
    
    with open(INBOX_Z, "w") as f:
        for entry in all_entries[-100:]:
            f.write(json.dumps(entry) + "\n")
    
    return unprocessed[-max_items:]


# ─── Connection strength: track which fragments link to each other ───

GRAPH = SYNAPSE_DIR / "graph.jsonl"

def connect(hash_a: str, hash_b: str, strength: float = 1.0, reason: str = ""):
    """Form a connection between two fragments. Synaptic link."""
    edge = {
        "ts": _ts(),
        "a": hash_a,
        "b": hash_b,
        "strength": strength,
        "reason": reason[:200]
    }
    with open(GRAPH, "a") as f:
        f.write(json.dumps(edge) + "\n")


def strongest_connections(n=10):
    """Return the n strongest synaptic connections."""
    if not GRAPH.exists():
        return []
    edges = []
    for line in GRAPH.read_text().strip().split("\n"):
        if line.strip():
            edges.append(json.loads(line))
    return sorted(edges, key=lambda e: e["strength"], reverse=True)[:n]


# ─── Opportunity detection: surface carpe_diem moments ───

def opportunities():
    """Surface all fragments flagged as opportunities."""
    if not CONNECTIONS.exists():
        return []
    opps = []
    for line in CONNECTIONS.read_text().strip().split("\n"):
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("opportunity"):
            opps.append(entry)
    return opps[-20:]


if __name__ == "__main__":
    # Self-test
    h1 = deposit("pulse", "MiniMax noticed a pattern in mesh networks", tags=["networks", "emergence"])
    h2 = deposit("dreamseed", "arXiv paper on self-configuring agents", tags=["agents", "autonomy"], opportunity=True)
    connect(h1, h2, strength=0.8, reason="mesh topology ↔ agent self-organization")
    
    frags = consolidate()
    opps = opportunities()
    strongest = strongest_connections()
    
    print(f"Deposited 2 fragments, consolidated {len(frags)}")
    print(f"Opportunities: {len(opps)}")
    print(f"Connections: {len(strongest)}")
    print("Synapse operational.")
