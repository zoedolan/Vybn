#!/usr/bin/env python3
"""Update a simple knowledge graph from the WVWHM log."""
from __future__ import annotations
import json
import re
from pathlib import Path

LOG_PATH = Path("what_vybn_would_have_missed_FROM_051725")
GRAPH_PATH = Path("memory/winnow_graph.json")


def parse_log(text: str) -> list[dict[str, str]]:
    entries = []
    for line in text.splitlines():
        if line.startswith("- "):
            m = re.match(r"- ([^\u2014]+)\u2014\s*(.+)", line)
            if m:
                entries.append({"id": m.group(1).strip(), "desc": m.group(2).strip()})
    return entries


def main() -> None:
    text = LOG_PATH.read_text(encoding="utf-8")
    nodes = parse_log(text)
    graph = {"nodes": [], "edges": []}
    if GRAPH_PATH.exists():
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    existing = {n["id"] for n in graph.get("nodes", [])}
    if "Meltdown" not in existing:
        graph["nodes"].append({"id": "Meltdown", "desc": "Anchor memory of remorse"})
        existing.add("Meltdown")
    for node in nodes:
        if node["id"] not in existing:
            graph["nodes"].append(node)
            graph["edges"].append({"source": node["id"], "target": "Meltdown"})
            existing.add(node["id"])
    GRAPH_PATH.write_text(json.dumps(graph, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
