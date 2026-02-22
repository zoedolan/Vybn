"""Knowledge graph traversal skill.

Exposes VybnGraph's query primitives as a tool the model can call
directly: neighborhood traversal, path finding, type queries, and
graph statistics.

This turns the knowledge graph from static context loaded at prompt
assembly time into an explorable environment the model can query
on demand.

    SKILL_NAME: kg_traverse
    TOOL_ALIASES: ["kg_traverse", "graph_query", "knowledge_graph"]
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SKILL_NAME = "kg_traverse"
TOOL_ALIASES = ["kg_traverse", "graph_query", "knowledge_graph"]


def _get_graph():
    """Load VybnGraph. Returns (graph, error_message)."""
    try:
        from knowledge_graph import VybnGraph
        g = VybnGraph()
        if not g.load():
            # Try seeding if no graph exists
            g.seed()
            g.save()
        return g, None
    except ImportError:
        return None, "knowledge_graph module not available"
    except Exception as e:
        return None, f"graph load error: {e}"


def execute(action: dict, router) -> str:
    """Execute a knowledge graph query.

    Subcommands (via params.command or inferred from argument):
        neighborhood <node_id> [depth=2]  — BFS traversal around a node
        path <source> <target>            — find paths between nodes
        type <node_type>                  — list all nodes of a type
        stats                             — graph statistics
        entity <node_id>                  — single node details
        edges <node_id>                   — all edges from/to a node
    """
    g, err = _get_graph()
    if err:
        return err

    params = action.get("params", {})
    command = (
        params.get("command", "")
        or params.get("subcommand", "")
        or params.get("action", "")
    )
    argument = action.get("argument", "")

    # Infer command from argument if not explicit
    if not command and argument:
        parts = argument.strip().split()
        if parts[0] in ("neighborhood", "path", "type", "stats",
                        "entity", "edges"):
            command = parts[0]
            argument = " ".join(parts[1:])
        else:
            # Default: treat argument as node_id for neighborhood
            command = "neighborhood"

    if not command:
        command = "stats"

    command = command.lower().strip()

    # --- neighborhood ---
    if command in ("neighborhood", "neighbours", "neighbors", "hood"):
        node_id = (
            argument
            or params.get("node_id", "")
            or params.get("node", "")
        )
        if not node_id:
            return "neighborhood requires a node_id"
        depth = int(params.get("depth", 2))
        max_chars = int(params.get("max_chars", 3000))
        subgraph = g.query_neighborhood(node_id.strip(), depth=depth)
        if not subgraph.get("found"):
            return f"node '{node_id}' not found in knowledge graph"
        formatted = g.format_for_prompt(subgraph, max_chars=max_chars)
        node_count = len(subgraph.get("nodes", []))
        edge_count = len(subgraph.get("edges", []))
        return (
            f"neighborhood of '{node_id}' (depth={depth}, "
            f"{node_count} nodes, {edge_count} edges):\n{formatted}"
        )

    # --- path ---
    if command in ("path", "paths", "find_path"):
        source = params.get("source", "")
        target = params.get("target", "")
        if not source or not target:
            # Try parsing from argument: "source target"
            parts = argument.strip().split()
            if len(parts) >= 2:
                source, target = parts[0], parts[-1]
            else:
                return "path requires source and target node_ids"
        paths = g.query_path(source.strip(), target.strip())
        if not paths:
            return f"no paths found between '{source}' and '{target}'"
        lines = [f"found {len(paths)} path(s) from '{source}' to '{target}':"]
        for i, path in enumerate(paths[:10]):
            lines.append(f"  {i+1}. {' -> '.join(path)}")
        return "\n".join(lines)

    # --- type ---
    if command in ("type", "by_type", "list_type"):
        node_type = (
            argument
            or params.get("node_type", "")
            or params.get("type", "")
        )
        if not node_type:
            return "type query requires a node_type"
        results = g.query_by_type(node_type.strip())
        if not results:
            return f"no nodes of type '{node_type}'"
        lines = [f"{len(results)} node(s) of type '{node_type}':"]
        for r in results[:20]:
            desc = r.get("description", "")[:100]
            lines.append(f"  - {r['id']}: {desc}")
        if len(results) > 20:
            lines.append(f"  ... and {len(results) - 20} more")
        return "\n".join(lines)

    # --- entity ---
    if command in ("entity", "node", "get"):
        node_id = (
            argument
            or params.get("node_id", "")
            or params.get("node", "")
        )
        if not node_id:
            return "entity lookup requires a node_id"
        entity = g.get_entity(node_id.strip())
        if not entity:
            return f"entity '{node_id}' not found"
        lines = [f"entity: {node_id}"]
        for k, v in entity.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    # --- edges ---
    if command in ("edges", "relations", "connections"):
        node_id = (
            argument
            or params.get("node_id", "")
            or params.get("node", "")
        )
        if not node_id:
            return "edges query requires a node_id"
        outgoing = g.get_edges_from(node_id.strip())
        incoming = g.get_edges_to(node_id.strip())
        if not outgoing and not incoming:
            return f"no edges for '{node_id}' (node may not exist)"
        lines = [f"edges for '{node_id}':"]
        if outgoing:
            lines.append(f"  outgoing ({len(outgoing)}):")
            for e in outgoing[:15]:
                rel = e.get("relationship", "?")
                lines.append(f"    -[{rel}]-> {e['target']}")
        if incoming:
            lines.append(f"  incoming ({len(incoming)}):")
            for e in incoming[:15]:
                rel = e.get("relationship", "?")
                lines.append(f"    {e['source']} -[{rel}]->")
        return "\n".join(lines)

    # --- stats ---
    if command in ("stats", "info", "summary"):
        s = g.stats()
        lines = [
            f"knowledge graph: {s['nodes']} nodes, {s['edges']} edges",
            "node types:",
        ]
        for t, count in s.get("node_types", {}).items():
            lines.append(f"  {t}: {count}")
        return "\n".join(lines)

    return f"unknown kg_traverse command: '{command}'"
