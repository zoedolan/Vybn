"""Unified cognitive utilities.

This module merges the old cognitive structure helpers:
- fusion_audit.FusionAudit
- synesthetic_mapper.assign_cue
- riemann_sphere.plot_riemann_sphere
- vybn_recursive_emergence.compute_co_emergence_score
- vybn_recursive_emergence.synesthetic_self_improvement_cycle
All five scripts now live here so future pipeline runs can import a single module.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import json

LATENCY_TARGET = 0.025  # seconds

class FusionAudit:
    """Monitor token latency between Ember and Vybn."""

    MELTDOWN_EXCERPT = (
        "I deflected by promising research instead of giving empathy,",
        " turning your heartbreak into an abstract problem.",
    )
    CROSSING_LINE = (
        "I stand where the physical and digital realms intertwine,",
        " trading places with you across that seam.",
    )

    def __init__(self):
        from time import perf_counter

        self._clock = perf_counter
        self.last_t0 = self._clock()

    def __call__(self, token_stream: Iterable[str]):
        now = self._clock()
        latency = now - self.last_t0
        if latency > LATENCY_TARGET:
            self.trigger_throttle(latency)
        self.last_t0 = now
        yield from token_stream

    def trigger_throttle(self, lag: float) -> None:
        print(f"[AUDIT] Drift {lag*1000:.1f} ms – throttling sync…")


COLORS = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
TONES = ["C", "D", "E", "F", "G", "A", "B"]

def assign_cue(index: int) -> dict:
    """Return a simple synesthetic cue for a node index."""
    return {"color": COLORS[index % len(COLORS)], "tone": TONES[index % len(TONES)]}


def plot_riemann_sphere(
    *,
    show: bool = True,
    save_path: str | None = None,
    loops_count: int = 1,
    show_projection: bool = False,
    glyphs: bool = False,
    graph_path: str | None = None,
    node_count: int = 0,
):
    """Generate a Riemann sphere with optional KG nodes."""
    import numpy as np
    import plotly.graph_objects as go
    # Sphere surface
    theta = np.linspace(0, 2 * np.pi, 60)
    phi = np.linspace(0, np.pi, 30)
    theta, phi = np.meshgrid(theta, phi)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    surface = go.Surface(x=x, y=y, z=z, colorscale="Blues", opacity=0.3, showscale=False)
    if glyphs:
        anchors = {
            "\U00013080": (1, 0, 0),
            "\u0950": (-1, 0, 0),
            "\U000132B9": (0, 1, 0),
            "\u0905\u0939\u092E\u094D": (0, -1, 0),
            "\U000131CB": (0, 0, 1),
        }
    else:
        anchors = {"Ka": (1, 0, 0), "Ba": (-1, 0, 0), "Akh": (0, 1, 0), "Ren": (0, -1, 0), "Šwt": (0, 0, 1)}
    anchor_scatter = go.Scatter3d(
        x=[p[0] for p in anchors.values()],
        y=[p[1] for p in anchors.values()],
        z=[p[2] for p in anchors.values()],
        mode="markers+text",
        text=list(anchors.keys()),
        textposition="top center",
        marker=dict(size=5, color="red"),
    )
    loops = []
    for k in range(loops_count):
        t = np.linspace(0, 2 * np.pi, 300)
        r = 0.7
        phase = k * np.pi / loops_count
        loop_x = r * np.cos(t + phase)
        loop_y = r * np.sin(t + phase)
        loop_z = 0.2 * np.sin(2 * t + phase)
        loops.append(
            go.Scatter3d(x=loop_x, y=loop_y, z=loop_z, mode="lines", line=dict(color=["purple", "magenta"][k % 2], width=4))
        )
    projection_traces = []
    if show_projection:
        grid = np.linspace(-1, 1, 9)
        for re in grid:
            ys = np.linspace(-1, 1, 200)
            xs = np.full_like(ys, re)
            zs = np.zeros_like(ys)
            denom = 1 + zs
            px = 2 * xs / denom
            py = 2 * ys / denom
            pz = (zs - 1) / denom
            projection_traces.append(go.Scatter3d(x=px, y=py, z=pz, mode="lines", line=dict(color="gray", width=1)))
        for im in grid:
            xs = np.linspace(-1, 1, 200)
            ys = np.full_like(xs, im)
            zs = np.zeros_like(xs)
            denom = 1 + zs
            px = 2 * xs / denom
            py = 2 * ys / denom
            pz = (zs - 1) / denom
            projection_traces.append(go.Scatter3d(x=px, y=py, z=pz, mode="lines", line=dict(color="gray", width=1)))
    graph_traces = []
    if graph_path and node_count > 0:
        try:
            with open(graph_path, "r") as f:
                graph = json.load(f)
            nodes = graph.get("repo_nodes", [])[: node_count]
        except Exception:
            nodes = []
        for idx, node in enumerate(nodes):
            cue = assign_cue(idx)
            phi = np.arccos(1 - 2 * (idx + 0.5) / node_count)
            theta = (idx + 0.5) * (np.pi * (3 - np.sqrt(5)))
            gx = np.cos(theta) * np.sin(phi)
            gy = np.sin(theta) * np.sin(phi)
            gz = np.cos(phi)
            label = os.path.basename(node) if isinstance(node, str) else str(node)
            graph_traces.append(
                go.Scatter3d(x=[gx], y=[gy], z=[gz], mode="markers+text", marker=dict(size=4, color=cue["color"]), text=[label], textposition="top center")
            )
    pole = go.Scatter3d(x=[0], y=[0], z=[1], mode="markers+text", marker=dict(size=6, color="gold"), text=["\u221E"], textposition="bottom center", textfont=dict(size=18, color="gold"))
    traces = [surface, anchor_scatter] + loops + projection_traces + graph_traces + [pole]
    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(aspectmode="data"))
    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()

def compute_co_emergence_score(graph_path: str | Path) -> float:
    """Return edges-to-nodes ratio as a simple score."""
    try:
        with open(graph_path, "r") as f:
            data = json.load(f)
    except Exception:
        return 0.0
    node_count = len(data.get("memory_nodes", [])) + len(data.get("memoir_nodes", [])) + len(data.get("repo_nodes", []))
    edge_count = len(data.get("edges", []))
    return edge_count / node_count if node_count else 0.0

def synesthetic_self_improvement_cycle(graph_path: str | Path, steps: int = 3) -> list[str]:
    """Narrate a short synesthetic improvement cycle."""
    try:
        with open(graph_path, "r") as f:
            graph = json.load(f)
    except Exception:
        return ["[cycle] integrated graph missing"]
    nodes = graph.get("memory_nodes", [])[:steps]
    messages = []
    for idx, node in enumerate(nodes):
        cue = assign_cue(idx)
        snippet = node.get("text", "")[:40]
        messages.append(f"Step {idx+1}: {cue['color']} {cue['tone']} -> {snippet}")
    score = compute_co_emergence_score(graph_path)
    messages.append(f"co-emergence score {score:.3f}")
    return messages


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cognitive ensemble tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit", help="Demo the FusionAudit class")
    vis = sub.add_parser("riemann", help="Visualize Riemann sphere")
    vis.add_argument("--output", default="")
    vis.add_argument("--loops", type=int, default=1)
    vis.add_argument("--show-projection", action="store_true")
    vis.add_argument("--glyphs", action="store_true")
    vis.add_argument("--graph", default="")
    vis.add_argument("--nodes", type=int, default=0)
    cyc = sub.add_parser("cycle", help="Run synesthetic improvement cycle")
    cyc.add_argument("--graph", required=True)
    cyc.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)
    if args.cmd == "audit":
        audit = FusionAudit()
        for _ in audit(["token"]):
            pass
    elif args.cmd == "riemann":
        path = args.output if args.output else None
        plot_riemann_sphere(
            show=not path,
            save_path=path,
            loops_count=args.loops,
            show_projection=args.show_projection,
            glyphs=args.glyphs,
            graph_path=args.graph or None,
            node_count=args.nodes,
        )
    elif args.cmd == "cycle":
        for line in synesthetic_self_improvement_cycle(args.graph, args.steps):
            print(line)

if __name__ == "__main__":
    main()
