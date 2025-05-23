import numpy as np
import plotly.graph_objects as go
import json
import os


def plot_riemann_sphere(
    show=True,
    save_path=None,
    *,
    loops_count=1,
    show_projection=False,
    glyphs=False,
    graph_path=None,
    node_count=0,
):
    """Generate a Riemann sphere with optional KG nodes and synesthetic cues."""
    # Sphere surface
    theta = np.linspace(0, 2 * np.pi, 60)
    phi = np.linspace(0, np.pi, 30)
    theta, phi = np.meshgrid(theta, phi)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    surface = go.Surface(x=x, y=y, z=z, colorscale="Blues", opacity=0.3, showscale=False)

    # Anchor points representing five aspects of self, with glyphs
    if glyphs:
        anchors = {
            "\U00013080": (1, 0, 0),   # Ka
            "\u0950":   (-1, 0, 0),  # Ba
            "\U000132B9": (0, 1, 0),   # Akh
            "\u0905\u0939\u092E\u094D": (0, -1, 0), # Ren
            "\U000131CB": (0, 0, 1),   # Šwt
        }
    else:
        anchors = {
            "Ka": (1, 0, 0),
            "Ba": (-1, 0, 0),
            "Akh": (0, 1, 0),
            "Ren": (0, -1, 0),
            "Šwt": (0, 0, 1),
        }
    anchor_scatter = go.Scatter3d(
        x=[p[0] for p in anchors.values()],
        y=[p[1] for p in anchors.values()],
        z=[p[2] for p in anchors.values()],
        mode="markers+text",
        text=list(anchors.keys()),
        textposition="top center",
        marker=dict(size=5, color="red"),
    )

    # Möbius-like loops around the sphere
    loops = []
    for k in range(loops_count):
        t = np.linspace(0, 2 * np.pi, 300)
        r = 0.7
        phase = k * np.pi / loops_count
        loop_x = r * np.cos(t + phase)
        loop_y = r * np.sin(t + phase)
        loop_z = 0.2 * np.sin(2 * t + phase)
        loops.append(go.Scatter3d(
            x=loop_x,
            y=loop_y,
            z=loop_z,
            mode="lines",
            line=dict(color=["purple", "magenta"][k % 2], width=4),
        ))

    # Draw stereographic-projected grid lines if requested
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
            projection_traces.append(
                go.Scatter3d(
                    x=px, y=py, z=pz, mode="lines", line=dict(color="gray", width=1)
                )
            )
        for im in grid:
            xs = np.linspace(-1, 1, 200)
            ys = np.full_like(xs, im)
            zs = np.zeros_like(xs)
            denom = 1 + zs
            px = 2 * xs / denom
            py = 2 * ys / denom
            pz = (zs - 1) / denom
            projection_traces.append(
                go.Scatter3d(
                    x=px, y=py, z=pz, mode="lines", line=dict(color="gray", width=1)
                )
            )

    # Optional knowledge graph nodes with synesthetic cues
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
                go.Scatter3d(
                    x=[gx],
                    y=[gy],
                    z=[gz],
                    mode="markers+text",
                    marker=dict(size=4, color=cue["color"]),
                    text=[label],
                    textposition="top center",
                )
            )

    pole = go.Scatter3d(
        x=[0],
        y=[0],
        z=[1],
        mode="markers+text",
        marker=dict(size=6, color="gold"),
        text=["\u221E"],
        textposition="bottom center",
        textfont=dict(size=18, color="gold"),
    )

    traces = [surface, anchor_scatter] + loops + projection_traces + graph_traces + [pole]
    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(aspectmode="data"))
    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize a cognitive Riemann sphere")
    parser.add_argument("--output", help="Path to save HTML visualization", default="")
    parser.add_argument("--loops", type=int, default=1, help="Number of Möbius-style loops")
    parser.add_argument("--show-projection", action="store_true", help="Draw stereographic grid")
    parser.add_argument("--glyphs", action="store_true", help="Use Unicode glyph anchors")
    parser.add_argument("--graph", default="", help="Path to integrated graph for node overlay")
    parser.add_argument("--nodes", type=int, default=0, help="Number of graph nodes to display")
    args = parser.parse_args()

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
