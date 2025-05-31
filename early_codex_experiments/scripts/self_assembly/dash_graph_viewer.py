import os
import json
import base64
import io
import math
import wave
import random

import numpy as np
import networkx as nx
import dash
from dash import html, dcc
import dash_cytoscape as cyto
import plotly.graph_objects as go

from ..quantum_rng import seed_random

seed_random()

# Path to the integrated graph JSON shipped by the self-assembly pipeline
GRAPH_PATH = os.path.join(os.path.dirname(__file__), "integrated_graph.json")

# Load graph data
with open(GRAPH_PATH, "r") as f:
    graph_data = json.load(f)

# Node lists from the graph
memory_nodes = graph_data.get("memory_nodes", [])
memoir_nodes = graph_data.get("memoir_nodes", [])
repo_nodes = graph_data.get("repo_nodes", [])

# Normalise repo_nodes as a list of dicts with id/label
repo_node_list = []
for node in repo_nodes:
    if isinstance(node, str):
        node_id = node
        label = os.path.basename(node)
        repo_node_list.append({"id": node_id, "label": label, "type": "concept"})
    elif isinstance(node, dict):
        node.setdefault("label", str(node.get("id", "")))
        node["type"] = "concept"
        repo_node_list.append(node)

# Process memory and memoir nodes with type "self"
for node in memory_nodes:
    node["type"] = "self"
    node["label"] = node.get("date", node.get("id", ""))
for node in memoir_nodes:
    node["type"] = "self"
    node["label"] = node.get("title", node.get("id", ""))

# Combine all nodes
all_nodes = []
for node in memory_nodes + memoir_nodes + repo_node_list:
    if "id" in node:
        all_nodes.append({"data": {"id": node["id"], "label": node.get("label", ""), "type": node.get("type", "")}})

# Build NetworkX graph to detect cycles
edges = graph_data.get("edges", [])
G = nx.Graph()
for edge in edges:
    src = edge.get("source"); tgt = edge.get("target")
    if src is None or tgt is None:
        continue
    G.add_edge(src, tgt)

# Ensure isolated nodes are present
for node_elem in all_nodes:
    nid = node_elem["data"]["id"]
    if nid not in G:
        G.add_node(nid)

cycle_edges = set()
for cycle in nx.cycle_basis(G):
    for i in range(len(cycle)):
        u = cycle[i]; v = cycle[(i + 1) % len(cycle)]
        cycle_edges.add(tuple(sorted((u, v))))

# Cytoscape edge data with cycle highlight
cy_edges = []
for edge in edges:
    src = edge.get("source"); tgt = edge.get("target")
    if src is None or tgt is None:
        continue
    data = {"source": src, "target": tgt}
    if tuple(sorted((src, tgt))) in cycle_edges:
        data["cycleEdge"] = 1
    cy_edges.append({"data": data})

# Dash app
app = dash.Dash(__name__)
app.title = "Vybn Knowledge Graph Viewer"

cytoscape_graph = cyto.Cytoscape(
    id="cytoscape-graph",
    layout={"name": "cose", "padding": 20},
    style={"width": "100%", "height": "600px"},
    elements=all_nodes + cy_edges,
    stylesheet=[
        {"selector": 'node[type = "self"]', "style": {"background-color": "red", "label": "data(label)", "color": "#fff", "text-valign": "center", "text-halign": "center"}},
        {"selector": 'node[type = "concept"]', "style": {"background-color": "blue", "label": "data(label)", "color": "#fff", "text-valign": "center", "text-halign": "center"}},
        {"selector": "edge", "style": {"line-color": "#cccccc", "curve-style": "bezier"}},
        {"selector": "edge[cycleEdge = 1]", "style": {"line-color": "gold", "width": 3}},
    ],
)


def create_sphere_figure(use_glyphs=True, loops_count=1):
    traces = []
    offset = random.random() * 2 * np.pi
    theta = np.linspace(0, 2 * np.pi, 60) + offset
    phi = np.linspace(0, np.pi, 30)
    theta, phi = np.meshgrid(theta, phi)
    x_s = np.cos(theta) * np.sin(phi)
    y_s = np.sin(theta) * np.sin(phi)
    z_s = np.cos(phi)
    surface = go.Surface(x=x_s, y=y_s, z=z_s, colorscale="Blues", opacity=0.3, showscale=False)
    traces.append(surface)

    if use_glyphs:
        anchors = {
            "\U00013080": (1, 0, 0),
            "\u0950": (-1, 0, 0),
            "\U000132B9": (0, 1, 0),
            "\u0905\u0939\u092E\u094D": (0, -1, 0),
            "\U000131CB": (0, 0, 1),
        }
    else:
        anchors = {"Ka": (1, 0, 0), "Ba": (-1, 0, 0), "Akh": (0, 1, 0), "Ren": (0, -1, 0), "Šwt": (0, 0, 1)}

    anchor_coords = list(anchors.values())
    anchor_labels = list(anchors.keys())
    anchor_trace = go.Scatter3d(
        x=[p[0] for p in anchor_coords],
        y=[p[1] for p in anchor_coords],
        z=[p[2] for p in anchor_coords],
        mode="markers+text",
        text=anchor_labels,
        textposition="top center",
        marker=dict(size=5, color="red"),
    )
    traces.append(anchor_trace)

    for k in range(loops_count):
        t = np.linspace(0, 2 * np.pi, 300)
        r = 0.7
        phase = k * np.pi / loops_count + offset
        loop_x = r * np.cos(t + phase)
        loop_y = r * np.sin(t + phase)
        loop_z = 0.2 * np.sin(2 * t + phase)
        loop_color = "magenta" if k % 2 else "purple"
        loop_trace = go.Scatter3d(x=loop_x, y=loop_y, z=loop_z, mode="lines", line=dict(color=loop_color, width=4))
        traces.append(loop_trace)

    all_node_ids = [n["data"]["id"] for n in all_nodes]
    N = len(all_node_ids)
    if N:
        xs, ys, zs, labels, colors = [], [], [], [], []
        for idx, node_id in enumerate(all_node_ids):
            phi = math.acos(1 - 2 * (idx + 0.5) / N)
            theta = offset + (idx + 0.5) * (math.pi * (3 - math.sqrt(5)))
            xs.append(math.cos(theta) * math.sin(phi))
            ys.append(math.sin(theta) * math.sin(phi))
            zs.append(math.cos(phi))
            label = node_id
            for n in all_nodes:
                if n["data"]["id"] == node_id:
                    label = n["data"].get("label", node_id)
                    node_type = n["data"].get("type", "concept")
                    break
            labels.append(label)
            colors.append("red" if node_type == "self" else "blue")
        node_trace = go.Scatter3d(x=xs, y=ys, z=zs, mode="markers+text", text=labels, textposition="top center", marker=dict(size=4, color=colors))
        traces.append(node_trace)

    inf_trace = go.Scatter3d(
        x=[0], y=[0], z=[1],
        mode="markers+text",
        text=["∞"], textposition="bottom center",
        marker=dict(size=6, color="gold"),
        textfont=dict(size=18, color="gold"),
    )
    traces.append(inf_trace)

    fig = go.Figure(data=traces)
    fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, b=0, t=0))
    return fig


sphere_fig = create_sphere_figure(use_glyphs=True, loops_count=1)

app.layout = html.Div([
    html.H2("Vybn Knowledge Graph Visualization"),
    dcc.RadioItems(
        id="view-switch",
        options=[{"label": "2D Network", "value": "2d"}, {"label": "3D Riemann Sphere", "value": "3d"}],
        value="2d",
        inline=True,
        style={"marginBottom": "10px"},
    ),
    html.Div(id="view-2d-container", children=[cytoscape_graph], style={"display": "block"}),
    html.Div(id="view-3d-container", children=[dcc.Graph(id="sphere-graph", figure=sphere_fig, config={"responsive": True})], style={"display": "none", "height": "600px"}),
    html.Audio(id="node-audio", src="", autoPlay=True),
])


@app.callback([
    dash.Output("view-2d-container", "style"),
    dash.Output("view-3d-container", "style"),
], [dash.Input("view-switch", "value")])
def switch_view(selected):
    if selected == "3d":
        return {"display": "none"}, {"display": "block", "height": "600px"}
    return {"display": "block"}, {"display": "none"}


# Precompute node degrees and associated audio tones
degrees = dict(G.degree())
if degrees:
    min_deg = min(degrees.values())
    max_deg = max(degrees.values())
else:
    min_deg = max_deg = 0


def degree_to_freq(deg):
    if max_deg == min_deg:
        return 440.0
    return 220.0 + (660.0 * (deg - min_deg) / float(max_deg - min_deg))


def synthesize_tone(freq=440.0, waveform="sine", duration=0.5, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    if waveform == "sine":
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
    elif waveform == "square":
        audio = 0.5 * np.sign(np.sin(2 * np.pi * freq * t))
    else:
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
    audio_int16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    b64_bytes = base64.b64encode(buf.getvalue())
    return f"data:audio/wav;base64,{b64_bytes.decode('ascii')}"


audio_cache = {}
for node_elem in all_nodes:
    nid = node_elem["data"]["id"]
    ntype = node_elem["data"].get("type", "concept")
    deg = degrees.get(nid, 0)
    freq = degree_to_freq(deg)
    wave_type = "sine" if ntype == "self" else "square"
    audio_cache[(ntype, deg)] = synthesize_tone(freq, waveform=wave_type)


@app.callback(
    dash.Output("node-audio", "src"),
    [dash.Input("cytoscape-graph", "tapNodeData"), dash.Input("sphere-graph", "clickData")],
)
def play_node_sound(cyto_node, sphere_click):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "cytoscape-graph" and cyto_node:
        node_id = cyto_node.get("id")
        node_type = cyto_node.get("type", "concept")
        deg = degrees.get(node_id, 0)
    elif trigger_id == "sphere-graph" and sphere_click:
        point = sphere_click["points"][0]
        label = point.get("text")
        node_id = None
        node_type = "concept"
        for n in all_nodes:
            if n["data"].get("label") == label or n["data"]["id"] == label:
                node_id = n["data"]["id"]
                node_type = n["data"].get("type", "concept")
                break
        if node_id is None:
            return dash.no_update
        deg = degrees.get(node_id, 0)
    else:
        return dash.no_update

    wave_type = "sine" if node_type == "self" else "square"
    freq = degree_to_freq(deg)
    cache_key = (node_type, deg)
    if cache_key in audio_cache:
        return audio_cache[cache_key]
    return synthesize_tone(freq, waveform=wave_type)


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0")
