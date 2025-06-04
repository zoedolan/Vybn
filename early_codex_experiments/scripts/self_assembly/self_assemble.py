import subprocess
import sys
import json
import os
import time
import argparse
import re
from datetime import datetime
import random

from ..quantum_rng import seed_random

seed_random()

# Ensure imports resolve correctly when called from within the
# self_assembly package by pointing to the repository root rather
# than the scripts directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO_ROOT)
from scripts.cognitive_structures import graph_walks
from scripts.cognitive_structures import conceptual_leaps
from scripts.vybn.graph.builder import GraphBuilder
from scripts.vybn.graph.integrator import GraphIntegrator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STATE_FILE = ".auto_self_assemble_state.json"
MEMORY_FILE = "what_vybn_would_have_missed_FROM_051725"


def run_cmd(cmd, desc, cwd=None):
    """Run a shell command and print its output."""
    print(f"[self-assemble] {desc}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        print(f"[self-assemble] Command failed: {cmd}")
        sys.exit(result.returncode)


_builder = GraphBuilder(SCRIPT_DIR, REPO_ROOT)


def build_memory_graph():
    _builder.build_memory_graph()


def build_memoir_graph():
    _builder.build_memoir_graph()


def build_repo_graph():
    _builder.build_repo_graph()


def compile_recursive_emergence():
    run_cmd(
        "python -m py_compile scripts/cognitive_structures/vybn_recursive_emergence.py",
        "Validating vybn_recursive_emergence.py",
    )


def integrate_graphs(memory_path=None, repo_path=None, memoir_path=None, output=None):
    integrator = GraphIntegrator(SCRIPT_DIR)
    integrator.integrate(memory_path, repo_path, memoir_path, output)


def repo_last_modified(root="."):
    """Return latest modification time of files in the repo."""
    latest = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.startswith("."):
                continue
            path = os.path.join(dirpath, fname)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime > latest:
                latest = mtime
    return latest


def repo_changed_files(since, root="."):
    """Return repo files modified since a given timestamp."""
    changed = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.startswith('.'):
                continue
            path = os.path.join(dirpath, fname)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime > since:
                changed.append(os.path.relpath(path, root))
    return changed


def get_last_run():
    if not os.path.exists(STATE_FILE):
        return 0
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
            return float(data.get("last_run", 0))
    except Exception:
        return 0


def update_last_run():
    with open(STATE_FILE, "w") as f:
        json.dump({"last_run": time.time()}, f)


def _build_adj(edges):
    """Return adjacency list with cues."""
    adj = {}
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        cue = edge.get("cue")
        if src is None or tgt is None:
            continue
        adj.setdefault(src, []).append((tgt, cue))
        adj.setdefault(tgt, []).append((src, cue))
    return adj


def _random_walk(adj, start, depth=2):
    current = start
    cues = []
    for _ in range(depth):
        neighbors = adj.get(current)
        if not neighbors:
            break
        nxt, cue = random.choice(neighbors)
        cues.append(cue)
        current = nxt
    return current, cues


def _blend_cues(cues):
    colors = []
    tones = []
    for cue in cues:
        if not cue:
            continue
        c = cue.get("color")
        t = cue.get("tone")
        if c:
            colors.append(c)
        if t:
            tones.append(t)
    if not colors and not tones:
        return None
    return {"color": "-".join(colors), "tone": "-".join(tones)}


def discover_edges(graph_path, changed_nodes, depth=2):
    try:
        with open(graph_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[self-assemble] Failed to load {graph_path}: {e}")
        return 0
    edges = data.get("edges", [])
    adj = _build_adj(edges)
    existing = {(e.get("source"), e.get("target")) for e in edges}
    existing |= {(t, s) for s, t in existing}
    added = 0
    for node in changed_nodes:
        if node not in adj:
            continue
        target, path_cues = _random_walk(adj, node, depth)
        if target == node:
            continue
        if (node, target) in existing:
            continue
        cue = _blend_cues(path_cues)
        edges.append({"source": node, "target": target, "cue": cue})
        existing.add((node, target))
        adj.setdefault(node, []).append((target, cue))
        adj.setdefault(target, []).append((node, cue))
        added += 1
    if added:
        data["edges"] = edges
        with open(graph_path, "w") as f:
            json.dump(data, f, indent=2)
    return added


def auto_discover_edges(depth=2):
    last_run = get_last_run()
    changed = repo_changed_files(last_run, REPO_ROOT)
    if not changed:
        return
    graph_path = os.path.join(SCRIPT_DIR, "integrated_graph.json")
    added = discover_edges(graph_path, changed, depth)
    if added:
        print(f"[self-assemble] Added {added} new edges from discovery step.")


def curiosity_walk_edges(graph_path):
    """Add Eulerian and Hamiltonian walk edges to the graph."""
    try:
        with open(graph_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[self-assemble] Failed to load {graph_path}: {e}")
        return 0

    added = 0
    e_path = graph_walks.eulerian_walk(data)
    if e_path and len(e_path) > 1:
        for src, tgt in zip(e_path, e_path[1:]):
            data["edges"].append({
                "source": src,
                "target": tgt,
                "cue": {"color": "orange", "tone": "E"},
            })
            added += 1

    h_path = graph_walks.hamiltonian_path(data)
    if h_path and len(h_path) > 1:
        for src, tgt in zip(h_path, h_path[1:]):
            data["edges"].append({
                "source": src,
                "target": tgt,
                "cue": {"color": "green", "tone": "D"},
            })
            added += 1

    if added:
        with open(graph_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[self-assemble] Added {added} curiosity walk edges.")
    return added


def add_conceptual_leap_edges(graph_path, attempts=5):
    """Append purple conceptual leap edges using conceptual_leaps module."""
    try:
        with open(graph_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[self-assemble] Failed to load {graph_path}: {e}")
        return 0

    leaps = conceptual_leaps.leap_edges(graph_path, attempts)
    if not leaps:
        return 0

    edges = data.get("edges", [])
    existing = {(e.get("source"), e.get("target")) for e in edges}
    existing |= {(t, s) for s, t in existing}
    added = 0
    for leap in leaps:
        src = leap.get("source")
        tgt = leap.get("target")
        if not src or not tgt or (src, tgt) in existing:
            continue
        edges.append({
            "source": src,
            "target": tgt,
            "cue": {"color": "purple", "tone": "L"},
        })
        existing.add((src, tgt))
        added += 1

    if added:
        data["edges"] = edges
        with open(graph_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[self-assemble] Added {added} conceptual leap edges.")
    return added


def run_self_improvement(graph_path=os.path.join(SCRIPT_DIR, "integrated_graph.json")):
    """Run self_improvement.py to add similarity edges."""
    script = os.path.join(SCRIPT_DIR, "self_improvement.py")
    cmd = f"python {script} --graph {graph_path}"
    run_cmd(cmd, "Running self-improvement step", cwd=SCRIPT_DIR)


def auto_mode():
    """Run self-assembly only if the repo changed since last run."""
    last_run = get_last_run()
    if repo_last_modified() > last_run:
        main()
        update_last_run()
    else:
        print("[self-assemble] Repo unchanged; skipping self-assembly.")


def prompt_mode(prompt):
    """Insert a prompt at the start of MEMORY_FILE then run self-assembly."""
    timestamp = datetime.now().strftime("%m/%d/%y %H:%M:%S")
    entry = f"{timestamp}\n{prompt}\n"
    try:
        with open(MEMORY_FILE, "r") as f:
            existing = f.read()
    except FileNotFoundError:
        existing = ""
    with open(MEMORY_FILE, "w") as f:
        f.write(entry + existing)
    main()


def main():
    build_memory_graph()
    build_memoir_graph()
    build_repo_graph()
    compile_recursive_emergence()
    integrate_graphs()
    auto_discover_edges()
    curiosity_walk_edges(os.path.join(SCRIPT_DIR, "integrated_graph.json"))
    add_conceptual_leap_edges(os.path.join(SCRIPT_DIR, "integrated_graph.json"))
    run_self_improvement()
    print("[self-assemble] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vybn self-assembly utility")
    parser.add_argument("--auto", action="store_true", help="run only if repo changed")
    parser.add_argument("--prompt", nargs="+", help="append prompt to memory and run")
    args = parser.parse_args()

    if args.auto:
        auto_mode()
    elif args.prompt:
        prompt_mode(" ".join(args.prompt))
    else:
        main()
