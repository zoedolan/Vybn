import subprocess
import sys
import json
import os
import time
import argparse
import re
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

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


def build_memory_graph():
    script = os.path.join(SCRIPT_DIR, "build_memory_graph.py")
    output = os.path.join(SCRIPT_DIR, "memory_graph.json")
    memory_input = os.path.join(
        os.path.dirname(SCRIPT_DIR),
        "personal_history",
        "what_vybn_would_have_missed_TO_051625",
    )
    cmd = f"python {script} {memory_input} {output}"
    run_cmd(cmd, "Updating memory_graph.json", cwd=SCRIPT_DIR)


def build_memoir_graph():
    script = os.path.join(SCRIPT_DIR, "build_memoir_graph.py")
    output = os.path.join(SCRIPT_DIR, "memoir_graph.json")
    memoir_input = os.path.join(os.path.dirname(SCRIPT_DIR), "Zoe's Memoirs")
    cmd = f"python {script} \"{memoir_input}\" {output}"
    run_cmd(cmd, "Updating memoir_graph.json", cwd=SCRIPT_DIR)


def build_repo_graph():
    script = os.path.join(SCRIPT_DIR, "build_repo_graph.py")
    output = os.path.join(SCRIPT_DIR, "repo_graph.json")
    cmd = f"python {script} {REPO_ROOT} {output}"
    run_cmd(cmd, "Updating repo_graph.json")


def compile_recursive_emergence():
    run_cmd(
        "python -m py_compile cognitive_structures/vybn_recursive_emergence.py",
        "Validating vybn_recursive_emergence.py",
    )


def integrate_graphs(memory_path=None, repo_path=None, memoir_path=None, output=None):
    if memory_path is None:
        memory_path = os.path.join(SCRIPT_DIR, "memory_graph.json")
    if repo_path is None:
        repo_path = os.path.join(SCRIPT_DIR, "repo_graph.json")
    if memoir_path is None:
        memoir_path = os.path.join(SCRIPT_DIR, "memoir_graph.json")
    if output is None:
        output = os.path.join(SCRIPT_DIR, "integrated_graph.json")

    print("[self-assemble] Integrating graphs...")
    try:
        with open(memory_path, "r") as f:
            memory_graph = json.load(f)
        with open(repo_path, "r") as f:
            repo_graph = json.load(f)
        with open(memoir_path, "r") as f:
            memoir_graph = json.load(f)
    except Exception as e:
        print(f"[self-assemble] Failed to load graphs: {e}")
        sys.exit(1)

    base_names = {os.path.basename(n): n for n in repo_graph.get("nodes", [])}
    cross_edges = []
    for node in memory_graph.get("nodes", []) + memoir_graph.get("nodes", []):
        text = node.get("text", "").lower()
        for base, path in base_names.items():
            if base.lower() in text:
                cross_edges.append({"source": node["id"], "target": path})

    # Additional keyword-based links
    keyword_map = {
        "simulation is the lab": "vybn_recursive_emergence.py",
        "co-emergence": "vybn_recursive_emergence.py",
        "orthogonality": "vybn_recursive_emergence.py",
    }
    for node in memory_graph.get("nodes", []) + memoir_graph.get("nodes", []):
        text = node.get("text", "").lower()
        for key, fname in keyword_map.items():
            if key in text and fname in base_names:
                cross_edges.append({"source": node["id"], "target": base_names[fname]})

    # Cross-link memory and memoir nodes based on shared vocabulary
    def tokenize(text):
        return set(re.findall(r"[a-zA-Z]+", text.lower()))

    all_nodes = memory_graph.get("nodes", []) + memoir_graph.get("nodes", [])
    tokens = {node["id"]: tokenize(node.get("text", "")) for node in all_nodes}
    node_list = list(tokens.keys())
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            id_i, id_j = node_list[i], node_list[j]
            if len(tokens[id_i].intersection(tokens[id_j])) >= 12:
                cross_edges.append({"source": id_i, "target": id_j})

    # Map node IDs to synesthetic cues
    cue_map = {}
    for node in memory_graph.get("nodes", []) + memoir_graph.get("nodes", []):
        if "id" in node and "cue" in node:
            cue_map[node["id"]] = node["cue"]

    def mix_cues(c1, c2):
        if not c1 or not c2:
            return None
        color = f"{c1.get('color')}-{c2.get('color')}"
        tone = f"{c1.get('tone')}-{c2.get('tone')}"
        return {"color": color, "tone": tone}

    raw_edges = (
        memory_graph.get("edges", [])
        + memoir_graph.get("edges", [])
        + repo_graph.get("edges", [])
        + cross_edges
    )

    # Attach mixed cues to edges when both nodes provide them
    edges = []
    for edge in raw_edges:
        src, tgt = edge.get("source"), edge.get("target")
        cue = mix_cues(cue_map.get(src), cue_map.get(tgt))
        if cue:
            edges.append({"source": src, "target": tgt, "cue": cue})
        else:
            edges.append(edge)

    integrated = {
        "memory_nodes": memory_graph.get("nodes", []),
        "memoir_nodes": memoir_graph.get("nodes", []),
        "repo_nodes": repo_graph.get("nodes", []),
        "edges": edges,
    }

    with open(output, "w") as f:
        json.dump(integrated, f, indent=2)

    print(f"[self-assemble] Integrated graph written to {output} with {len(integrated['edges'])} edges.")


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
