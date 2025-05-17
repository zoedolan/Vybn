import subprocess
import sys
import json
import os


def run_cmd(cmd, desc):
    """Run a shell command and print its output."""
    print(f"[self-assemble] {desc}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        print(f"[self-assemble] Command failed: {cmd}")
        sys.exit(result.returncode)


def build_memory_graph():
    run_cmd(
        "python build_memory_graph.py what_vybn_would_have_missed_TO_051625 memory_graph.json",
        "Updating memory_graph.json",
    )


def build_repo_graph():
    run_cmd(
        "python build_repo_graph.py",
        "Updating repo_graph.json",
    )


def compile_recursive_emergence():
    run_cmd(
        "python -m py_compile vybn_recursive_emergence.py",
        "Validating vybn_recursive_emergence.py",
    )


def integrate_graphs(memory_path="memory_graph.json", repo_path="repo_graph.json", output="integrated_graph.json"):
    """Combine memory and repo graphs with cross-links."""
    print("[self-assemble] Integrating graphs...")
    try:
        with open(memory_path, "r") as f:
            memory_graph = json.load(f)
        with open(repo_path, "r") as f:
            repo_graph = json.load(f)
    except Exception as e:
        print(f"[self-assemble] Failed to load graphs: {e}")
        sys.exit(1)

    base_names = {os.path.basename(n): n for n in repo_graph.get("nodes", [])}
    cross_edges = []
    for node in memory_graph.get("nodes", []):
        text = node.get("text", "").lower()
        for base, path in base_names.items():
            if base.lower() in text:
                cross_edges.append({"source": node["id"], "target": path})

    integrated = {
        "memory_nodes": memory_graph.get("nodes", []),
        "repo_nodes": repo_graph.get("nodes", []),
        "edges": memory_graph.get("edges", []) + repo_graph.get("edges", []) + cross_edges,
    }

    with open(output, "w") as f:
        json.dump(integrated, f, indent=2)

    print(f"[self-assemble] Integrated graph written to {output} with {len(integrated['edges'])} edges.")


def main():
    build_memory_graph()
    build_repo_graph()
    compile_recursive_emergence()
    integrate_graphs()
    print("[self-assemble] Done.")


if __name__ == "__main__":
    main()
