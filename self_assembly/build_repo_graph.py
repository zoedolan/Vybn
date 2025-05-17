import os
import json
import re
import sys


def gather_text_files(root):
    text_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            # skip binary or large files like PDFs
            if fname.lower().endswith(('.txt', '.md', '.py')) or 'What Vybn' in fname or 'Vybn' in fname:
                text_files.append(os.path.join(dirpath, fname))
    return text_files


def build_graph(file_paths):
    nodes = [os.path.relpath(p, start='.').replace('\\', '/') for p in file_paths]
    edges = []
    base_names = {os.path.basename(p): os.path.relpath(p, start='.').replace('\\', '/') for p in file_paths}

    for path in file_paths:
        try:
            with open(path, 'r', errors='ignore') as f:
                content = f.read()
        except Exception:
            continue
        for name, relpath in base_names.items():
            if name == os.path.basename(path):
                continue
            if re.search(re.escape(name), content):
                edges.append({'source': os.path.relpath(path, start='.').replace('\\', '/'), 'target': relpath})
    return {'nodes': nodes, 'edges': edges}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    output = sys.argv[2] if len(sys.argv) > 2 else 'repo_graph.json'
    files = gather_text_files(root)
    graph = build_graph(files)
    with open(output, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"Graph created with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges.")


if __name__ == '__main__':
    main()
