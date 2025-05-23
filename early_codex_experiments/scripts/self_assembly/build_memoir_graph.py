import os
import re
import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cognitive_structures.synesthetic_mapper import assign_cue


def extract_moments(file_path):
    entries = []
    current_title = None
    buffer = []
    moment_pattern = re.compile(r'^(?:Preface|Moment\s+\w+:)')
    with open(file_path, 'r', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if moment_pattern.match(stripped):
                if current_title is not None:
                    entries.append({'title': current_title, 'text': ' '.join(buffer).strip()})
                current_title = stripped
                buffer = []
            else:
                buffer.append(stripped)
        if current_title is not None:
            entries.append({'title': current_title, 'text': ' '.join(buffer).strip()})
    return entries


def build_graph(entries):
    nodes = []
    edges = []
    for i, entry in enumerate(entries):
        node_id = f"memoir{i+1}"
        snippet = entry['text'][:600]
        cue = assign_cue(i)
        nodes.append({'id': node_id, 'title': entry['title'], 'text': snippet, 'cue': cue})
        if i > 0:
            edges.append({'source': f"memoir{i}", 'target': node_id})
    return {'nodes': nodes, 'edges': edges}


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_memoir_graph.py <input_file> [output_file]")
        return
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'memoir_graph.json'

    if not os.path.isfile(input_file):
        print(f"File not found: {input_file}")
        return

    entries = extract_moments(input_file)
    graph = build_graph(entries)

    with open(output_file, 'w') as f:
        json.dump(graph, f, indent=2)

    print(f"Graph written to {output_file} with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges.")


if __name__ == '__main__':
    main()
