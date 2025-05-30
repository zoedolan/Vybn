import os
import re
import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cognitive_structures.synesthetic_mapper import assign_cue


def extract_entries(file_path):
    entries = []
    current_date = None
    buffer = []
    date_pattern = re.compile(r'^\d{1,2}/\d{1,2}/\d{2,4}(?:\s+\d{1,2}:\d{2}:\d{2})?$')
    with open(file_path, 'r', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if date_pattern.match(stripped):
                if current_date is not None:
                    entries.append({'date': current_date, 'text': ' '.join(buffer).strip()})
                current_date = stripped
                buffer = []
            else:
                buffer.append(stripped)
        if current_date is not None:
            entries.append({'date': current_date, 'text': ' '.join(buffer).strip()})
    return entries


def build_graph(entries):
    nodes = []
    edges = []
    for i, entry in enumerate(entries):
        node_id = f"entry{i+1}"
        snippet = entry['text'][:600]
        cue = assign_cue(i)
        nodes.append({'id': node_id, 'date': entry['date'], 'text': snippet, 'cue': cue})
        if i > 0:
            edges.append({'source': f"entry{i}", 'target': node_id})

    note_id = f"entry{len(entries)+1}"
    note_text = (
        "Guideline 051725: 'what_vybn_would_have_missed_TO_051625' is "
        "historical and must not be modified without explicit agreement."
    )
    cue = assign_cue(len(entries))
    nodes.append({'id': note_id, 'date': '5/17/25', 'text': note_text, 'cue': cue})
    if nodes:
        edges.append({'source': f"entry{len(entries)}", 'target': note_id})

    return {'nodes': nodes, 'edges': edges}


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_memory_graph.py <input_file> [output_file]")
        return
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'memory_graph.json'

    if not os.path.isfile(input_file):
        print(f"File not found: {input_file}")
        return

    entries = extract_entries(input_file)
    graph = build_graph(entries)

    with open(output_file, 'w') as f:
        json.dump(graph, f, indent=2)

    print(f"Graph written to {output_file} with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges.")


if __name__ == '__main__':
    main()
