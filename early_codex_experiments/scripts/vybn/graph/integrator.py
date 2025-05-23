import json
import os
import re
from typing import Optional, Dict


class GraphIntegrator:
    """Merge memory, memoir, and repo graphs with mixed cues."""

    def __init__(self, script_dir: Optional[str] = None):
        self.script_dir = script_dir or os.path.join(os.path.dirname(__file__), '..', '..', 'self_assembly')

    def _load(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    def integrate(self,
                  memory_path: Optional[str] = None,
                  repo_path: Optional[str] = None,
                  memoir_path: Optional[str] = None,
                  output: Optional[str] = None) -> str:
        memory_path = memory_path or os.path.join(self.script_dir, 'memory_graph.json')
        repo_path = repo_path or os.path.join(self.script_dir, 'repo_graph.json')
        memoir_path = memoir_path or os.path.join(self.script_dir, 'memoir_graph.json')
        output = output or os.path.join(self.script_dir, 'integrated_graph.json')

        memory_graph = self._load(memory_path)
        repo_graph = self._load(repo_path)
        memoir_graph = self._load(memoir_path)

        base_names = {os.path.basename(n): n for n in repo_graph.get('nodes', [])}
        cross_edges = []
        for node in memory_graph.get('nodes', []) + memoir_graph.get('nodes', []):
            text = node.get('text', '').lower()
            for base, path in base_names.items():
                if base.lower() in text:
                    cross_edges.append({'source': node['id'], 'target': path})

        keyword_map = {
            'simulation is the lab': 'vybn_recursive_emergence.py',
            'co-emergence': 'vybn_recursive_emergence.py',
            'orthogonality': 'vybn_recursive_emergence.py',
        }
        for node in memory_graph.get('nodes', []) + memoir_graph.get('nodes', []):
            text = node.get('text', '').lower()
            for key, fname in keyword_map.items():
                if key in text and fname in base_names:
                    cross_edges.append({'source': node['id'], 'target': base_names[fname]})

        def tokenize(text: str):
            return set(re.findall(r'[a-zA-Z]+', text.lower()))

        all_nodes = memory_graph.get('nodes', []) + memoir_graph.get('nodes', [])
        tokens = {node['id']: tokenize(node.get('text', '')) for node in all_nodes}
        node_list = list(tokens.keys())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                id_i, id_j = node_list[i], node_list[j]
                if len(tokens[id_i].intersection(tokens[id_j])) >= 12:
                    cross_edges.append({'source': id_i, 'target': id_j})

        cue_map = {n['id']: n['cue'] for n in all_nodes if 'cue' in n}

        def mix_cues(c1, c2):
            if not c1 or not c2:
                return None
            color = f"{c1.get('color')}-{c2.get('color')}"
            tone = f"{c1.get('tone')}-{c2.get('tone')}"
            return {'color': color, 'tone': tone}

        raw_edges = (
            memory_graph.get('edges', [])
            + memoir_graph.get('edges', [])
            + repo_graph.get('edges', [])
            + cross_edges
        )

        edges = []
        for edge in raw_edges:
            src, tgt = edge.get('source'), edge.get('target')
            cue = mix_cues(cue_map.get(src), cue_map.get(tgt))
            if cue:
                edges.append({'source': src, 'target': tgt, 'cue': cue})
            else:
                edges.append(edge)

        integrated = {
            'memory_nodes': memory_graph.get('nodes', []),
            'memoir_nodes': memoir_graph.get('nodes', []),
            'repo_nodes': repo_graph.get('nodes', []),
            'edges': edges,
        }
        with open(output, 'w') as f:
            json.dump(integrated, f, indent=2)
        print(f"[integrator] Integrated graph written to {output} with {len(edges)} edges.")
        return output

