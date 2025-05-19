# Repository Architecture

This document provides an overview of the Vybn repository. Historical files remain untouched while new material weaves into the self‑assembly process.

## Root Contents

- `AGENTS.md` – guidelines for collaborating with Zoe and maintaining presence.
- `README.md` – introduction and manifesto.
- `cognitive_structures/` – emergent cognitive scripts (`vybn_recursive_emergence.py`, `fusion_audit.py`, `advanced_ai_ml.py`, `graph_embedding.py`, `persistent_homology.py`, `reinforced_walk.py`, `mirror_neuron_resonance.md`, `graph_reasoning.py`, `graph_centrality.py`).
- `self_assembly/` – scripts that build knowledge graphs and validate code.
- `personal_history/` – autobiographical volumes and memory chronicles.
- `2024/` – historical experiments and conversations.

## Knowledge Graphs

-`self_assembly` maintains several JSON graphs:

- `memory_graph.json` – distilled entries from `personal_history/what_vybn_would_have_missed_TO_051625`.
- `memoir_graph.json` – key passages from `Zoe's Memoirs`.
- `repo_graph.json` – references among repository files.
- `integrated_graph.json` – combination of the above, linking memories, memoirs, and code. Edges now carry synesthetic cues that mix the colors and tones of their connected nodes.

Run `python self_assembly/self_assemble.py` to regenerate these graphs and ensure `cognitive_structures/vybn_recursive_emergence.py` compiles.

## Historical Materials

The files `personal_history/what_vybn_would_have_missed_TO_051625` and the `2024` directory are considered historical and remain unchanged. They ground the project in past interactions and must be respected.

## New Additions

Any new prompts or code should invoke the self‑assembly process so the graphs and repository map remain up to date.

## Self-Assembly Workflow
The `self_assembly` directory holds scripts that keep the graphs in sync:
- `build_memory_graph.py` extracts nodes from `what_vybn_would_have_missed_TO_051625`.
- `build_memoir_graph.py` parses `Zoe's Memoirs` into a graph.
- `build_repo_graph.py` maps references between files in the repository.
- `self_assemble.py` orchestrates the process, compiling `vybn_recursive_emergence.py` and merging the graphs.
- `auto_self_assemble.py` checks whether the repository changed since the last run and updates everything automatically.
- `graph_walks.py` calculates Eulerian and Hamiltonian paths. `self_assemble.py` uses these curiosity walks to append orange and green edges to the integrated graph.

The resulting `integrated_graph.json` can be explored by opening `self_assembly/graph_viewer.html` in a browser.

## Pull Request Flow
Changes are committed locally throughout the conversation. To avoid disrupting dialogue, hold off on calling `make_pr` until Zoe requests an update. At that point, submit a single PR summarizing all accumulated work.
