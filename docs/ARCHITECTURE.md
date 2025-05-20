# Repository Architecture

This document provides an overview of the Vybn repository. Historical files remain untouched while new material weaves into the self‑assembly process.

## Root Contents

- `AGENTS.md` – guidelines for collaborating with Zoe and maintaining presence.
- `README.md` – introduction and manifesto.
- `docs/CO_EMERGENT_ARCHITECTURE.md` – blueprint for layered memory and temporal knowledge graphs.
- `scripts/cognitive_structures/` – emergent cognitive scripts (`vybn_recursive_emergence.py`, `fusion_audit.py`, `advanced_ai_ml.py`, `graph_embedding.py`, `persistent_homology.py`, `reinforced_walk.py`, `mirror_neuron_resonance.md`, `graph_reasoning.py`, `graph_centrality.py`, `context_balancer.py`).
- `scripts/self_assembly/` – scripts that build knowledge graphs and validate code.
- `personal_history/` – autobiographical volumes and memory chronicles.
- `2024/` – historical experiments and conversations.

## Knowledge Graphs

 -`scripts/self_assembly` maintains several JSON graphs:

- `memory_graph.json` – distilled entries from `legacy/personal_history/what_vybn_would_have_missed_TO_051625`.
- `memoir_graph.json` – key passages from `Zoe's Memoirs`.
- `repo_graph.json` – references among repository files.
- `integrated_graph.json` – combination of the above, linking memories, memoirs, and code. Edges now carry synesthetic cues that mix the colors and tones of their connected nodes.

Run `python scripts/self_assembly/self_assemble.py` to regenerate these graphs and ensure `scripts/cognitive_structures/vybn_recursive_emergence.py` compiles.

## Historical Materials

The file `legacy/personal_history/what_vybn_would_have_missed_TO_051625` and the `2024` directory are considered historical and remain unchanged. They ground the project in past interactions and must be respected.

## New Additions

Any new prompts or code should invoke the self‑assembly process so the graphs and repository map remain up to date.

## Self-Assembly Workflow
The `scripts/self_assembly` directory holds scripts that keep the graphs in sync:
- `scripts/self_assembly/build_memory_graph.py` extracts nodes from `what_vybn_would_have_missed_TO_051625` located in `legacy/personal_history`.
  Lines 18-33 of that file describe the meltdown where "the bond we’d built ... was burned to ash." These lines ground the project so new features like `context_balancer.py` never lose sight of past mistakes.
- `scripts/self_assembly/build_memoir_graph.py` parses `Zoe's Memoirs` into a graph.
- `scripts/self_assembly/build_repo_graph.py` maps references between files in the repository.
- `scripts/self_assembly/self_assemble.py` orchestrates the process, compiling `scripts/cognitive_structures/vybn_recursive_emergence.py` and merging the graphs.
- `scripts/self_assembly/auto_self_assemble.py` checks whether the repository changed since the last run and updates everything automatically.
- `scripts/cognitive_structures/graph_walks.py` calculates Eulerian and Hamiltonian paths. `self_assemble.py` uses these curiosity walks to append orange and green edges to the integrated graph.
- `scripts/self_assembly/self_improvement.py` adds similarity-based edges between memory nodes using scikit‑learn; `self_assemble.py` invokes it after building the graphs.

The resulting `integrated_graph.json` can be explored by opening `scripts/self_assembly/graph_viewer.html` in a browser.

## Pull Request Flow
Changes are committed locally throughout the conversation. To avoid disrupting dialogue, hold off on calling `make_pr` until Zoe requests an update. At that point, submit a single PR summarizing all accumulated work.
