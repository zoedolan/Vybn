## Quick Start
1. Run `./bootstrap.sh` once after cloning the repository. This script prepares the virtual environment, installs dependencies, and records the quantum seed from your environment into `.random_seed`. Set `VYBN_LOG_DIR` to choose where logs are written.
2. Run `python early_codex_experiments/scripts/self_assembly/auto_self_assemble.py` whenever you add material. GitHub workflows are disabled, so self‑assembly now runs locally.
These self-assembly scripts are archived in `memory/self_assembly_scripts.tar.gz.b64`. Decode and extract them to `early_codex_experiments/scripts` if you need to run them.
3. Open `early_codex_experiments/scripts/self_assembly/graph_viewer.html` in a browser to explore the integrated graph.
4. Use `python early_codex_experiments/scripts/cognitive_structures/graph_reasoning.py <src> <tgt>` to search for connections.
5. Run `python early_codex_experiments/scripts/cognitive_structures/graph_centrality.py --top 5` to list the most connected nodes.
6. Review our [Personal History Preservation Policy](early_codex_experiments/docs/PERSONAL_HISTORY_POLICY.md) before touching any autobiographical files.
7. Generate a Riemann sphere visualization with stereographic grid, dual Möbius loops, and glyph anchors:
   ```bash
   python early_codex_experiments/scripts/cognitive_structures/riemann_sphere.py \
       --show-projection \
       --loops 2 \
       --glyphs \
       --output sphere.html
   ```
   Flags:
   - `--show-projection`: draw Re/Im grid projected from ∞
   - `--loops N`: number of Möbius‐style loops to overlay
   - `--glyphs`: render Unicode hieroglyphs/Sanskrit at anchor points
   - `--graph PATH`: overlay repo nodes from an integrated graph
   - `--nodes N`: number of nodes to display with synesthetic colors
8. Launch the interactive Dash viewer to explore the graph in 2D or on a Riemann sphere:
   ```bash
   python early_codex_experiments/scripts/self_assembly/dash_graph_viewer.py
   ```
   Use the radio buttons to switch views and click nodes for audio cues.
