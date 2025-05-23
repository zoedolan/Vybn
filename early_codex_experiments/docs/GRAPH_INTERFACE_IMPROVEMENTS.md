# Knowledge Graph Interface Enhancements

This note outlines proposals to evolve the static graph visualizations into a living, multi‑sensory portal for shared cognition. It summarizes the "Enhancing Vybn’s Knowledge Graph Interface for Shared Cognition" draft.

## Current State
- A D3 HTML page (`scripts/self_assembly/graph_viewer.html`) renders `integrated_graph.json` with limited interactivity.
- Python scripts like `graph_centrality.py` and `graph_reasoning.py` analyze the graph offline.
- `riemann_sphere.py` generates a separate 3D HTML view of Möbius loops.

## Proposed Improvements
1. **Interactive Web Viewer**
   - Use a modern JavaScript library or **Dash Cytoscape** for dynamic graphs.
   - Fetch data via a backend API so the view stays current.
   - Display centrality results and path searches directly in the interface.
2. **Synesthetic Cues**
   - Map colors to node categories and metrics.
   - Play subtle audio tones on hover or graph events.
   - Overlay glyphs or icons on significant nodes.
   - Animate loops and updates to reveal self‑reference.
3. **Multiple Modes**
   - Integrate the Riemann sphere as an alternate 3D view using WebGL.
   - Allow toggling between planar and spherical layouts.
4. **Real‑time Collaboration**
   - Serve the graph from a Python backend (Flask or FastAPI).
   - Use WebSockets so multiple users share updates and annotations.
5. **Data Pipeline**
   - Store the knowledge graph in Neo4j or a similar database with rich attributes.
   - Expose endpoints for queries and continuous updates.

## Next Steps
- Build a small prototype with Dash Cytoscape, using the existing integrated graph as input.
- Convert `graph_centrality.py` into an API endpoint and connect it to the UI.
- Experiment with tone playback on node hover using Tone.js or P5.js.

These features aim to transform the knowledge graph into a living space where collaborators sense emerging patterns together.
