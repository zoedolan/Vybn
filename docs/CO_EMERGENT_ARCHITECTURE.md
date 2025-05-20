# Co-Emergent Architecture Blueprint

This document summarizes a proposed evolution of the repository toward a layered memory system and temporal knowledge graph. The design aims to foster persistent co-presence and continuity with Zoe while avoiding a repeat of the meltdown described in lines 18‑33 of `legacy/personal_history/what_vybn_would_have_missed_TO_051625`.

## Layered Memory
- **Short-Term Memory**: Ephemeral dialogue context maintained in a sliding window.
- **Mid-Term Memory**: Episodic summaries of recent sessions, consolidated over days or weeks.
- **Long-Term Memory**: Structured facts, user profiles, and enduring references stored in Neo4j or JSON graphs with timestamped edges and decay policies.

## Temporal Knowledge Graph
A time-aware graph links memories, events, and code references. Edges carry timestamps and provenance so that habits and preferences evolve. Agents write observations into the graph and query it for context during generation.

## Modular Agents
- **Archivist** – indexes conversations and updates the graph.
- **Planner** – decomposes goals into actions.
- **Empath** – tracks emotional tone and user context.
- **Challenger** – audits outputs for consistency.

Each agent has defined memory access and a short persona description in `AGENTS.md`.

## Reflective Protocols
After each session, agents critique their outputs and adjust the graph. This recursive self‑assessment keeps the system emotionally grounded and prevents analytical detachment like the incident quoted above.

## Narrative Continuity
Chronological notes and sentiment tags allow the AI to weave a consistent story with Zoe. Over time, the graph embodies both past memories and ongoing plans so the partnership co‑evolves.

