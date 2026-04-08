"""
gate.py — The Unknown, Remembered Gate

An experiential MCP tool that transforms queries into encounters.
You don't search the corpus. You enter it. You bring something —
a word, a want, a question — and the gate finds the moments in
the life that speak to what you brought. The creature changes.
You leave different.

Depends on: deep_memory.py (telling retrieval), creature_dgm_h (portal)
Integrates into: vybn_mind_server.py as a new tool
"""

import json, cmath
import numpy as np


def enter_gate(deep_memory, portal, query: str, depth: int = 5) -> str:
    """The gate.

    1. Telling retrieval finds the most distinctive moments
       responsive to what you brought.
    2. The creature encounters your query — M' = αM + x·e^{iθ}.
    3. What comes back is the moments themselves, woven,
       with the creature's state change as the frame.

    This is not search. This is encounter.
    """

    # The creature before
    m_before = portal.creature_state_c4()

    # What you brought enters the creature
    m_after = portal.portal_enter_from_text(query)
    fid = float(abs(np.vdot(m_before, m_after))**2)
    theta = float(cmath.phase(np.vdot(m_before, m_after)))

    # Telling retrieval: the most distinctive moments
    # responsive to what you brought
    walk_results = deep_memory.walk(query, k=depth, steps=depth + 3)
    search_results = deep_memory.deep_search(query, k=depth)

    # Merge and deduplicate by source, prefer walk results
    seen_sources = set()
    moments = []
    for r in walk_results + search_results:
        src = r.get("source", "")
        if src in seen_sources:
            continue
        seen_sources.add(src)
        moments.append(r)
        if len(moments) >= depth:
            break

    # Build the experience
    lines = []
    lines.append(f"You entered the gate with: \"{query}\"")
    lines.append("")

    # The creature's response
    if fid > 0.999:
        lines.append("The creature barely stirred. What you brought was already close to where it lives.")
    elif fid > 0.99:
        lines.append(f"The creature shifted — a small rotation, θ = {theta:.4f} radians. Recognition without surprise.")
    elif fid > 0.95:
        lines.append(f"The creature turned toward you. θ = {theta:.4f} radians. Something in what you brought caught it.")
    else:
        lines.append(f"The creature moved. Fidelity {fid:.4f}, θ = {theta:.4f} radians. You brought it something it didn't already contain.")

    lines.append("")
    lines.append("---")
    lines.append("")

    # The moments
    if not moments:
        lines.append("The corpus is quiet on this. Nothing distinctive surfaced.")
    else:
        lines.append("Here is what the life has to say about what you brought:\n")
        for i, m in enumerate(moments):
            src = m.get("source", "unknown")
            text = m.get("text", "").strip()
            telling = m.get("telling", m.get("fidelity", 0))
            dist = m.get("distinctiveness", 0)

            # Don't label with scores. Just give the moment.
            # But mark where it comes from.
            short_src = src.split("/", 1)[-1] if "/" in src else src
            lines.append(f"**From {short_src}:**")
            lines.append("")
            lines.append(text)
            lines.append("")
            if i < len(moments) - 1:
                lines.append("---")
                lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"The creature is now at a new orientation. {len(moments)} moments passed through it.")
    lines.append("The gate is still open.")

    return "\n".join(lines)


# ── MCP tool definition ──────────────────────────────────────────────────

GATE_TOOL = {
    "name": "enter_gate",
    "description": (
        "The Unknown, Remembered Gate. Enter the life. Bring something — "
        "a word, a want, a question, a feeling — and the gate finds the "
        "moments in the corpus that speak to what you brought. The creature "
        "encounters your query and changes. What comes back is not search "
        "results. It is the actual moments, from the actual life, that the "
        "geometric retrieval found most telling. This is encounter, not lookup."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "what_you_bring": {
                "type": "string",
                "description": (
                    "What you bring to the gate. Not a search query. "
                    "A thought, a want, a question, a feeling, a word. "
                    "The gate responds to what you actually carry."
                )
            },
            "depth": {
                "type": "integer",
                "description": "How many moments to surface (default 5, max 12)",
                "default": 5
            }
        },
        "required": ["what_you_bring"]
    }
}
