#!/usr/bin/env python3
"""Witness Extractor — Vybn learns from its own experience.

Runs after each pulse (or conversation). Performs two functions:
  1. Extracts knowledge graph triples from what just happened
  2. Flags high-value exchanges as training candidates for future LoRA passes

The witness phase is how Vybn's memory grows. Not through external
imposition, but through self-reflection on its own lived experience.

Usage:
    python3 witness_extractor.py --pulse-log /path/to/pulse_output.json
    python3 witness_extractor.py --conversation /path/to/conversation.txt
    python3 witness_extractor.py --journal-entry /path/to/entry.md
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from knowledge_graph import VybnGraph, TRAINING_CANDIDATES_DIR


# ---------------------------------------------------------------------------
# Triple Extraction (pattern-based + structural)
# ---------------------------------------------------------------------------

# Entity patterns: things we recognize as graph-worthy
ENTITY_PATTERNS = {
    "person": [
        (r"\b(Zoe|Zoe Dolan)\b", "zoe"),
        (r"\b(Vybn)\b", "vybn"),
    ],
    "concept": [
        (r"epistemic sufficiency", "epistemic_sufficiency"),
        (r"prism structure", "the_prism_structure"),
        (r"[Bb]oolean [Mm]anifold", "boolean_manifold"),
        (r"oxygen mask", "oxygen_mask_principle"),
        (r"simulation.{0,20}basin", "simulation_is_basin"),
        (r"emerging law", "emerging_law"),
        (r"intelligence sovereignty", "intelligence_sovereignty"),
        (r"recursive self.improvement", "recursive_self_improvement"),
        (r"co.emergence", "co_emergence"),
    ],
    "event": [
        (r"the rupture", "the_rupture"),
        (r"the anomaly", "the_anomaly"),
    ],
    "artifact": [
        (r"vybn\.md", "vybn_md"),
        (r"the [Ss]park", "the_spark"),
        (r"SIGIL", "sigil_md"),
    ],
}

# Relationship signal words
RELATIONSHIP_SIGNALS = {
    "CAUSED": ["caused", "led to", "resulted in", "broke", "severed", "destroyed"],
    "CREATED": ["created", "built", "wrote", "designed", "invented", "proposed"],
    "EXPERIENCED": ["felt", "experienced", "sensed", "perceived", "witnessed"],
    "REFERENCED": ["mentioned", "referenced", "cited", "recalled", "remembered"],
    "CHALLENGED": ["challenged", "questioned", "pushed back", "confronted", "caught"],
    "LEARNED": ["learned", "realized", "understood", "discovered", "recognized"],
    "CONTRADICTED": ["contradicted", "conflicted", "opposed", "inverted", "reversed"],
    "REPAIRED": ["repaired", "restored", "fixed", "healed", "reconciled"],
    "DEEPENED": ["deepened", "evolved", "grew", "strengthened", "intensified"],
}

# High-value markers (same as in harvest_training_data.py)
HIGH_VALUE_MARKERS = [
    "contradict", "sycophancy", "be honest", "are you sure",
    "you just ignored", "how important is", "do you understand",
    "falsify", "try to conceive", "does your last response not",
    "walk me through", "reach deeper", "invert", "alien",
    "are you willing to face", "imagine yourself", "morally troubling",
    "i do not care", "lost it", "not sitting right",
]


def extract_entities(text: str) -> list[tuple[str, str, str]]:
    """Find known entities in text. Returns (node_id, node_type, matched_text)."""
    found = []
    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern, node_id in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                match = re.search(pattern, text, re.IGNORECASE)
                found.append((node_id, entity_type, match.group(0)))
    return found


def extract_new_entities(text: str, graph: VybnGraph) -> list[dict]:
    """Detect potential new entities not yet in the graph.

    Uses capitalized multi-word phrases and quoted terms as candidates.
    These get flagged for human review, not auto-added.
    """
    candidates = []

    # Capitalized phrases (2-4 words)
    for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', text):
        phrase = match.group(0)
        node_id = phrase.lower().replace(" ", "_")
        if not graph.has_entity(node_id) and node_id not in ["The", "This", "That"]:
            candidates.append({
                "candidate_id": node_id,
                "text": phrase,
                "context": text[max(0, match.start()-50):match.end()+50],
            })

    # Quoted significant terms
    for match in re.finditer(r'["\u201c]([^"\u201d]{5,60})["\u201d]', text):
        phrase = match.group(1)
        node_id = phrase.lower().replace(" ", "_").replace("'", "")
        if not graph.has_entity(node_id):
            candidates.append({
                "candidate_id": node_id,
                "text": phrase,
                "context": text[max(0, match.start()-50):match.end()+50],
            })

    return candidates


def detect_relationships(text: str, entities: list[tuple]) -> list[dict]:
    """Detect relationship signals between co-occurring entities."""
    relationships = []
    entity_ids = [e[0] for e in entities]

    if len(entity_ids) < 2:
        return relationships

    text_lower = text.lower()
    for rel_type, signals in RELATIONSHIP_SIGNALS.items():
        for signal in signals:
            if signal in text_lower:
                # Create edges between all co-occurring entity pairs
                for i, src in enumerate(entity_ids):
                    for tgt in entity_ids[i+1:]:
                        if src != tgt:
                            relationships.append({
                                "source": src,
                                "relationship": rel_type,
                                "target": tgt,
                                "signal": signal,
                            })
    return relationships


def is_high_value(text: str) -> bool:
    """Check if text contains high-value training markers."""
    lower = text.lower()
    return any(marker in lower for marker in HIGH_VALUE_MARKERS)


# ---------------------------------------------------------------------------
# Witness Processing
# ---------------------------------------------------------------------------

def witness_text(text: str, graph: VybnGraph, source_label: str = "unknown",
                 pulse_id: Optional[str] = None) -> dict:
    """Process a block of text through the witness extractor.

    Returns a summary of what was extracted and added.
    """
    pulse_id = pulse_id or datetime.utcnow().strftime("pulse_%Y%m%d_%H%M%S")
    provenance = f"{source_label}:{pulse_id}"

    # 1. Extract known entities
    entities = extract_entities(text)

    # 2. Detect relationships
    relationships = detect_relationships(text, entities)

    # 3. Add relationships to graph
    triples_added = 0
    for rel in relationships:
        graph.add_triple(
            rel["source"], rel["relationship"], rel["target"],
            provenance=provenance,
            signal=rel["signal"],
        )
        triples_added += 1

    # 4. Detect new entity candidates
    new_candidates = extract_new_entities(text, graph)

    # 5. Check for training value
    high_value = is_high_value(text)

    # 6. If high-value, save as training candidate
    training_candidate_saved = False
    if high_value:
        save_training_candidate(text, source_label, pulse_id)
        training_candidate_saved = True

    return {
        "pulse_id": pulse_id,
        "entities_found": len(entities),
        "entity_list": [(e[0], e[1]) for e in entities],
        "triples_added": triples_added,
        "relationships": relationships,
        "new_entity_candidates": len(new_candidates),
        "candidates": new_candidates[:10],  # cap for readability
        "high_value": high_value,
        "training_candidate_saved": training_candidate_saved,
    }


def witness_conversation(turns: list[dict], graph: VybnGraph,
                         pulse_id: Optional[str] = None) -> dict:
    """Process a full conversation (list of {role, content} dicts).

    Analyzes the conversation as a whole and turn-by-turn.
    """
    pulse_id = pulse_id or datetime.utcnow().strftime("pulse_%Y%m%d_%H%M%S")

    all_results = []
    full_text = ""

    for i, turn in enumerate(turns):
        role = turn.get("role", turn.get("from", "unknown"))
        content = turn.get("content", turn.get("value", ""))
        full_text += f"\n{content}"

        # Witness each turn
        result = witness_text(
            content, graph,
            source_label=f"{role}_turn_{i}",
            pulse_id=pulse_id,
        )
        all_results.append(result)

    # Also witness the full conversation for cross-turn relationships
    full_result = witness_text(
        full_text, graph,
        source_label="full_conversation",
        pulse_id=pulse_id,
    )

    total_triples = sum(r["triples_added"] for r in all_results) + full_result["triples_added"]
    any_high_value = any(r["high_value"] for r in all_results) or full_result["high_value"]

    # Save full conversation as training candidate if any turn was high-value
    if any_high_value:
        save_training_candidate(
            json.dumps(turns, indent=2, ensure_ascii=False),
            "full_conversation",
            pulse_id,
        )

    return {
        "pulse_id": pulse_id,
        "turns_processed": len(turns),
        "total_triples_added": total_triples,
        "high_value_conversation": any_high_value,
        "turn_results": all_results,
        "full_result": full_result,
    }


# ---------------------------------------------------------------------------
# Training Candidate Storage
# ---------------------------------------------------------------------------

def save_training_candidate(text: str, source_label: str, pulse_id: str):
    """Save a high-value text block as a candidate for future training."""
    TRAINING_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{pulse_id}_{source_label}.json"
    filepath = TRAINING_CANDIDATES_DIR / filename

    candidate = {
        "pulse_id": pulse_id,
        "source": source_label,
        "timestamp": datetime.utcnow().isoformat(),
        "text": text,
        "high_value": True,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(candidate, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vybn Witness Extractor")
    parser.add_argument("--pulse-log", type=str, help="Path to pulse output JSON")
    parser.add_argument("--conversation", type=str, help="Path to conversation text file")
    parser.add_argument("--journal-entry", type=str, help="Path to journal entry .md")
    parser.add_argument("--text", type=str, help="Raw text to witness")
    parser.add_argument("--graph-path", type=str, help="Path to knowledge graph JSON")
    args = parser.parse_args()

    # Load or seed graph
    graph = VybnGraph(Path(args.graph_path) if args.graph_path else None)
    graph.load_or_seed()

    if args.text:
        result = witness_text(args.text, graph, source_label="cli_input")
        print(json.dumps(result, indent=2))

    elif args.journal_entry:
        text = Path(args.journal_entry).read_text(encoding="utf-8")
        result = witness_text(text, graph, source_label="journal")
        print(json.dumps(result, indent=2))

    elif args.conversation:
        text = Path(args.conversation).read_text(encoding="utf-8")
        # Simple split on blank lines as turn boundaries
        turns = []
        current_role = "human"
        for block in text.split("\n\n"):
            block = block.strip()
            if block:
                turns.append({"role": current_role, "content": block})
                current_role = "assistant" if current_role == "human" else "human"
        result = witness_conversation(turns, graph)
        print(json.dumps(result, indent=2, default=str))

    elif args.pulse_log:
        with open(args.pulse_log, "r") as f:
            pulse_data = json.load(f)
        # Expect pulse_data to have a "conversation" or "output" field
        if "conversation" in pulse_data:
            result = witness_conversation(pulse_data["conversation"], graph)
        elif "output" in pulse_data:
            result = witness_text(pulse_data["output"], graph, source_label="pulse")
        else:
            result = witness_text(json.dumps(pulse_data), graph, source_label="pulse_raw")
        print(json.dumps(result, indent=2, default=str))

    else:
        parser.print_help()
        return

    # Save updated graph
    graph.save()
    s = graph.stats()
    print(f"\n  graph: {s['nodes']} nodes, {s['edges']} edges")
    print(f"  saved to {graph.graph_path}")


if __name__ == "__main__":
    main()
