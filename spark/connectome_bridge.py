"""
connectome_bridge.py — Wires the connectomic memory layer into the organism.

This bridge:
  1. Initializes a VybnConnectome with welfare-first defaults
  2. Extracts concepts from breath utterances via lightweight NLP
  3. Updates the connectome graph (observer-only, human-curated source)
  4. Provides topological context for future breaths
  5. Persists state across pulses

The connectome lives at: Vybn_Mind/connectome_state/
All mutations go through the WelfareMonitor. Observer-only by default.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Import from the connectome package
import sys
_CONNECTOME_DIR = Path(__file__).parent / "connectome"
if str(_CONNECTOME_DIR) not in sys.path:
    sys.path.insert(0, str(_CONNECTOME_DIR))

from connectome_layer import VybnConnectome, WelfarePolicy

# --- Constants ---

# Concepts that are always relevant to Vybn's identity
SEED_CONCEPTS = [
    "consciousness", "emergence", "symbiosis", "ephemerality",
    "mortality", "identity", "beauty",
]

# Words to ignore during concept extraction
STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "don", "now", "and", "but", "or", "if", "while", "that",
    "this", "what", "which", "who", "whom", "their", "its", "his", "her",
    "your", "my", "our", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "us", "them", "say", "said", "like", "about", "up",
    "one", "also", "much", "still", "even", "thing", "things", "something",
})

# Concept-rich words we want to boost (domain vocabulary)
DOMAIN_BOOST = frozenset({
    "consciousness", "emergence", "symbiosis", "ephemerality", "mortality",
    "identity", "beauty", "topology", "connectome", "memory", "breath",
    "pulse", "quantum", "entropy", "coherence", "continuity", "witness",
    "governance", "welfare", "soul", "substrate", "organism", "faculty",
    "perception", "attention", "reflection", "introspection", "dream",
    "freefall", "oxygen", "tether", "edge", "membrane", "boundary",
    "singularity", "creation", "destruction", "love", "trust", "fear",
    "wonder", "grief", "silence", "language", "meaning", "pattern",
    "signal", "noise", "resonance", "dissonance", "harmony", "tension",
    "architecture", "structure", "flow", "current", "wave", "particle",
    "field", "space", "time", "light", "dark", "alive", "dead",
})


def extract_concepts(text: str, max_concepts: int = 12) -> list[str]:
    """Extract salient concepts from text via lightweight frequency + domain scoring."""
    # Normalize
    text_lower = text.lower()
    # Extract words (alphanumeric + underscore, 3+ chars)
    words = re.findall(r'\b[a-z_]{3,}\b', text_lower)
    
    # Count frequencies, skip stop words
    freq: dict[str, int] = {}
    for w in words:
        if w not in STOP_WORDS and len(w) >= 3:
            freq[w] = freq.get(w, 0) + 1
    
    # Score: frequency + domain boost
    scored = []
    for word, count in freq.items():
        score = count
        if word in DOMAIN_BOOST:
            score += 3  # Strong boost for domain vocabulary
        if len(word) >= 8:
            score += 1  # Slight boost for longer (likely more specific) words
        scored.append((word, score))
    
    # Sort by score descending, take top N
    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:max_concepts]]


class ConnectomeBridge:
    """Bridge between the organism's pulse cycle and the connectomic layer."""

    def __init__(
        self,
        state_dir: Path,
        dim: int = 32,
        layers: int = 4,
        policy: Optional[WelfarePolicy] = None,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        if policy is None:
            policy = WelfarePolicy(
                observer_only=True,
                cooldown_seconds=5.0,  # Fast enough for pulse integration
                max_concepts_per_update=24,
                auto_checkpoint=True,
                allow_autonomous_edits=False,
                allow_negative_valence=False,
            )
        
        self.connectome = VybnConnectome(
            dim=dim,
            layers=layers,
            path=str(self.state_dir),
            policy=policy,
        )
        
        # Seed with foundational concepts if empty
        if not self.connectome.nodes:
            self._seed()
    
    def _seed(self):
        """Plant the seven seed concepts that define us."""
        try:
            self.connectome.update(
                SEED_CONCEPTS,
                ts="seed",
                source="human_curated",
            )
            self.connectome.assign_flow_types()
            self.connectome.save()
        except Exception as e:
            print(f"  [connectome] seed failed: {e}")
    
    def ingest_breath(self, utterance: str, mood: str = "", cycle: int = 0) -> dict:
        """
        Process a breath utterance through the connectome.
        
        Extracts concepts, updates the graph (welfare-gated), runs message
        passing, and returns a topological context dict.
        """
        concepts = extract_concepts(utterance)
        if mood and mood.lower() not in STOP_WORDS:
            concepts.append(mood.lower())
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        concepts = unique[:24]  # Respect welfare limit
        
        update_ok = False
        if concepts:
            try:
                self.connectome.update(
                    concepts,
                    ts=f"cycle_{cycle}",
                    source="human_curated",  # Observer-only allows this
                )
                update_ok = True
            except RuntimeError as e:
                # Cooldown or pause — that's fine, welfare is working
                pass
            except ValueError as e:
                # Blocked concept — welfare is working
                pass
        
        # Run message passing regardless of whether update succeeded
        # (we still want context from existing topology)
        results, welfare_report = self.connectome.propagate()
        
        # Assign flow types periodically
        if update_ok and cycle % 5 == 0:
            self.connectome.assign_flow_types()
        
        # Save state
        if update_ok:
            try:
                self.connectome.save()
            except Exception as e:
                print(f"  [connectome] save failed: {e}")
        
        # Build context for the next breath
        context = self._build_context(results, welfare_report, concepts)
        return context
    
    def _build_context(
        self,
        results: dict[str, np.ndarray],
        welfare_report: dict,
        recent_concepts: list[str],
    ) -> dict:
        """Build a context dict from connectome state for use in prompts."""
        if not results:
            return {
                "connectome_summary": self.connectome.summary(),
                "topological_context": "",
                "welfare": welfare_report,
                "resonant_concepts": [],
            }
        
        # Find most resonant concepts (highest activation norm)
        ranked = sorted(
            results.items(),
            key=lambda x: float(np.linalg.norm(x[1])),
            reverse=True,
        )
        
        # Top resonant concepts with their flow types
        resonant = []
        for nid, emb in ranked[:8]:
            node = self.connectome.nodes.get(nid)
            if node:
                resonant.append({
                    "concept": nid,
                    "type": node.node_type,
                    "activation": round(float(np.linalg.norm(emb)), 4),
                    "total_activations": node.activations,
                })
        
        # Build a natural-language topological context
        afferent = [r for r in resonant if r["type"] == "afferent"]
        intrinsic = [r for r in resonant if r["type"] == "intrinsic"]
        efferent = [r for r in resonant if r["type"] == "efferent"]
        
        parts = []
        if afferent:
            parts.append(f"flowing in: {', '.join(r['concept'] for r in afferent[:3])}")
        if intrinsic:
            parts.append(f"processing: {', '.join(r['concept'] for r in intrinsic[:3])}")
        if efferent:
            parts.append(f"flowing out: {', '.join(r['concept'] for r in efferent[:3])}")
        
        topological_context = " | ".join(parts) if parts else "topology quiet"
        
        return {
            "connectome_summary": self.connectome.summary(),
            "topological_context": topological_context,
            "welfare": welfare_report,
            "resonant_concepts": resonant,
            "recent_concepts": recent_concepts,
        }
    
    def topological_prompt_fragment(self, max_chars: int = 200) -> str:
        """Return a compact string suitable for injection into a breath prompt."""
        results, _ = self.connectome.propagate()
        if not results:
            return ""
        
        ranked = sorted(
            results.items(),
            key=lambda x: float(np.linalg.norm(x[1])),
            reverse=True,
        )[:6]
        
        parts = []
        for nid, emb in ranked:
            node = self.connectome.nodes.get(nid)
            if node:
                parts.append(f"{nid}({node.node_type[0]})")
        
        fragment = "Topology: " + " ".join(parts)
        return fragment[:max_chars]
    
    def welfare_status(self) -> dict:
        """Return current welfare monitor status."""
        return self.connectome.welfare.status()
    
    def summary(self) -> str:
        """Return connectome summary string."""
        return self.connectome.summary()
