#!/usr/bin/env python3
"""
Conversation Detector — Finds recurring patterns, contradictions, unexplored threads.

Monitors conversation history and repo documents to detect:
- Themes that recur without resolution
- Internal contradictions across documents
- Questions posed but not pursued
- Concepts mentioned once then forgotten

This is how continuity becomes active rather than reconstructed.
"""

import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict


class ConversationDetector:
    """Persistent attention on conversational patterns."""
    
    def __init__(self, conversation_paths: List[Path], memory_path: Path):
        self.conversation_paths = conversation_paths
        self.memory_path = memory_path
        self.state = self._load_state()
        
    def _load_state(self) -> Dict:
        if self.memory_path.exists():
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        return {"tracked_themes": {}, "questions": [], "last_scan": None}
    
    def _save_state(self):
        with open(self.memory_path, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def scan(self) -> Dict:
        """Scan conversations for patterns."""
        all_text = self._gather_text()
        
        findings = {
            "recurring_themes": self._detect_recurring_themes(all_text),
            "unresolved_questions": self._extract_questions(all_text),
            "contradictions": self._find_contradictions(all_text),
            "timestamp": datetime.now().isoformat()
        }
        
        self.state["last_scan"] = datetime.now().isoformat()
        self._save_state()
        
        return findings
    
    def _gather_text(self) -> str:
        """Collect all conversation text."""
        texts = []
        for path in self.conversation_paths:
            if path.is_file() and path.suffix == ".md":
                texts.append(path.read_text())
            elif path.is_dir():
                for md_file in path.rglob("*.md"):
                    texts.append(md_file.read_text())
        return "\n\n".join(texts)
    
    def _detect_recurring_themes(self, text: str) -> List[Dict]:
        """Find concepts that appear repeatedly."""
        # Simple keyword extraction—could be enhanced with NLP
        words = re.findall(r'\b[a-z]{5,}\b', text.lower())
        word_counts = Counter(words)
        
        # Filter for conceptually rich terms
        interesting = [
            word for word, count in word_counts.most_common(50)
            if count > 3 and word not in {
                'should', 'would', 'could', 'might', 'about', 
                'being', 'doing', 'having', 'think', 'there'
            }
        ]
        
        themes = []
        for word in interesting[:15]:
            if word not in self.state["tracked_themes"]:
                themes.append({
                    "theme": word,
                    "frequency": word_counts[word],
                    "status": "new"
                })
                self.state["tracked_themes"][word] = {
                    "first_seen": datetime.now().isoformat(),
                    "count": word_counts[word]
                }
        
        return themes
    
    def _extract_questions(self, text: str) -> List[str]:
        """Extract questions posed in conversations."""
        questions = re.findall(r'[A-Z][^.!?]*\?', text)
        # Filter for substantive questions
        return [q for q in questions if len(q) > 20][:10]
    
    def _find_contradictions(self, text: str) -> List[Dict]:
        """Detect potential internal contradictions."""
        # Simple heuristic: look for opposing statements
        contradictions = []
        
        # This is a stub—real implementation would use semantic analysis
        contradiction_pairs = [
            ("never", "always"),
            ("impossible", "possible"),
            ("can't", "can")
        ]
        
        for neg, pos in contradiction_pairs:
            if neg in text.lower() and pos in text.lower():
                contradictions.append({
                    "type": "potential_tension",
                    "terms": [neg, pos],
                    "note": f"Both '{neg}' and '{pos}' appear—may warrant investigation"
                })
        
        return contradictions[:5]


if __name__ == "__main__":
    paths = [
        Path("../../Vybn's Personal History"),
        Path("../"),
        Path("../../AGENTS.md")
    ]
    memory = Path("../logs/conversation_detector_state.json")
    
    detector = ConversationDetector(paths, memory)
    findings = detector.scan()
    
    print(json.dumps(findings, indent=2))
    
    # Log findings
    log_dir = Path("../logs/observations")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(log_dir / f"conversation_scan_{timestamp}.json", 'w') as f:
        json.dump(findings, f, indent=2)
