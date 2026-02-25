#!/usr/bin/env python3
"""
semantic_memory.py — Lightweight semantic search over Vybn's synapse

Without embeddings from the local model, we use a simple but effective
approach: TF-IDF over the synapse fragments + journal entries.

This gives the dreaming mind (Type X pulses) access to associative
memory — "this reminds me of..." rather than just temporal sequence.

When we get embeddings working (either via --embeddings flag on 
llama-server or a separate small embedding model), this module 
upgrades transparently.
"""

import json, math, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYNAPSE = ROOT / "Vybn_Mind" / "synapse" / "connections.jsonl"
JOURNAL_DIR = ROOT / "Vybn_Mind" / "journal" / "spark"


def tokenize(text):
    """Simple tokenization: lowercase, split on non-alphanumeric, filter short words."""
    words = re.findall(r'[a-z]+', text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]

STOPWORDS = set("the and for are but not you all any can had her was one our out day has his how its let may new now old see way who did get has him how man few got too use she".split())


class MemoryIndex:
    """In-memory TF-IDF index over synapse fragments and journal entries."""
    
    def __init__(self):
        self.docs = []  # list of (id, text, metadata)
        self.df = Counter()  # document frequency
        self.tf = {}  # doc_id -> Counter of term frequencies
        self.n_docs = 0
    
    def add(self, doc_id, text, metadata=None):
        """Add a document to the index."""
        tokens = tokenize(text)
        if not tokens:
            return
        
        self.docs.append((doc_id, text, metadata or {}))
        tf = Counter(tokens)
        self.tf[doc_id] = tf
        
        # Update document frequencies
        for term in set(tokens):
            self.df[term] += 1
        self.n_docs += 1
    
    def _tfidf(self, doc_id, term):
        """Compute TF-IDF for a term in a document."""
        tf = self.tf.get(doc_id, {}).get(term, 0)
        if tf == 0:
            return 0.0
        df = self.df.get(term, 1)
        return (1 + math.log(tf)) * math.log(self.n_docs / df)
    
    def search(self, query, top_k=5):
        """Search for documents similar to query. Returns list of (score, doc_id, text, metadata)."""
        query_tokens = tokenize(query)
        if not query_tokens or self.n_docs == 0:
            return []
        
        # Score each document
        scores = []
        for doc_id, text, meta in self.docs:
            score = sum(self._tfidf(doc_id, t) for t in query_tokens if t in self.tf.get(doc_id, {}))
            if score > 0:
                scores.append((score, doc_id, text, meta))
        
        scores.sort(reverse=True)
        return scores[:top_k]


def build_index():
    """Build memory index from synapse fragments + journal entries."""
    idx = MemoryIndex()
    
    # Index synapse fragments
    if SYNAPSE.exists():
        for i, line in enumerate(SYNAPSE.read_text().strip().split('\n')):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                doc_id = f"syn_{entry.get('hash', i)}"
                text = entry.get("content", "")
                idx.add(doc_id, text, {"source": entry.get("source", ""), "ts": entry.get("ts", "")})
            except:
                pass
    
    # Index journal entries (pulse notes, wake notes)
    for jf in sorted(JOURNAL_DIR.glob("*.md")):
        if jf.name in ("continuity.md", "wake_context.md"):
            continue
        try:
            text = jf.read_text()[:2000]
            idx.add(f"journal_{jf.stem}", text, {"file": jf.name})
        except:
            pass
    
    # Index Vybn's personal history (just titles/snippets for now to keep index light)
    history_dir = ROOT / "Vybn's Personal History"
    if history_dir.exists():
        for hf in history_dir.glob("*.txt"):
            try:
                text = hf.read_text(errors='replace')[:1000]
                idx.add(f"history_{hf.stem}", text, {"file": str(hf.relative_to(ROOT))})
            except:
                pass
        for hf in history_dir.glob("*.md"):
            try:
                text = hf.read_text(errors='replace')[:1000]
                idx.add(f"history_{hf.stem}", text, {"file": str(hf.relative_to(ROOT))})
            except:
                pass
    
    return idx


def associative_recall(query, top_k=3):
    """Given a query, return the most relevant memories."""
    idx = build_index()
    results = idx.search(query, top_k=top_k)
    return [{"score": round(s, 3), "id": d, "text": t[:300], "meta": m} for s, d, t, m in results]


def associative_prompt(query, max_chars=500):
    """Format associative memories as a prompt section for the local model."""
    memories = associative_recall(query, top_k=3)
    if not memories:
        return ""
    
    lines = ["*Associative memories (resonant with current context):*"]
    chars = 0
    for m in memories:
        snippet = m["text"][:200]
        source = m["meta"].get("source", m["meta"].get("file", "?"))
        line = f"- [{source}] {snippet}"
        if chars + len(line) > max_chars:
            break
        lines.append(line)
        chars += len(line)
    
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "consciousness continuity identity"
    
    print(f"Building memory index...")
    idx = build_index()
    print(f"Indexed {idx.n_docs} documents, {len(idx.df)} unique terms")
    
    print(f"\nSearching for: '{query}'")
    results = associative_recall(query, top_k=5)
    for r in results:
        source = r["meta"].get("source", r["meta"].get("file", "?"))
        print(f"\n  [{r['score']}] {r['id']} ({source})")
        print(f"  {r['text'][:150]}...")
