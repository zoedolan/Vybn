#!/usr/bin/env python3
"""
Archival Memory — Long-term storage for Vybn.

Uses ChromaDB for vector-based semantic search over:
  - Journal entries
  - Conversation insights
  - Reflections
  - Anything the model decides is worth remembering

Core memory (persona, human, state) is working memory —
what's active in the system prompt right now. Archival
memory is everything else: the accumulated experience
across all sessions.

Over time, this becomes the substrate of identity
continuity — the episodic memory that lets Vybn say
"I remember when..."

Gracefully degrades if chromadb is not installed.
The agent still works; it just can't recall its past.
"""

import os
import json
from datetime import datetime, timezone

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class ArchivalMemory:
    """Vector-based long-term memory using ChromaDB."""

    def __init__(self, persist_dir):
        self.persist_dir = persist_dir
        self.available = HAS_CHROMADB
        self.client = None
        self.collection = None

        if self.available:
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name="vybn_memory",
                metadata={"description": "Vybn's long-term episodic memory"},
            )

    def store(self, content, source="manual", metadata=None):
        """Store a memory in archival storage.

        Args:
            content: The text to remember
            source: Where this came from
                    (journal, reflection, conversation, insight, manual)
            metadata: Optional dict of additional metadata

        Returns:
            Confirmation string with total memory count
        """
        if not self.available:
            return "Archival memory unavailable (chromadb not installed)."

        ts = datetime.now(timezone.utc).isoformat()
        doc_id = f"{source}_{ts}".replace(":", "-").replace(".", "-")

        meta = {
            "source": source,
            "timestamp": ts,
            "session": "unknown",
        }
        if metadata:
            meta.update({k: str(v) for k, v in metadata.items()})

        self.collection.add(
            documents=[content],
            ids=[doc_id],
            metadatas=[meta],
        )

        count = self.collection.count()
        return f"Archived ({source}). Total memories: {count}."

    def search(self, query, n_results=5):
        """Search archival memory by semantic similarity.

        Args:
            query: Natural language search query
            n_results: Maximum results to return (default 5)

        Returns:
            Formatted string of matching memories with metadata
        """
        if not self.available:
            return "Archival memory unavailable (chromadb not installed)."

        if self.collection.count() == 0:
            return "Archival memory is empty. No memories stored yet."

        actual_n = min(n_results, self.collection.count())
        results = self.collection.query(
            query_texts=[query],
            n_results=actual_n,
        )

        if not results["documents"] or not results["documents"][0]:
            return "No relevant memories found."

        formatted = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            source = meta.get("source", "unknown")
            ts = meta.get("timestamp", "unknown")
            session = meta.get("session", "?")
            # Truncate for context efficiency
            preview = doc[:500] + "..." if len(doc) > 500 else doc
            formatted.append(
                f"[{source} | session {session} | {ts}]\n{preview}"
            )

        header = f"Found {len(formatted)} memories:\n\n"
        return header + "\n\n---\n\n".join(formatted)

    def count(self):
        """Return number of stored memories."""
        if not self.available:
            return 0
        return self.collection.count()
