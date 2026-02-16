"""Semantic search skill - embedding-based search over memories and codebase.

SKILL_NAME: semantic_search
TOOL_ALIASES: ["semantic_search", "embed_search", "vector_search"]

NOTE: Requires sentence-transformers library. Install with:
  pip install sentence-transformers
"""

import json
from pathlib import Path
import pickle

SKILL_NAME = "semantic_search"
TOOL_ALIASES = ["semantic_search", "embed_search", "vector_search"]

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


def execute(action: dict, router) -> str:
    """Semantic search over journal entries and codebase."""
    if not EMBEDDINGS_AVAILABLE:
        return (
            "semantic search requires sentence-transformers library.\n"
            "Install with: pip install sentence-transformers"
        )
    
    params = action.get("params", {})
    query = (
        action.get("argument", "")
        or params.get("query", "")
        or params.get("search", "")
    )
    
    if not query:
        return "no search query specified"
    
    scope = params.get("scope", "journal")  # journal, code, or all
    limit = int(params.get("limit", 5))
    
    # Load or create embedding index
    index_file = router.journal_dir / "embeddings.pkl"
    
    if index_file.exists():
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
    else:
        # Build index
        index_data = _build_index(router, scope)
        with open(index_file, 'wb') as f:
            pickle.dump(index_data, f)
    
    # Load model and compute query embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    
    # Compute similarities
    embeddings = np.array(index_data['embeddings'])
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top results
    top_indices = np.argsort(similarities)[-limit:][::-1]
    
    results = []
    for idx in top_indices:
        doc = index_data['documents'][idx]
        score = similarities[idx]
        results.append((doc, score))
    
    if not results:
        return f"no results found for '{query}'"
    
    output = f"semantic search results for '{query}' (scope: {scope}):\n\n"
    for doc, score in results:
        output += f"[{score:.3f}] {doc['name']}\n"
        output += f"  {doc['snippet'][:200]}...\n\n"
    
    return output


def _build_index(router, scope: str) -> dict:
    """Build semantic search index."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    documents = []
    texts = []
    
    # Index journal entries
    if scope in ["journal", "all"]:
        for md_file in router.journal_dir.glob("*.md"):
            if md_file.name in ["continuity.md", "bookmarks.md", "reminders.json", "watches.json"]:
                continue
            
            try:
                content = md_file.read_text(encoding="utf-8")
                documents.append({
                    "type": "journal",
                    "name": md_file.name,
                    "snippet": content[:500].replace("\n", " "),
                })
                texts.append(content[:2000])
            except Exception:
                pass
    
    # Index code files
    if scope in ["code", "all"]:
        for py_file in router.repo_root.rglob("*.py"):
            if ".git" in str(py_file) or "__pycache__" in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding="utf-8")
                documents.append({
                    "type": "code",
                    "name": str(py_file.relative_to(router.repo_root)),
                    "snippet": content[:500].replace("\n", " "),
                })
                texts.append(content[:2000])
            except Exception:
                pass
    
    # Compute embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return {
        "documents": documents,
        "embeddings": embeddings.tolist(),
        "model": "all-MiniLM-L6-v2",
    }
