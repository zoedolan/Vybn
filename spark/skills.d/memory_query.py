"""Structured memory query - SQLite-based journal search.

SKILL_NAME: memory_query
TOOL_ALIASES: ["memory_query", "query_memory", "search_journal"]
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

SKILL_NAME = "memory_query"
TOOL_ALIASES = ["memory_query", "query_memory", "search_journal"]


def _ensure_db(journal_dir: Path) -> Path:
    """Create SQLite database if it doesn't exist."""
    db_path = journal_dir / "memory.db"
    
    if db_path.exists():
        return db_path
    
    # Create database and import existing markdown journals
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            timestamp TEXT,
            title TEXT,
            content TEXT,
            tags TEXT
        )
    """)
    
    # Import existing markdown files
    for md_file in journal_dir.glob("*.md"):
        if md_file.name in ["continuity.md", "bookmarks.md", "memory.db"]:
            continue
        
        try:
            content = md_file.read_text(encoding="utf-8")
            lines = content.split("\n", 2)
            title = lines[0].lstrip("# ").strip() if lines else "untitled"
            
            cursor.execute(
                "INSERT OR IGNORE INTO entries (filename, timestamp, title, content) VALUES (?, ?, ?, ?)",
                (md_file.name, md_file.stat().st_mtime, title, content)
            )
        except Exception:
            pass
    
    conn.commit()
    conn.close()
    
    return db_path


def execute(action: dict, router) -> str:
    """Query structured memory using SQL-like patterns."""
    params = action.get("params", {})
    query = (
        action.get("argument", "")
        or params.get("query", "")
        or params.get("search", "")
    )
    
    if not query:
        return "no search query specified"
    
    db_path = _ensure_db(router.journal_dir)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Full-text search
        cursor.execute(
            """
            SELECT filename, title, content, timestamp
            FROM entries
            WHERE content LIKE ? OR title LIKE ?
            ORDER BY timestamp DESC
            LIMIT 10
            """,
            (f"%{query}%", f"%{query}%")
        )
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return f"no memory entries matching '{query}'"
        
        output = f"found {len(results)} entries matching '{query}':\n\n"
        for filename, title, content, ts in results:
            snippet = content[:300].replace("\n", " ")
            output += f"[{filename}] {title}\n{snippet}...\n\n"
        
        return output
    
    except Exception as e:
        return f"memory query error: {e}"
