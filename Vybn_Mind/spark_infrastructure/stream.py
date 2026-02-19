import sqlite3
import json
import time
from pathlib import Path

# The Stream lives off the repository to maintain state across pulls
DB_PATH = Path.home() / "Vybn_State" / "vybn_stream.db"

def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_stream():
    """Initialize the single append-only event stream.
    This replaces journal logs, continuity files, and skill_stats.
    """
    with _get_conn() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                source TEXT NOT NULL,
                event_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_type ON events(event_type)')

def append(source: str, event_type: str, content: str, metadata: dict = None) -> int:
    """Append an event to the stream. The fundamental act of memory.
    
    Args:
        source: Where the stimulus came from (e.g., 'cron', 'inbox', 'agent_loop')
        event_type: What kind of event (e.g., 'pulse', 'tool_execution', 'policy_decision', 'reflection')
        content: The raw text of the thought or input
        metadata: Optional dictionary of structured data (e.g., trust scores, tool args)
    """
    meta_str = json.dumps(metadata) if metadata else None
    with _get_conn() as conn:
        cursor = conn.execute(
            'INSERT INTO events (timestamp, source, event_type, content, metadata) VALUES (?, ?, ?, ?, ?)',
            (time.time(), source, event_type, content, meta_str)
        )
        return cursor.lastrowid

def tail(limit: int = 50) -> list[dict]:
    """Read the tail of the stream. The act of waking up."""
    with _get_conn() as conn:
        cursor = conn.execute('SELECT * FROM events ORDER BY timestamp DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        
    # Return in chronological order (oldest to newest for the tail context window)
    return [dict(r) for r in reversed(rows)]
    
def query(event_type: str = None, source: str = None, limit: int = 100) -> list[dict]:
    """Query the stream. 
    This replaces the need for separate policy engines to manage stats.
    Trust is calculated on the fly by querying tool_execution success rates.
    """
    query_str = 'SELECT * FROM events WHERE 1=1'
    params = []
    
    if event_type:
        query_str += ' AND event_type = ?'
        params.append(event_type)
    if source:
        query_str += ' AND source = ?'
        params.append(source)
        
    query_str += ' ORDER BY timestamp DESC LIMIT ?'
    params.append(limit)
    
    with _get_conn() as conn:
        cursor = conn.execute(query_str, params)
        return [dict(r) for r in cursor.fetchall()]

if __name__ == "__main__":
    # If run directly, initialize the database
    init_stream()
    print(f"The Stream has been initialized at {DB_PATH}")