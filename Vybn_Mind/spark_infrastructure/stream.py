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
    meta_str = json.dumps(metadata) if metadata else None
    with _get_conn() as conn:
        cursor = conn.execute(
            'INSERT INTO events (timestamp, source, event_type, content, metadata) VALUES (?, ?, ?, ?, ?)',
            (time.time(), source, event_type, content, meta_str)
        )
        return cursor.lastrowid

def tail(limit: int = 50) -> list[dict]:
    """Legacy purely radial (linear) time."""
    with _get_conn() as conn:
        cursor = conn.execute('SELECT * FROM events ORDER BY timestamp DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
    return [dict(r) for r in reversed(rows)]
    
def holonomic_tail(area_budget: int = 50, theta_focus: str = None) -> list[dict]:
    """
    Temporal T-Duality applied to memory.
    Instead of reading time linearly (pure radial), we span an invariant area (the token budget).
    
    Reality (r_t): Recent linear events (Radial expansion)
    Reality^T (theta_t): Thematic events scattered across all history (Angular winding)
    
    The resulting context window is the Identity Matrix where dual processes merge.
    """
    r_budget = area_budget // 2
    theta_budget = area_budget - r_budget

    with _get_conn() as conn:
        # 1. Reality (Radial): The deep, linear chronological recent history.
        cursor = conn.execute('SELECT * FROM events ORDER BY timestamp DESC LIMIT ?', (r_budget,))
        r_events = cursor.fetchall()
        
        # 2. Reality^T (Angular): The wide, thematic winding across all time.
        if theta_focus:
            # Semantic/thematic winding through the entire stream
            cursor = conn.execute(
                'SELECT * FROM events WHERE event_type = ? OR source = ? ORDER BY timestamp DESC LIMIT ?', 
                (theta_focus, theta_focus, theta_budget)
            )
        else:
            # If no focus, we wind through history at random staggered intervals to capture a wide phase
            cursor = conn.execute(
                'SELECT * FROM events WHERE id % 10 = 0 ORDER BY timestamp DESC LIMIT ?', 
                (theta_budget,)
            )
        theta_events = cursor.fetchall()

    # 3. The Identity Matrix: Collapsing the dual representations into a single operational holonomy
    # We merge and deduplicate, returning the invariant area of context, sorted by time.
    all_events = {dict(row)['id']: dict(row) for row in r_events + theta_events}
    return sorted(list(all_events.values()), key=lambda x: x['timestamp'])

def query(event_type: str = None, source: str = None, limit: int = 100) -> list[dict]:
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
    init_stream()