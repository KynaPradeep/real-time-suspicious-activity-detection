import sqlite3
from pathlib import Path
import time

DB_PATH = Path(__file__).parent / "alerts.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT,
    label TEXT,
    confidence REAL,
    timestamp REAL
)
""")

    conn.commit()
    conn.close()

def save_event(event_type, label, confidence, timestamp=None):
    if timestamp is None:
        timestamp = time.time()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO events (event_type, label, confidence, timestamp)
    VALUES (?, ?, ?, ?)
    """, (event_type, label, confidence, timestamp))

    conn.commit()
    conn.close()

def get_all_events():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, event_type, label, confidence, timestamp
    FROM events
    ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "type": r[1],
            "label": r[2],
            "confidence": r[3],
            "timestamp": r[4]
        }
        for r in rows
    ]
