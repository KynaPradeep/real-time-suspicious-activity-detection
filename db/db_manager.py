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
        timestamp REAL,
        sms_sent INTEGER DEFAULT 0
    )
    """)

    conn.commit()
    conn.close()

def save_event(event_type, label, confidence, timestamp=None, sms_sent=0):
    if timestamp is None:
        timestamp = time.time()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO events (event_type, label, confidence, timestamp, sms_sent)
    VALUES (?, ?, ?, ?, ?)
    """, (event_type, label, confidence, timestamp, sms_sent))

    conn.commit()
    conn.close()

def get_all_events():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, event_type, label, confidence, timestamp, sms_sent
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
            "timestamp": r[4],
            "sms_sent": r[5]
        }
        for r in rows
    ]
