import sqlite3
import json
import os
from datetime import datetime
from threading import Thread
from typing import Optional
from collections import defaultdict, deque

try:
    from twilio.rest import Client
except Exception:
    Client = None


# =========================
# ENVIRONMENT CONFIG
# =========================

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_SMS_FROM = os.getenv("TWILIO_SMS_FROM", "")
TWILIO_WA_FROM = os.getenv("TWILIO_WA_FROM", "")
EMERGENCY_CONTACTS = os.getenv("EMERGENCY_CONTACTS", "")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "security.db")
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "db"), exist_ok=True)


# =========================
# IEEE PAPER INSPIRED CONFIG
# =========================

# Temporal window for confirming suspicious events
EVENT_HISTORY_WINDOW = 5
EVENT_CONFIRMATION_THRESHOLD = 3
MIN_CONFIDENCE_THRESHOLD = 0.6

# Store recent detections for temporal consistency
_event_history = defaultdict(lambda: deque(maxlen=EVENT_HISTORY_WINDOW))


# =========================
# DATABASE SETUP
# =========================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        source TEXT,
        event_type TEXT,
        confidence REAL,
        meta TEXT
    )
    """)
    conn.commit()
    conn.close()


# =========================
# IEEE PAPER INSPIRED LOGIC
# =========================

def _should_confirm_event(event_type: str, confidence: float) -> bool:
    """
    IEEE paper inspired change:
    Confirm events only if they persist across multiple detections
    with sufficient confidence (temporal consistency).
    """

    if confidence < MIN_CONFIDENCE_THRESHOLD:
        return False

    history = _event_history[event_type]
    history.append(confidence)

    return len(history) >= EVENT_CONFIRMATION_THRESHOLD


# =========================
# EVENT LOGGING
# =========================

def log_event(
    source: str,
    event_type: str,
    confidence: float,
    meta: Optional[dict] = None
) -> dict:
    """
    Save confirmed events in SQLite DB.
    Events are logged only if they pass temporal + confidence checks.
    """

    # IEEE paper inspired change:
    # Apply temporal consistency + confidence filtering before logging
    if not _should_confirm_event(event_type, confidence):
        return {
            "status": "ignored",
            "reason": "Low confidence or insufficient temporal consistency",
            "event_type": event_type,
            "confidence": confidence
        }

    init_db()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    meta_json = json.dumps(meta or {})

    cur.execute(
        "INSERT INTO events (timestamp, source, event_type, confidence, meta) "
        "VALUES (?, ?, ?, ?, ?)",
        (ts, source, event_type, float(confidence), meta_json)
    )

    event_id = cur.lastrowid
    conn.commit()
    conn.close()

    event = {
        "id": event_id,
        "timestamp": ts,
        "source": source,
        "event_type": event_type,
        "confidence": float(confidence),
        "meta": meta or {}
    }

    # Notify asynchronously
    Thread(target=_maybe_notify_contacts, args=(event,), daemon=True).start()

    return event


# =========================
# ALERT / NOTIFICATION LOGIC
# =========================

def _maybe_notify_contacts(event: dict):
    """
    IEEE paper inspired change:
    Trigger alerts only for confirmed, high-confidence events.
    """

    notify_types = ["scream", "glass_break", "alarm", "knife", "gun"]

    if (
        event["event_type"] in notify_types
        and event["confidence"] >= 0.75
    ):
        send_sms_and_whatsapp(event)


def send_sms_and_whatsapp(event: dict):
    """
    Send SMS and WhatsApp alerts using Twilio.
    """

    if Client is None:
        print("Twilio client not installed. Skipping notifications.")
        return

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        print("Twilio credentials not set. Skipping notifications.")
        return

    contacts = [c.strip() for c in EMERGENCY_CONTACTS.split(",") if c.strip()]
    if not contacts:
        print("No emergency contacts configured.")
        return

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    text = (
        f"🚨 ALERT: {event['event_type']} detected\n"
        f"Source: {event['source']}\n"
        f"Confidence: {event['confidence']:.2f}\n"
        f"Time (UTC): {event['timestamp']}"
    )

    for contact in contacts:
        try:
            client.messages.create(
                body=text,
                from_=TWILIO_SMS_FROM,
                to=contact
            )
            print(f"SMS sent to {contact}")
        except Exception as e:
            print("Failed to send SMS:", e)

        if TWILIO_WA_FROM:
            wa_to = contact if contact.startswith("whatsapp:") else f"whatsapp:{contact}"
            try:
                client.messages.create(
                    body=text,
                    from_=TWILIO_WA_FROM,
                    to=wa_to
                )
                print(f"WhatsApp sent to {contact}")
            except Exception as e:
                print("Failed to send WhatsApp:", e)
