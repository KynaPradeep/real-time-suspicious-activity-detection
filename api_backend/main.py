from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db.db_manager import save_event, get_all_events, init_db
from notifications import send_sms_alert
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

@app.get("/")
def home():
    return {"message": "Backend running"}

@app.post("/audio_alert")
def audio_alert(data: dict):
    label = data["label"]
    confidence = float(data["confidence"])
    timestamp = data.get("timestamp", time.time())

    sms_sent = 0

    if label != "normal" and confidence >= 0.85:
        send_sms_alert("AUDIO", label, confidence)
        sms_sent = 1

    save_event(
        event_type="audio",
        label=label,
        confidence=confidence,
        timestamp=timestamp,
        sms_sent=sms_sent
    )

    return {"status": "ok"}

@app.post("/video_alert")
def video_alert(data: dict):
    label = data["label"]
    confidence = float(data["confidence"])
    timestamp = data.get("timestamp", time.time())

    sms_sent = 0
    dangerous_objects = ["knife", "gun", "scissors"]

    if label in dangerous_objects and confidence >= 0.85:
        send_sms_alert("VIDEO", label, confidence)
        sms_sent = 1

    save_event(
        event_type="video",
        label=label,
        confidence=confidence,
        timestamp=timestamp,
        sms_sent=sms_sent
    )

    return {"status": "ok"}

@app.get("/events")
def list_events():
    return get_all_events()
