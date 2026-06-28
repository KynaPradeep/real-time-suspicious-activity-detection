
import os
import sqlite3
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path

from .event_manager import init_db

app = FastAPI(title="Suspicious Activity Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "security.db")

init_db()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active_connections.remove(ws)

    async def broadcast(self, message: dict):
        text = json.dumps(message)
        for conn in list(self.active_connections):
            try:
                await conn.send_text(text)
            except Exception:
                self.disconnect(conn)

manager = ConnectionManager()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/alerts/latest")
def get_latest(n: int = 20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, source, event_type, confidence, meta FROM events ORDER BY id DESC LIMIT ?", (n,))
    rows = cur.fetchall()
    conn.close()
    events = []
    for r in rows:
        events.append({
            "id": r[0],
            "timestamp": r[1],
            "source": r[2],
            "event_type": r[3],
            "confidence": r[4],
            "meta": json.loads(r[5] or "{}")
        })
    return events

@app.get("/alerts/all")
def get_all(limit: int = 1000):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, source, event_type, confidence, meta FROM events ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    events = []
    for r in rows:
        events.append({
            "id": r[0],
            "timestamp": r[1],
            "source": r[2],
            "event_type": r[3],
            "confidence": r[4],
            "meta": json.loads(r[5] or "{}")
        })
    return events

@app.websocket("/ws/alerts")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
