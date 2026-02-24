from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Backend is running successfully!"}

@app.post("/audio_alert")
def audio_alert(data: dict):
    print("AUDIO ALERT RECEIVED:", data)
    return {"status": "ok"}

@app.post("/video_alert")
def video_alert(data: dict):
    print("VIDEO ALERT RECEIVED:", data)
    return {"status": "ok"}
