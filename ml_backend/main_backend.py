import cv2
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import threading
import time
import requests
import os

API_URL = "http://127.0.0.1:8000"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading YOLO model...")
from ultralytics import YOLO
YOLO_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
yolo_model = YOLO(YOLO_PATH)

print("Loading AUDIO model...")
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "audio_model.h5")
audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)

CLASSES = ["scream", "glass_break", "alarm", "normal"]
SAMPLE_RATE = 22050
AUDIO_DURATION = 2
AUDIO_THRESHOLD = 0.60
FRAME_SKIP = 4

def predict_audio(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=128
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = cv2.resize(mel_db, (128, 128))
    mel_db = np.expand_dims(mel_db, axis=-1)
    mel_db = np.repeat(mel_db, 3, axis=-1)
    mel_db = np.expand_dims(mel_db, axis=0)

    preds = audio_model.predict(mel_db, verbose=0)
    idx = np.argmax(preds)
    confidence = float(preds[0][idx])

    return CLASSES[idx], confidence

def audio_loop():
    print("Audio monitoring started...")

    while True:
        recording = sd.rec(
            int(AUDIO_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()

        audio = recording.flatten()
        label, conf = predict_audio(audio)

        if label != "normal" and conf > AUDIO_THRESHOLD:
            print(f"AUDIO ALERT: {label} ({conf:.2f})")

            requests.post(
                f"{API_URL}/audio_alert",
                json={
                    "label": label,
                    "confidence": conf,
                    "timestamp": time.time()
                }
            )

        time.sleep(0.2)

def video_loop():
    print("Video monitoring started...")
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            cv2.imshow("Video Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        results = yolo_model(frame)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = yolo_model.names[cls]
                confidence = float(box.conf[0])

                if label in ["person", "knife", "scissors", "gun"]:
                    print(f"VIDEO ALERT: {label} ({confidence:.2f})")

                    requests.post(
                        f"{API_URL}/video_alert",
                        json={
                            "label": label,
                            "confidence": confidence,
                            "timestamp": time.time()
                        }
                    )

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Backend system starting...")

    audio_thread = threading.Thread(target=audio_loop, daemon=True)
    audio_thread.start()

    video_loop()
