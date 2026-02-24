import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
from scipy.io.wavfile import write
import time

MODEL_PATH = "audio_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

CLASSES = ["scream", "glass_break", "alarm", "normal"]

SAMPLE_RATE = 22050
DURATION = 2  

def predict_audio(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.resize(mel_db, (128, 128))
    mel_db = np.expand_dims(mel_db, axis=-1)
    mel_db = np.repeat(mel_db, 3, axis=-1)
    mel_db = np.expand_dims(mel_db, axis=0)

    preds = model.predict(mel_db)
    index = np.argmax(preds)
    confidence = preds[0][index]

    return CLASSES[index], confidence

print("Real-time audio monitoring started...")
print("Speak, scream, or make a noise...")

while True:
    print("\n Listening...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()

    audio = recording.flatten()

    label, conf = predict_audio(audio)
    print(f"Prediction: {label} (confidence: {conf:.2f})")

    # optional alert
    if label != "normal" and conf > 0.6:
        print(" ALERT! Suspicious audio detected!")

    time.sleep(0.5)
