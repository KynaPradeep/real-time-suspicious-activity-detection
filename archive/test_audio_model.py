import numpy as np
import librosa
import tensorflow as tf
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "audio_model.h5")

CLASSES = ["scream", "glass_break", "alarm", "normal"]

model = tf.keras.models.load_model(MODEL_PATH)


def predict_audio(file_path):

    y, sr = librosa.load(file_path, sr=22050)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_db = cv2.resize(
        mel_db,
        (128, 128)
    )

    mel_db = np.expand_dims(
        mel_db,
        axis=-1
    )

    mel_db = np.repeat(
        mel_db,
        3,
        axis=-1
    )

    mel_db = np.expand_dims(
        mel_db,
        axis=0
    )

    pred = model.predict(
        mel_db,
        verbose=0
    )

    class_id = np.argmax(pred)

    print("\nRaw predictions:")
    for i, c in enumerate(CLASSES):
        print(f"{c}: {pred[0][i]:.4f}")

    print(
        f"\nPrediction: {CLASSES[class_id]} "
        f"(confidence: {pred[0][class_id]:.4f})"
    )


predict_audio(
    os.path.join(
        BASE_DIR,
        "audio_dataset",
        "alarm",
        "35444__fons__synth-alarm-9.wav"
    )
)