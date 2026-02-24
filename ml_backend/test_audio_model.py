import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL_PATH = "audio_model.h5"
CLASSES = ["scream", "glass_break", "alarm", "normal"]

model = tf.keras.models.load_model(MODEL_PATH)

def predict_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S = librosa.power_to_db(S, ref=np.max)
    S = np.resize(S, (128,128))
    S = np.stack([S, S, S], axis=-1) / 255.0
    S = np.expand_dims(S, axis=0)

    pred = model.predict(S)
    class_id = np.argmax(pred)
    
    print(f"Prediction: {CLASSES[class_id]} (confidence: {pred[0][class_id]:.2f})")

predict_audio("audio_dataset/scream/220663__marionagm90__scream.wav")
