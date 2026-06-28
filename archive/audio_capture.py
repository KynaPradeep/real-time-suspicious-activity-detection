import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import queue

SAMPLE_RATE = 22050      
DURATION = 2              

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def record_audio():
    print("Recording audio... Press CTRL+C to stop.")
    
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback):
        while True:
            audio_data = audio_queue.get().flatten()
    
            mel = librosa.feature.melspectrogram(y=audio_data, sr=SAMPLE_RATE, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            plt.clf()
            librosa.display.specshow(mel_db, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
            plt.title("Mel Spectrogram (Live Audio)")
            plt.pause(0.01)

if __name__ == "__main__":
    plt.figure(figsize=(6, 4))
    record_audio()
