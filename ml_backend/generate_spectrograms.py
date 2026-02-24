import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

AUDIO_DATASET_PATH = "audio_dataset"
SPEC_DATASET_PATH = "spectrogram_dataset"

CLASSES = ["scream", "glass_break", "alarm", "normal"]

for c in CLASSES:
    os.makedirs(os.path.join(SPEC_DATASET_PATH, c), exist_ok=True)

def create_spectrogram(audio_file, save_path):
    y, sr = librosa.load(audio_file, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(3, 3))
    plt.axis("off")
    librosa.display.specshow(mel_db, sr=sr, x_axis=None, y_axis=None)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def generate_all():
    for label in CLASSES:
        class_path = os.path.join(AUDIO_DATASET_PATH, label)
        output_path = os.path.join(SPEC_DATASET_PATH, label)

        for file in os.listdir(class_path):
            if file.endswith(".wav") or file.endswith(".mp3"):
                input_file = os.path.join(class_path, file)
                output_file = os.path.join(output_path, file.replace(".wav", ".png").replace(".mp3", ".png"))

                print(f"🎵 Processing {input_file} → {output_file}")
                create_spectrogram(input_file, output_file)

if __name__ == "__main__":
    print("Generating Spectrogram Dataset ...")
    generate_all()
    print("Done! Spectrogram images saved.")
