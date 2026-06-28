# Real-Time Suspicious Activity Detection System

A real-time surveillance system that combines computer vision and deep learning-based audio classification to detect suspicious activities.

## Features

- Intruder detection using YOLOv8
- Audio event classification using a CNN
- Detection of scream, glass break, and alarm sounds
- Live monitoring dashboard
- SQLite database event logging
- Real-time audio and video processing

## Deep Learning Pipeline

Audio Dataset → Spectrogram Generation → CNN Training → Real-Time Audio Classification

## Technologies Used

- Python
- TensorFlow / Keras
- YOLOv8
- OpenCV
- Librosa
- FastAPI
- SQLite

## Project Structure

- `ml_backend/` – Deep learning models and detection logic
- `api_backend/` – FastAPI backend
- `db/` – Event database management
- `frontend/` – Monitoring dashboard

## Future Improvements

- Improved audio dataset
- Mobile notifications
- Multi-camera support