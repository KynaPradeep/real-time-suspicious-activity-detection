Real-Time Suspicious Activity Detection System

A deep learning based surveillance system that detects suspicious human activity using CNN/YOLO models and sends real-time SMS alerts to designated contacts.

📌 Overview

This project is designed to enhance security by automatically detecting suspicious activities from video input. It leverages computer vision and deep learning techniques to classify behavior and instantly notify users via SMS.

🧠 Key Features

📷 Real-time video processing using OpenCV

🧠 Activity detection using CNN / YOLO

🚨 Suspicious activity identification

📩 Instant SMS alerts using API integration

⚡ Automated monitoring system

🛠️ Tech Stack

Python

OpenCV

TensorFlow / PyTorch

YOLO (Object Detection)

Twilio API (SMS alerts)

⚙️ How It Works

Capture video stream

Process frames using OpenCV

Run detection model (CNN/YOLO)

Identify suspicious activity

Trigger SMS alert

📂 Project Structure
ml_backend/        # ML models and detection logic  
frontend/          # UI components (if applicable)  
api_backend/       # Backend APIs  
docs/              # Documentation  
🎯 Use Cases

Smart surveillance systems

Home security

Public safety monitoring

Office/building security

📈 Future Improvements

Improve detection accuracy

Add multi-class activity detection

Deploy as web/mobile app

Integrate live CCTV feeds
