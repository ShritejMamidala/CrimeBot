# CrimeBOT: AI-Powered Porch Pirate Detection

> Real-time surveillance system using computer vision and machine learning to help identify porch pirates and prevent package theft.

---

## 📽️ Demo Video

[![Watch the video](https://img.youtube.com/vi/ccKL_JBEgvA/maxresdefault.jpg)](https://youtu.be/ccKL_JBEgvA)

---

## 📖 Overview

**CrimeBOT** is an AI-driven tool designed to help identify potential porch pirates by analyzing webcam footage in real-time. Using a combination of object detection, face tracking, and advanced machine learning models, it aims to assist law enforcement in recognizing and documenting suspects efficiently.

---

## Key Features

-  **Face Detection & Tracking**: Utilizes YOLOv8 to detect faces and assign persistent IDs across frames.
-  **Dual CNN Models**: One CNN predicts demographic data (e.g., age, gender), while the other extracts 28 visual attributes (e.g., glasses, hairstyle).
-  **Natural Language Output**: Feature vectors from CNNs are processed by a Large Language Model (LLM) to generate clear, objective, human-readable suspect descriptions.
-  **Real-Time Video Processing**: Designed for seamless integration with home security systems or live webcam feeds.

---


## 🚀 How It Works

1. **Input**: Live webcam footage.
2. **Face Detection**: YOLOv8 identifies faces and assigns tracking IDs.
3. **Attribute Extraction**: CNNs extract demographic + physical attributes.
4. **Description Generation**: Processed via LLM to create natural-language suspect profiles.
5. **Output**: Real-time visual tagging + descriptive metadata for use by law enforcement.

---

## 🔒 Ethics & Privacy

CrimeBOT was built with ethical surveillance in mind. It does not perform identity recognition or personal data logging. The generated descriptions are objective and based solely on observable traits.

---
