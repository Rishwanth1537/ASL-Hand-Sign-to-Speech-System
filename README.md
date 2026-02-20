# 🖐️ ASL Hand Sign to Speech System

An AI-powered pipeline that converts **American Sign Language (ASL)** hand signs into spoken English.

This project integrates:
- **MediaPipe** – Hand landmark detection  
- **TensorFlow** – Neural network classifier  
- **OpenAI LLM** – Sentence refinement  
- **Text-to-Speech (pyttsx3)** – Speech output  

---

## 📌 Project Description

This system captures hand landmarks using MediaPipe, classifies ASL letters using a trained neural network, forms words from predicted letters, and refines them into grammatically correct English sentences using an LLM before generating speech output.

The project demonstrates integration of:
- Computer Vision
- Deep Learning
- Natural Language Processing
- Speech Synthesis

---

## 🧠 Model Architecture

- **Input:** 63 features (21 hand landmarks × x, y, z)
- Dense(256) → BatchNorm → Dropout(0.3)
- Dense(128) → BatchNorm → Dropout(0.3)
- Dense(64) → BatchNorm → Dropout(0.3)
- Output: Softmax (29 classes)

Training includes:
- 80/20 train-validation split
- Early stopping
- Label encoding persistence

---

## 🚀 System Pipeline

1. Capture hand landmarks (MediaPipe)
2. Classify ASL letter (Neural Network)
3. Build words from predicted letters
4. Send word sequence to LLM
5. Generate natural English sentence
6. Convert to speech

---

## 🖥️ Controls

| Key        | Action |
|------------|--------|
| C          | Capture letter |
| BACKSPACE  | Delete letter |
| SPACE      | Finish word |
| ENTER      | Generate sentence + speak |
| Q          | Quit |

---
## Dataset

The dataset is not included in this repository due to size limitations.

Download it here:
[(https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data)]

---

## ⚙️ Installation

```bash
pip install -U tensorflow==2.20.0 numpy pandas==2.2.2 scikit-learn
pip install mediapipe opencv-python keyboard pyttsx3 python-dotenv openai
```
- By implementing landmarks_csv.py file you will get asl_landmarks.csv dataset then import it to the ASL_NN.py
- You will get asl_landmark_model_new.keras and label_encoder_new.pkl after implementing ASL_NN.py
---

## 🎯 Future Improvements

- Temporal smoothing across frames
- Automatic letter detection (no keypress)
- Confidence-based filtering
- Continuous sign recognition

