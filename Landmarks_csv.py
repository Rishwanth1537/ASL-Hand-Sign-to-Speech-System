import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# --- PATHS ---
DATASET_PATH = "asl_alphabet_train"   # Folder with subfolders A/, B/, ...
CSV_OUT = "asl_landmarks.csv"

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# --- Collect All Files ---
image_paths = []
labels = []

for label in os.listdir(DATASET_PATH):
    label_folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_folder):
        continue

    for fname in os.listdir(label_folder):
        if fname.lower().endswith(".jpg") or fname.lower().endswith(".png"):
            image_paths.append(os.path.join(label_folder, fname))
            labels.append(label)

# --- Prepare Storage ---
rows = []
print(f"[+] Processing {len(image_paths)} images...")

# --- MAIN LOOP ---
for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Resize to fixed size (optional)
    img = cv2.resize(img, (640, 480))

    # Convert for MP
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        # Skip images with no landmarks
        continue

    hand_landmarks = results.multi_hand_landmarks[0]

    # Extract the 21 x,y,z landmarks
    landmark_list = []
    for lm in hand_landmarks.landmark:
        landmark_list.extend([lm.x, lm.y, lm.z])

    # Append label and row
    rows.append(landmark_list + [label])

# --- SAVE CSV ---
columns = []
for i in range(21):
    columns += [f"x{i}", f"y{i}", f"z{i}"]
columns.append("label")

df = pd.DataFrame(rows, columns=columns)
df.to_csv(CSV_OUT, index=False)

print(f"[+] Done! Saved {len(df)} samples to {CSV_OUT}")
