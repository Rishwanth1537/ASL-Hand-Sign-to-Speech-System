import os
import time
import numpy as np
import tensorflow as tf
import pickle
import keyboard
import pyttsx3
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = tf.keras.models.load_model(
    "asl_landmark_model_new.keras",
    compile=False
)

with open("label_encoder_new.pkl", "rb") as f:
    label_encoder = pickle.load(f)


client = OpenAI(api_key=api_key)

tts = pyttsx3.init()
tts.setProperty("rate", 160)



current_word = ""
words = []

print("\n=== ASL → LLM → SPEECH READY ===")
print("C         → capture letter (MediaPipe side)")
print("BACKSPACE → delete letter")
print("SPACE     → finish word")
print("ENTER     → generate sentence + speak")
print("================================\n")

while True:

    if os.path.exists("landmarks.npy"):
        landmarks = np.load("landmarks.npy")
        os.remove("landmarks.npy")

        if landmarks.shape[0] == 63:
            preds = model.predict(landmarks.reshape(1, -1), verbose=0)
            idx = np.argmax(preds)
            letter = label_encoder.inverse_transform([idx])[0]
            confidence = preds[0][idx]

            current_word += letter
            print(f"Letter: {letter} | Conf: {confidence:.2f} | Word: {current_word}")

    if keyboard.is_pressed("backspace"):
        if current_word:
            current_word = current_word[:-1]
            print("Backspace →", current_word if current_word else "[empty]")
            time.sleep(0.3)

    if keyboard.is_pressed("space"):
        if current_word:
            words.append(current_word)
            print("Word added:", current_word)
            print("Words:", words)
            current_word = ""
            time.sleep(0.5)

    if keyboard.is_pressed("enter"):
        if current_word:
            words.append(current_word)
            current_word = ""

        if words:
            print("\nRaw ASL words:", words)

            prompt = (
                "You are an ASL language assistant.\n"
                "Convert the following ASL words into a natural, grammatically correct English sentence.\n"
                "ASL may omit articles, tense markers, or word order—fix these while preserving the meaning.\n"
                "Return only the final English sentence.\n\n"
                f"{words}"
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            sentence = response.choices[0].message.content.strip()

            print("\nFinal sentence:", sentence)

            tts.say(sentence)
            tts.runAndWait()

            words.clear()

        time.sleep(1)

    time.sleep(0.1)
