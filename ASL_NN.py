!pip install -U tensorflow==2.20.0 numpy pandas==2.2.2 scikit-learn
import tensorflow as tf, numpy as np, pandas as pd
print(tf.__version__, np.__version__, pd.__version__)
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import pickle
df = pd.read_csv("asl_landmarks.csv")

print("CSV shape:", df.shape)
print("Unique labels:", df["label"].nunique())
X = df.iloc[:, :-1].values.astype("float32")
y = df.iloc[:, -1].values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

NUM_CLASSES = len(label_encoder.classes_)
print("Number of classes:", NUM_CLASSES)


with open("label_encoder_new.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

model = Sequential([
    Dense(256, activation="relu", input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(29, activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop]
)

model.save("asl_landmark_model_new.keras")
print("Model saved safely")

