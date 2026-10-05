import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from config import SIGNS, SEQ_LEN, FEATURES, DATA_DIR, MODEL_PATH, LABELS_PATH

# 1. Load data
X, y = [], []
for label, sign in enumerate(SIGNS):
    for f in sorted((DATA_DIR / sign).glob("*.npy")):
        X.append(np.load(f))
        y.append(label)
X, y = np.array(X), np.array(y)
print("Data shape:", X.shape)                    # (num_examples, 30, 225)

# 2. Split into train and test
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Build the model
model = Sequential([
    Input(shape=(SEQ_LEN, FEATURES)),
    LSTM(64, return_sequences=True),
    LSTM(128),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(len(SIGNS), activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 4. Train (stops early when it stops improving)
model.fit(Xtr, to_categorical(ytr, len(SIGNS)),
          validation_data=(Xte, to_categorical(yte, len(SIGNS))),
          epochs=200, batch_size=16,
          callbacks=[EarlyStopping(patience=20, restore_best_weights=True)])

# 5. Save
model.save(MODEL_PATH)
json.dump(SIGNS, open(LABELS_PATH, "w"))

# 6. Evaluate
pred = np.argmax(model.predict(Xte), axis=1)
print(classification_report(yte, pred, target_names=SIGNS))
ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=SIGNS).plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig("../results/confusion_matrix.png")