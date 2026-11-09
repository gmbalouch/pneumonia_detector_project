"""Retrain the model locally. Needs: pip install medmnist tensorflow numpy matplotlib"""
import numpy as np
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST
from tensorflow.keras import layers, models


def load(split):
    ds = PneumoniaMNIST(split=split, download=True)
    X = np.array([np.array(img, dtype=np.float32) / 255.0 for img, _ in ds])[..., np.newaxis]
    y = np.array([lbl[0] for _, lbl in ds])
    return X, y


X_train, y_train = load("train")
X_val, y_val = load("val")
X_test, y_test = load("test")

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, 3, activation="relu"), layers.MaxPool2D(2),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPool2D(2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=10, batch_size=64)

print("Test:", model.evaluate(X_test, y_test))
model.save("model/pneumonia_model.keras")

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend(); plt.savefig("training_loss.png")
