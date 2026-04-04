"""
cnn_model.py
========================
Person 2 — CNN Model Training & Evaluation

Uses preprocessed files from:
    DataFiles/

Outputs:
    FigureFiles/
        - cnn_training_curve.png
        - cnn_confusion_matrix.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DATA_DIR = "DataFiles"
FIG_DIR = "FigureFiles"

CLASS_NAMES = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────
def load_data():

    print("="*50)
    print("Loading preprocessed CIFAR-10 data")
    print("="*50)

    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_test  = np.load(f"{DATA_DIR}/y_test.npy")

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STEP 2 — DATA AUGMENTATION
# ─────────────────────────────────────────────
def create_augmentation():

    augmenter = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    return augmenter


# ─────────────────────────────────────────────
# STEP 3 — BUILD CNN
# ─────────────────────────────────────────────
def build_cnn():

    model = models.Sequential([

        # Block 1
        layers.Conv2D(32,(3,3),padding='same',
                      input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),

        # Block 2
        layers.Conv2D(64,(3,3),padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2,2),

        # Block 3
        layers.Conv2D(128,(3,3),padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Flatten(),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


# ─────────────────────────────────────────────
# STEP 4 — TRAIN MODEL
# ─────────────────────────────────────────────
def train(model, X_train, y_train):

    augmenter = create_augmentation()

    history = model.fit(
        augmenter(X_train),
        y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    return history


# ─────────────────────────────────────────────
# STEP 5 — TRAINING CURVES
# ─────────────────────────────────────────────
def plot_training(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')

    plt.plot(history.history['loss'], '--', label='Train Loss')
    plt.plot(history.history['val_loss'], '--', label='Val Loss')

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("CNN Training Curves")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cnn_training_curve.png", dpi=150)
    plt.close()

    print("Saved training curve.")


# ─────────────────────────────────────────────
# STEP 6 — EVALUATION + METRICS
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test):

    print("\nEvaluating CNN...")

    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    accuracy = np.mean(y_pred == y_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES
    ))

    # ── Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10,8))
    ConfusionMatrixDisplay(
        cm_norm,
        display_labels=CLASS_NAMES
    ).plot(ax=ax, cmap="Blues", values_format=".2f")

    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cnn_confusion_matrix.png", dpi=150)
    plt.close()

    print("Saved confusion matrix.")

    # ── Bootstrap Confidence Intervals
    rng = np.random.default_rng(42)
    n = len(y_test)

    accs, f1s = [], []

    for _ in range(1000):
        idx = rng.integers(0, n, n)

        accs.append(np.mean(y_pred[idx] == y_test[idx]))
        f1s.append(
            f1_score(y_test[idx], y_pred[idx], average="macro")
        )

    acc_ci = np.percentile(accs, [2.5, 97.5])
    f1_ci  = np.percentile(f1s,  [2.5, 97.5])

    print("\n95% Confidence Intervals")
    print(f"Accuracy CI : [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
    print(f"Macro-F1 CI : [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    X_train, X_test, y_train, y_test = load_data()

    model = build_cnn()

    history = train(model, X_train, y_train)

    plot_training(history)

    evaluate(model, X_test, y_test)

    print("\nDONE — CNN training complete!")


if __name__ == "__main__":
    main()