"""
cnn_model.py
Authors: Amy Nguyen & Nick Pham
Course: CSCE 478

This script handles the CNN side of our CIFAR-10 project.
It loads the preprocessed .npy files that cifar10_preprocessing.py
saved into DataFiles/, then builds and trains a 3-block CNN with
batch norm and dropout.

After training, it runs a bunch of evaluation stuff:
- Prints accuracy, F1, and classification report
- Saves a normalized confusion matrix
- Computes bootstrap confidence intervals (1000 resamples)
- Trains a quick logreg baseline so we can compare per-class accuracy
- Plots misclassified examples to see where the model messes up
- Generates Grad-CAM heatmaps to visualize what the CNN looks at
- Runs t-SNE on the penultimate layer to see how well classes separate

All figures get saved to FigureFiles/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score, accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = "DataFiles"
FIG_DIR  = "FigureFiles"

CLASS_NAMES = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

os.makedirs(FIG_DIR, exist_ok=True)

# --- model setup ---

def load_data():
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_test  = np.load(f"{DATA_DIR}/y_test.npy")

    print(f"Loaded data — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# augmentation to reduce overfitting
def create_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])


def build_cnn():
    model = models.Sequential([
        # block 1
        layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),

        # block 2
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),

        # block 3
        layers.Conv2D(128, (3,3), padding='same'),
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

# --- training & eval ---


def train(model, X_train, y_train):
    augmenter = create_augmentation()

    history = model.fit(
        augmenter(X_train), y_train,
        epochs=20, batch_size=64,
        validation_split=0.1, verbose=1
    )
    return history


def plot_training(history):
    plt.figure(figsize=(8, 5))
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



def evaluate(model, X_test, y_test):
    print("\nEvaluating CNN...")

    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    accuracy = np.mean(y_pred == y_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", values_format=".2f"
    )
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cnn_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion matrix.")

    rng = np.random.default_rng(42)
    n = len(y_test)
    accs, f1s = [], []

    for _ in range(1000):
        idx = rng.integers(0, n, n)
        accs.append(np.mean(y_pred[idx] == y_test[idx]))
        f1s.append(f1_score(y_test[idx], y_pred[idx], average="macro"))

    acc_ci = np.percentile(accs, [2.5, 97.5])
    f1_ci  = np.percentile(f1s,  [2.5, 97.5])

    print(f"\n95% CI — Accuracy: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
    print(f"95% CI — Macro-F1: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")

# --- extra plots for the report ---


def plot_baseline_vs_cnn(model, X_test, y_test):
    y_pred_cnn = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # quick baseline logreg for comparison
    X_train_flat = np.load(f"{DATA_DIR}/X_train_flat.npy")
    X_test_flat  = np.load(f"{DATA_DIR}/X_test_flat.npy")
    y_train_all  = np.load(f"{DATA_DIR}/y_train.npy")

    print("Training baseline logreg for comparison...")
    baseline = LogisticRegression(
        max_iter=1000, C=0.01, solver='saga',
        random_state=42, n_jobs=-1
    )
    baseline.fit(X_train_flat[:10000], y_train_all[:10000])
    y_pred_bl = baseline.predict(X_test_flat)

    cnn_acc, bl_acc = [], []
    for i in range(10):
        mask = (y_test == i)
        cnn_acc.append(np.mean(y_pred_cnn[mask] == y_test[mask]))
        bl_acc.append(np.mean(y_pred_bl[mask] == y_test[mask]))

    cnn_overall = accuracy_score(y_test, y_pred_cnn)
    bl_overall  = accuracy_score(y_test, y_pred_bl)

    x = np.arange(10)
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w/2, bl_acc, w,
           label=f'Logistic Regression ({bl_overall:.1%})',
           color='#9e9e9e', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, cnn_acc, w,
           label=f'CNN ({cnn_overall:.1%})',
           color='#1976d2', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy: Baseline vs CNN', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/baseline_vs_cnn_accuracy.png", dpi=150)
    plt.close()
    print("Saved baseline vs CNN comparison.")


def plot_misclassified(model, X_test, y_test):
    y_probs = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_probs, axis=1)
    wrong   = np.where(y_pred != y_test)[0]

    print(f"Misclassified: {len(wrong)}/{len(y_test)} ({len(wrong)/len(y_test):.1%})")

    rng = np.random.default_rng(42)
    sample = rng.choice(wrong, size=min(20, len(wrong)), replace=False)

    fig, axes = plt.subplots(4, 5, figsize=(14, 12))
    for i, ax in enumerate(axes.flatten()):
        if i < len(sample):
            idx = sample[i]
            ax.imshow(X_test[idx])
            conf = y_probs[idx, y_pred[idx]]
            ax.set_title(
                f"True: {CLASS_NAMES[y_test[idx]]}\n"
                f"Pred: {CLASS_NAMES[y_pred[idx]]} ({conf:.0%})",
                fontsize=8, color='red', fontweight='bold'
            )
        ax.axis('off')

    plt.suptitle('Misclassified Examples', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/misclassified_examples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved misclassified examples.")


def _gradcam_heatmap(model, img_array, conv_layer_name):
    inp = tf.keras.Input(shape=(32, 32, 3)) # need functional model for keras 3
    x = inp
    conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == conv_layer_name:
            conv_output = x

    grad_model = tf.keras.Model(inputs=inp, outputs=[conv_output, x])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        top_class = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(top_class, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_out[0] @ weights[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def plot_gradcam(model, X_test, y_test):
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            last_conv = layer.name
            break
    print(f"Grad-CAM using layer: {last_conv}")

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for i, ax in enumerate(axes.flatten()):
        idx = np.where(y_test == i)[0][0]
        img = X_test[idx:idx+1]

        heatmap = _gradcam_heatmap(model, img, last_conv)
        heatmap_resized = tf.image.resize(
            heatmap[..., tf.newaxis], (32, 32)
        ).numpy().squeeze()

        pred = np.argmax(model.predict(img, verbose=0))

        ax.imshow(X_test[idx])
        ax.imshow(heatmap_resized, cmap='jet', alpha=0.4)
        ax.set_title(f"True: {CLASS_NAMES[i]}\nPred: {CLASS_NAMES[pred]}", fontsize=9)
        ax.axis('off')

    plt.suptitle('Grad-CAM Heatmaps', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/gradcam_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved Grad-CAM heatmaps.")


def plot_tsne(model, X_test, y_test):
    inp = tf.keras.Input(shape=(32, 32, 3)) # need functional model for keras 3
    x = inp
    for layer in model.layers[:-1]:
        x = layer(x)
    feature_model = tf.keras.Model(inputs=inp, outputs=x)

    n = 3000
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=n, replace=False)

    print(f"Extracting features for {n} samples...")
    features = feature_model.predict(X_test[idx], verbose=0)
    labels = y_test[idx]

    print("Running t-SNE...")
    embeddings = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(10):
        mask = (labels == i)
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=[plt.cm.tab10(i)], label=CLASS_NAMES[i],
                   s=10, alpha=0.6, edgecolors='none')

    ax.legend(markerscale=3, fontsize=9)
    ax.set_title('t-SNE of CNN Features', fontsize=14, fontweight='bold')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/tsne_features.png", dpi=150)
    plt.close()
    print("Saved t-SNE plot.")


# ---

def main():
    X_train, X_test, y_train, y_test = load_data()

    model = build_cnn()
    history = train(model, X_train, y_train)

    plot_training(history)
    evaluate(model, X_test, y_test)

    # extra plots
    plot_baseline_vs_cnn(model, X_test, y_test)
    plot_misclassified(model, X_test, y_test)
    plot_gradcam(model, X_test, y_test)
    plot_tsne(model, X_test, y_test)

    print("\nAll done.")


if __name__ == "__main__":
    main()