"""
cifar10_preprocessing.py

Author: Dat Bui
Course: CSCE 478

Functionality:
- Loads the CIFAR-10 dataset using Keras
- Explores dataset structure, class distribution, and data integrity
- Visualizes sample images and class distribution
- Preprocesses data by normalizing pixel values and flattening inputs
- Trains a Logistic Regression (softmax) baseline model
- Evaluates performance using accuracy and a classification report
- Computes a normalized confusion matrix and 95% confidence interval
- Saves processed datasets for downstream CNN training

Outputs:
- DataFiles/
    * X_train.npy          (50000, 32, 32, 3)
    * X_test.npy           (10000, 32, 32, 3)
    * X_train_flat.npy     (50000, 3072)
    * X_test_flat.npy      (10000, 3072)
    * y_train.npy          (50000,)
    * y_test.npy           (10000,)
- FigureFiles/
    * cifar10_samples.png
    * class_distribution.png
    * baseline_confusion_matrix.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

DATA_DIR = "DataFiles"
FIG_DIR  = "FigureFiles"

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


#========================CREATE FOLDERS========================#

def create_folders():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


#========================LOAD DATA========================#

# Load the CIFAR-10 dataset and print basic information about the shapes and pixel ranges.
def load_data():
    print("=" * 50)
    print(" STEP 1 — Loading CIFAR-10 Dataset")
    print("=" * 50)

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print("X_train shape :", X_train.shape)
    print("X_test  shape :", X_test.shape)
    print("y_train shape :", y_train.shape)
    print("y_test  shape :", y_test.shape)
    print("Pixel range   :", X_train.min(), "-", X_train.max())
    print("Data type     :", X_train.dtype)

    return X_train, y_train, X_test, y_test


#========================EXPLORE DATA========================#

# Explore the dataset by printing class distribution, checking for NaN values, and verifying label ranges.
def explore_data(X_train, y_train):
    print("\n" + "=" * 50)
    print(" STEP 2 — Exploring Data")
    print("=" * 50)

    y_flat = y_train.flatten()

    print("\nClass distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(y_flat == i)
        print(f"  Class {i} ({name:>12}): {count} images")

    print("\nNaN in train:", np.isnan(X_train.astype('float32')).any())
    print("NaN in test :", np.isnan(X_train.astype('float32')).any())
    print("Label range  :", y_flat.min(), "-", y_flat.max())


#========================VISUALIZE SAMPLES========================#

# Visualize one sample image from each class and the overall class distribution as a bar chart. Save both figures.
def visualize_samples(X_train, y_train):
    print("\n" + "=" * 50)
    print(" STEP 3 — Visualizing Sample Images")
    print("=" * 50)

    y_flat = y_train.flatten()

    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    fig.suptitle('CIFAR-10 — One Sample Per Class',
                 fontsize=14, fontweight='bold', y=1.02)

    for i, ax in enumerate(axes.flatten()):
        idx = np.where(y_flat == i)[0][0]
        ax.imshow(X_train[idx])
        ax.set_title(f"{i}: {CLASS_NAMES[i]}", fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/cifar10_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR}/cifar10_samples.png")

    counts = [np.sum(y_flat == i) for i in range(10)]

    plt.figure(figsize=(11, 4))
    bars = plt.bar(CLASS_NAMES, counts,
                   color='steelblue', edgecolor='black', width=0.6)

    plt.title('Class Distribution — Training Set')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=30, ha='right')

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30,
                 str(count), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR}/class_distribution.png")


#========================PREPROCESS DATA========================#

# Normalize pixel values to [0, 1], flatten the images for logistic regression, 
# and flatten the labels. Print shapes and value ranges after each step.
def preprocess(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 50)
    print(" STEP 4 — Preprocessing")
    print("=" * 50)

    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0

    print("After normalization — min:", X_train.min(), " max:", X_train.max())

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)

    print("X_train_flat shape:", X_train_flat.shape)
    print("X_test_flat  shape:", X_test_flat.shape)

    y_train_flat = y_train.flatten()
    y_test_flat  = y_test.flatten()

    print("y_train_flat shape:", y_train_flat.shape)
    print("y_test_flat  shape:", y_test_flat.shape)

    return X_train, X_test, X_train_flat, X_test_flat, y_train_flat, y_test_flat


#========================TRAIN BASELINE MODEL========================#

# Train a Logistic Regression model on the flattened training data and return the trained model.
def train_baseline(X_train_flat, y_train_flat):
    print("\n" + "=" * 50)
    print(" STEP 5 — Training Softmax Baseline")
    print("=" * 50)

    model = LogisticRegression(
        max_iter=1000,
        C=0.01,
        solver='saga',
        multi_class='multinomial',
        random_state=42,
        n_jobs=-1
    )

    print("Training... (subset of 10k for speed)")
    model.fit(X_train_flat[:10000], y_train_flat[:10000])
    print("Training complete!")

    return model


#=========================EVALUATE BASELINE MODEL========================#

# Evaluate the baseline model on the test set, print accuracy, classification report, 
# and save a normalized confusion matrix. Also compute a 95% confidence interval using bootstrapping.
def evaluate(model, X_test_flat, y_test_flat):
    print("\n" + "=" * 50)
    print(" STEP 6 — Evaluating Baseline Model")
    print("=" * 50)

    y_pred = model.predict(X_test_flat)

    acc = accuracy_score(y_test_flat, y_pred)
    print(f"\nBaseline Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    print("\nPer-class Report:")
    print(classification_report(y_test_flat, y_pred, target_names=CLASS_NAMES))

    cm      = confusion_matrix(y_test_flat, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap='Blues', values_format='.2f')

    plt.xticks(rotation=40, ha='right')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/baseline_confusion_matrix.png', dpi=150)
    plt.close()
    print(f"Saved: {FIG_DIR}/baseline_confusion_matrix.png")

    rng = np.random.default_rng(42)
    boot_accs = []
    n = len(y_test_flat)

    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        boot_accs.append(accuracy_score(y_test_flat[idx], y_pred[idx]))

    ci_lo = np.percentile(boot_accs, 2.5)
    ci_hi = np.percentile(boot_accs, 97.5)

    print(f"Accuracy : {acc:.4f}")
    print(f"95% CI   : [{ci_lo:.4f}, {ci_hi:.4f}]")

    return acc, ci_lo, ci_hi


#========================SAVE PROCESSED FILES========================#

# Save the processed datasets as .npy files for later use in CNN training.
def save_files(X_train, X_test, X_train_flat, X_test_flat,
               y_train_flat, y_test_flat):

    print("\n" + "=" * 50)
    print(" STEP 7 — Saving Files")
    print("=" * 50)

    np.save(f'{DATA_DIR}/X_train.npy', X_train)
    np.save(f'{DATA_DIR}/X_test.npy', X_test)
    np.save(f'{DATA_DIR}/X_train_flat.npy', X_train_flat)
    np.save(f'{DATA_DIR}/X_test_flat.npy', X_test_flat)
    np.save(f'{DATA_DIR}/y_train.npy', y_train_flat)
    np.save(f'{DATA_DIR}/y_test.npy', y_test_flat)

    print(f"Saved all .npy files in {DATA_DIR}/")


#========================MAIN FUNCTION========================#

def main():
    create_folders()

    X_train, y_train, X_test, y_test = load_data()
    explore_data(X_train, y_train)
    visualize_samples(X_train, y_train)

    X_train, X_test, X_train_flat, X_test_flat, \
        y_train_flat, y_test_flat = preprocess(X_train, y_train, X_test, y_test)

    model = train_baseline(X_train_flat, y_train_flat)
    evaluate(model, X_test_flat, y_test_flat)

    save_files(X_train, X_test, X_train_flat, X_test_flat,
               y_train_flat, y_test_flat)

    print("\n" + "=" * 50)
    print(" DONE — Everything saved in folders.")
    print("=" * 50)


if __name__ == "__main__":
    main()