# CIFAR-10 Preprocessing and Baseline Model

This directory contains the code to download, explore, preprocess, and train a baseline model on the CIFAR-10 dataset.

## Scripts
- `cifar10_preprocessing.py`: The main script that performs all the operations described below.

## What the Script Does:
When you run `cifar10_preprocessing.py`, it performs the following steps:
1. **Loads Data**: Downloads the CIFAR-10 dataset using TensorFlow/Keras.
2. **Explores Data**: Checks the class distribution, looking for NaNs, and reviewing the label ranges.
3. **Visualizes Samples**: Generates a grid of sample images (one for each class) and a bar chart showing the class distribution, saving them to the `FigureFiles/` directory.
4. **Preprocesses Data**: Normalizes the pixel values (scaling from `0-255` to `0.0-1.0`) and flattens the 32x32x3 images into 1D arrays of size 3072.
5. **Trains a Baseline Model**: Trains a simple Multinomial Logistic Regression model using `scikit-learn` on a 10k subset of the training data.
6. **Evaluates the Model**: Generates accuracy scores, a classification report, and a confusion matrix which is saved to `FigureFiles/`.
7. **Saves Data**: Saves all the necessary training and testing tensors to the `DataFiles/` directory as binary `.npy` files.

## Generated Output Files

### `DataFiles/`
This folder contains the preprocessed datasets stored as NumPy binary files (`.npy`). They cannot be opened directly in a text editor.

- `X_train.npy` (Shape: `50000, 32, 32, 3`): Original normalized training images.
- `X_test.npy` (Shape: `10000, 32, 32, 3`): Original normalized testing images.
- `X_train_flat.npy` (Shape: `50000, 3072`): Flattened training images, ready for traditional ML models like SVM or Logistic Regression.
- `X_test_flat.npy` (Shape: `10000, 3072`): Flattened testing images.
- `y_train.npy` (Shape: `50000,`): Training labels.
- `y_test.npy` (Shape: `10000,`): Testing labels.

### `FigureFiles/`
This folder contains the generated visual plots.
- `cifar10_samples.png`: Example images from each of the 10 classes.
- `class_distribution.png`: Bar chart of how many images exist per class.
- `baseline_confusion_matrix.png`: Confusion matrix of the baseline Logistic Regression model running on the test set.

## How to Access the `.npy` Files
Because the files in `DataFiles/` are saved as binary arrays, you must load them with Python and NumPy if you want to inspect them or use them in future models:

```python
import numpy as np

# Example: Loading the flattened datasets
X_train_flat = np.load('DataFiles/X_train_flat.npy')
y_train = np.load('DataFiles/y_train.npy')

print("Training Data Shape:", X_train_flat.shape)
print("Training Labels Shape:", y_train.shape)
```
