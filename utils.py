"""
Utility Functions for Neural Network Training

Contains helper functions for:
- Loading MNIST dataset
- Data preprocessing (normalization, one-hot encoding)
- Visualization and plotting
"""

import numpy as np


def load_mnist():
    """
    Load MNIST dataset.
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    # We'll use keras.datasets just for loading the data
    # All neural network logic will be pure numpy
    from tensorflow.keras.datasets import mnist
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, y_train, X_test, y_test):
    """
    Preprocess MNIST data for neural network training.
    
    Steps:
    1. Flatten images from (28, 28) to (784,)
    2. Normalize pixel values to [0, 1]
    3. One-hot encode labels
    
    Returns:
        X_train, y_train, X_test, y_test (preprocessed)
    """
    # Flatten: (N, 28, 28) -> (N, 784)
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
    
    # Normalize to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # One-hot encode labels
    y_train = one_hot_encode(y_train, num_classes=10)
    y_test = one_hot_encode(y_test, num_classes=10)
    
    return X_train, y_train, X_test, y_test


def one_hot_encode(labels, num_classes):
    """
    Convert integer labels to one-hot encoded vectors.
    
    Args:
        labels: Array of integer labels (N,)
        num_classes: Number of classes
        
    Returns:
        One-hot encoded array (N, num_classes)
    """
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


def plot_training_history(history):
    """
    Plot training loss and accuracy curves.
    
    Args:
        history: Dictionary with 'loss' and 'accuracy' lists
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        plt.show()
        
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")


def accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Args:
        y_true: One-hot encoded true labels (N, num_classes)
        y_pred: Predicted probabilities (N, num_classes)
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(true_labels == pred_labels)

