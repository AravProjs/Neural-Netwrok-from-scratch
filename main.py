"""
MNIST Handwritten Digit Classification
Neural Network from Scratch using NumPy

This script trains a neural network on the MNIST dataset to classify
handwritten digits (0-9) with ~97% accuracy.

Architecture: 784 → 128 → 64 → 10 (with ReLU and Softmax)
"""

import numpy as np
from neural_network import NeuralNetwork, CrossEntropyLoss
from utils import load_mnist, preprocess_data, plot_training_history, accuracy


def create_mini_batches(X, y, batch_size):
    """
    Create mini-batches from the dataset.
    
    Args:
        X: Input data (N, features)
        y: Labels (N, classes)
        batch_size: Size of each mini-batch
    
    Yields:
        Tuples of (X_batch, y_batch)
    """
    n_samples = X.shape[0]
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Create batches
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        yield X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]


def train(network, loss_fn, X_train, y_train, X_test, y_test,
          epochs=10, learning_rate=0.01, batch_size=32):
    """
    Train the neural network using mini-batch gradient descent.
    
    Args:
        network: NeuralNetwork instance
        loss_fn: CrossEntropyLoss instance
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
        epochs: Number of training epochs
        learning_rate: Learning rate for gradient descent
        batch_size: Size of mini-batches
    
    Returns:
        history: Dictionary with training metrics
    """
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size
    
    # Track training history
    history = {
        'loss': [],
        'accuracy': [],
        'val_accuracy': []
    }
    
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    print(f"Training samples: {n_samples:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {n_batches}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print("=" * 60 + "\n")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # =====================================================================
        # TRAINING PHASE
        # =====================================================================
        batch_num = 0
        for X_batch, y_batch in create_mini_batches(X_train, y_train, batch_size):
            # -----------------------------------------------------------------
            # 1. FORWARD PASS
            # -----------------------------------------------------------------
            predictions = network.forward(X_batch)
            
            # -----------------------------------------------------------------
            # 2. COMPUTE LOSS
            # -----------------------------------------------------------------
            loss = loss_fn.compute_loss(predictions, y_batch)
            epoch_loss += loss
            
            # Track accuracy
            pred_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
            epoch_correct += np.sum(pred_labels == true_labels)
            epoch_total += len(y_batch)
            
            # -----------------------------------------------------------------
            # 3. COMPUTE GRADIENT
            # -----------------------------------------------------------------
            grad = loss_fn.compute_gradient(predictions, y_batch)
            
            # -----------------------------------------------------------------
            # 4. BACKWARD PASS
            # -----------------------------------------------------------------
            network.backward(grad)
            
            # -----------------------------------------------------------------
            # 5. UPDATE WEIGHTS
            # -----------------------------------------------------------------
            network.update_weights(learning_rate)
            
            batch_num += 1
            
            # Progress indicator every 100 batches
            if batch_num % 500 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} | "
                      f"Batch {batch_num}/{n_batches} | "
                      f"Loss: {loss:.4f}")
        
        # =====================================================================
        # EPOCH METRICS
        # =====================================================================
        avg_loss = epoch_loss / n_batches
        train_acc = epoch_correct / epoch_total
        
        # Evaluate on test set
        test_predictions = network.forward(X_test)
        test_acc = accuracy(y_test, test_predictions)
        
        # Store history
        history['loss'].append(avg_loss)
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(test_acc)
        
        # Print epoch summary
        print(f"\n{'-' * 60}")
        print(f"Epoch {epoch + 1}/{epochs} Complete")
        print(f"  Training Loss:     {avg_loss:.4f}")
        print(f"  Training Accuracy: {train_acc * 100:.2f}%")
        print(f"  Test Accuracy:     {test_acc * 100:.2f}%")
        print(f"{'-' * 60}\n")
    
    return history


def main():
    """Main function to train and evaluate the neural network."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # =========================================================================
    # 1. LOAD AND PREPROCESS DATA
    # =========================================================================
    print("\n" + "=" * 60)
    print("LOADING MNIST DATASET")
    print("=" * 60)
    
    # Load raw MNIST data
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Raw data loaded:")
    print(f"  X_train: {X_train.shape} (images)")
    print(f"  y_train: {y_train.shape} (labels)")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")
    
    # Preprocess: flatten, normalize, one-hot encode
    X_train, y_train, X_test, y_test = preprocess_data(
        X_train, y_train, X_test, y_test
    )
    print(f"\nAfter preprocessing:")
    print(f"  X_train: {X_train.shape} (flattened, normalized)")
    print(f"  y_train: {y_train.shape} (one-hot encoded)")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")
    
    # =========================================================================
    # 2. CREATE NETWORK AND LOSS FUNCTION
    # =========================================================================
    print("\n" + "=" * 60)
    print("CREATING NEURAL NETWORK")
    print("=" * 60)
    
    # Architecture: 784 → 128 → 64 → 10
    # - 784: Input (28x28 flattened images)
    # - 128: Hidden layer 1 with ReLU
    # - 64:  Hidden layer 2 with ReLU
    # - 10:  Output layer with Softmax (10 digit classes)
    network = NeuralNetwork([784, 128, 64, 10])
    
    # Cross-entropy loss for classification
    loss_fn = CrossEntropyLoss()
    print(f"\nLoss function: {loss_fn}")
    
    # =========================================================================
    # 3. TRAIN THE NETWORK
    # =========================================================================
    history = train(
        network=network,
        loss_fn=loss_fn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=10,
        learning_rate=0.1,  # Higher LR works well with mini-batch
        batch_size=32
    )
    
    # =========================================================================
    # 4. FINAL EVALUATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Final test accuracy
    final_predictions = network.forward(X_test)
    final_accuracy = accuracy(y_test, final_predictions)
    
    print(f"\nFinal Test Accuracy: {final_accuracy * 100:.2f}%")
    print(f"Correctly classified: {int(final_accuracy * X_test.shape[0]):,} / {X_test.shape[0]:,}")
    
    # =========================================================================
    # 5. SAVE TRAINING PLOT
    # =========================================================================
    print("\nSaving training history plot...")
    try:
        plot_training_history(history)
        print("Plot saved to 'training_history.png'")
    except Exception as e:
        print(f"Could not save plot: {e}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60 + "\n")
    
    return network, history


if __name__ == "__main__":
    network, history = main()
