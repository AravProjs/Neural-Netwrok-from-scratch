"""
Neural Network Implementation from Scratch

Contains:
- Layer class: Individual layer with weights, biases, and activation
- NeuralNetwork class: Full network with forward/backward propagation
"""

import numpy as np


# =============================================================================
# WEIGHT INITIALIZATION - He Initialization
# =============================================================================
# Formula: W = np.random.randn(fan_in, fan_out) * sqrt(2 / fan_in)
# 
# Why He Initialization?
# - Designed specifically for ReLU activation functions
# - ReLU kills ~50% of neurons (outputs 0 for negative inputs)
# - The factor of 2 compensates for this variance reduction
# - Keeps activations and gradients stable across deep networks
# =============================================================================


class Layer:
    """
    A single fully-connected layer in the neural network.
    
    Uses He initialization for weights:
        W ~ N(0, sqrt(2/fan_in))
        b = 0
    
    Attributes:
        weights: Weight matrix of shape (input_size, output_size)
        biases: Bias vector of shape (1, output_size)
    """
    
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize a fully-connected layer with He initialization.
        
        Args:
            input_size: Number of input features (fan_in)
            output_size: Number of output neurons (fan_out)
        """
        # He initialization: std = sqrt(2 / fan_in)
        std = np.sqrt(2.0 / input_size)
        
        self.weights = np.random.randn(input_size, output_size) * std
        self.biases = np.zeros((1, output_size))
        
        # Store dimensions for reference
        self.input_size = input_size
        self.output_size = output_size
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute z = X · W + b
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            z: Pre-activation output of shape (batch_size, output_size)
        """
        # Store input for use in backward pass later
        self.input = X
        
        # Linear transformation: z = X · W + b
        z = np.dot(X, self.weights) + self.biases
        
        return z
    
    def __repr__(self):
        return f"Layer(input={self.input_size}, output={self.output_size}, init='he')"


class NeuralNetwork:
    """
    A fully-connected neural network for multi-class classification.
    
    TODO: Will be implemented incrementally.
    """
    pass
