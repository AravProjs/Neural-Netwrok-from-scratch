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


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================


class ReLU:
    """
    Rectified Linear Unit (ReLU) Activation Function.
    
    Mathematical Definition:
        ReLU(x) = max(0, x)
        
        Or element-wise:
            f(x) = x  if x > 0
            f(x) = 0  if x <= 0
    
    Derivative:
        dReLU/dx = 1  if x > 0
        dReLU/dx = 0  if x <= 0
    
    Why ReLU?
        - Computationally efficient (just a threshold)
        - No vanishing gradient for positive values
        - Introduces non-linearity (allows network to learn complex patterns)
        - Sparse activation (many neurons output 0) → efficient
    
    Why store input during forward pass?
        - The backward pass needs to know WHERE the input was positive
        - We create a "mask" of which neurons were active
        - Gradient only flows through neurons that were active (x > 0)
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply ReLU activation: output = max(0, x)
        
        Args:
            x: Input array of any shape (batch_size, features)
            
        Returns:
            Output with same shape, negative values replaced with 0
        """
        # Store input for backward pass
        # We need to know which elements were positive
        self.input = x
        
        # ReLU: keep positive values, zero out negative values
        return np.maximum(0, x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute gradient of ReLU for backpropagation.
        
        The gradient of ReLU is:
            - 1 where input > 0 (gradient passes through)
            - 0 where input <= 0 (gradient is blocked)
        
        Chain rule: dL/dx = dL/dy * dy/dx
            where dy/dx is the ReLU derivative (0 or 1)
        
        Args:
            grad_output: Gradient flowing back from the next layer
                        Shape: (batch_size, features)
        
        Returns:
            grad_input: Gradient to pass to the previous layer
                       Shape: same as grad_output
        """
        # Create mask: 1 where input > 0, 0 elsewhere
        # This is the derivative of ReLU
        relu_derivative = (self.input > 0).astype(np.float32)
        
        # Element-wise multiply: gradient only flows where ReLU was active
        return grad_output * relu_derivative
    
    def __repr__(self):
        return "ReLU()"


class Softmax:
    """
    Softmax Activation Function (for multi-class classification output).
    
    Mathematical Definition:
        softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        
        Converts raw scores (logits) into probabilities that sum to 1.
    
    Numerical Stability Trick:
        Problem: exp(x) can overflow for large x (e.g., exp(1000) = inf)
        
        Solution: Subtract max(x) before exponentiating
            softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        
        This is mathematically equivalent but numerically stable because:
            - max(x - max(x)) = 0
            - All values are <= 0
            - exp(0) = 1, exp(negative) < 1
            - No overflow possible!
    
    Why store output during forward pass?
        - Softmax derivative depends on its OUTPUT, not input
        - d(softmax)/d(input) involves the softmax values themselves
        - For efficiency, we combine softmax + cross-entropy in practice
    
    Backward Pass Note:
        - The full Jacobian of softmax is complex (matrix, not vector)
        - When combined with cross-entropy loss, it simplifies beautifully
        - We'll handle the combined gradient in the loss function
        - Here we just pass through the gradient for now
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply softmax activation with numerical stability.
        
        Args:
            x: Logits of shape (batch_size, num_classes)
            
        Returns:
            Probabilities of shape (batch_size, num_classes)
            Each row sums to 1.
        """
        # Numerical stability: subtract max from each row
        # keepdims=True maintains shape for broadcasting
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        
        # Compute exponentials
        exp_x = np.exp(x_shifted)
        
        # Normalize: divide by sum of exponentials for each sample
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for softmax.
        
        Note: In practice, we combine softmax + cross-entropy loss
        because the combined gradient is much simpler:
            dL/dz = softmax(z) - y_true  (predicted - actual)
        
        For now, we just pass through the gradient.
        The actual gradient computation happens in the loss function.
        
        Args:
            grad_output: Gradient from loss function
            
        Returns:
            Same gradient (handled by combined softmax + cross-entropy)
        """
        # Pass through - actual gradient handled by cross-entropy loss
        return grad_output
    
    def __repr__(self):
        return "Softmax()"


class NeuralNetwork:
    """
    A fully-connected neural network for multi-class classification.
    
    TODO: Will be implemented incrementally.
    """
    pass
