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
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients for weights, biases, and input.
        
        Given the gradient of the loss with respect to this layer's OUTPUT,
        compute gradients with respect to weights, biases, and INPUT.
        
        Args:
            grad_output: Gradient from the next layer (dL/dz)
                        Shape: (batch_size, output_size)
        
        Returns:
            grad_input: Gradient to pass to the previous layer (dL/dX)
                       Shape: (batch_size, input_size)
        
        Math Derivation:
            Forward:  z = X · W + b
            
            We need:
                dL/dW = dL/dz · dz/dW
                dL/db = dL/dz · dz/db
                dL/dX = dL/dz · dz/dX
        """
        batch_size = grad_output.shape[0]
        
        # =====================================================================
        # 1. GRADIENT FOR WEIGHTS: dL/dW
        # =====================================================================
        # Forward was: z = X · W + b
        # dz/dW = X (the input to this layer)
        # 
        # Chain rule: dL/dW = X.T · dL/dz
        #
        # Why transpose X?
        #   - X shape: (batch_size, input_size)
        #   - grad_output shape: (batch_size, output_size)
        #   - We need result shape: (input_size, output_size) to match W
        #   - X.T @ grad_output: (input_size, batch_size) @ (batch_size, output_size)
        #                      = (input_size, output_size) ✓
        #
        # Why divide by batch_size?
        #   - We're computing the AVERAGE gradient across the batch
        #   - This makes learning rate independent of batch size
        #   - Equivalent to averaging the loss over the batch
        self.grad_weights = np.dot(self.input.T, grad_output) / batch_size
        
        # =====================================================================
        # 2. GRADIENT FOR BIASES: dL/db
        # =====================================================================
        # Forward was: z = X · W + b
        # dz/db = 1 (bias is added directly)
        #
        # Chain rule: dL/db = sum(dL/dz) across batch
        #
        # Why sum across axis=0?
        #   - grad_output shape: (batch_size, output_size)
        #   - Each sample contributes to the bias gradient
        #   - Sum gives total gradient, then divide by batch_size for average
        #   - Result shape: (1, output_size) to match self.biases
        #
        # Why divide by batch_size?
        #   - Same reason as weights: average gradient across batch
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True) / batch_size
        
        # =====================================================================
        # 3. GRADIENT FOR INPUT: dL/dX (to pass to previous layer)
        # =====================================================================
        # Forward was: z = X · W + b
        # dz/dX = W (the weights)
        #
        # Chain rule: dL/dX = dL/dz · W.T
        #
        # Why transpose W?
        #   - grad_output shape: (batch_size, output_size)
        #   - W shape: (input_size, output_size)
        #   - We need result shape: (batch_size, input_size) to match X
        #   - grad_output @ W.T: (batch_size, output_size) @ (output_size, input_size)
        #                      = (batch_size, input_size) ✓
        #
        # Note: We don't divide by batch_size here because:
        #   - This gradient flows to the previous layer
        #   - The previous layer will do its own averaging
        grad_input = np.dot(grad_output, self.weights.T)
        
        return grad_input
    
    def update_weights(self, learning_rate: float) -> None:
        """
        Update weights and biases using gradient descent.
        
        Update rule (vanilla gradient descent):
            W = W - learning_rate * dL/dW
            b = b - learning_rate * dL/db
        
        Args:
            learning_rate: Step size for gradient descent (e.g., 0.01)
        
        Note:
            - Must call backward() first to compute gradients
            - Subtracting gradient moves weights in direction that REDUCES loss
        """
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases
    
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


# =============================================================================
# LOSS FUNCTION
# =============================================================================


class CrossEntropyLoss:
    """
    Cross-Entropy Loss for multi-class classification.
    
    Mathematical Definition:
        L = -1/N * Σ Σ y_ij * log(p_ij)
        
        Where:
            N = batch size
            y_ij = 1 if sample i belongs to class j, else 0 (one-hot)
            p_ij = predicted probability that sample i belongs to class j
    
    For a single sample with true class k:
        L = -log(p_k)
        
        Intuition: We want p_k (probability of correct class) to be high.
                  When p_k → 1, loss → 0
                  When p_k → 0, loss → infinity
    
    Combined Softmax + Cross-Entropy Gradient:
        The gradient of cross-entropy loss with respect to the LOGITS (pre-softmax)
        has a beautifully simple form:
        
            dL/dz = softmax(z) - y = predictions - targets
        
        This is why we pass through the gradient in Softmax.backward()
        and compute the actual gradient here.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize cross-entropy loss.
        
        Args:
            epsilon: Small value to prevent log(0) which gives -infinity
        """
        self.epsilon = epsilon
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Predicted probabilities from Softmax
                        Shape: (batch_size, num_classes)
                        Each row sums to 1
            
            targets: One-hot encoded true labels
                    Shape: (batch_size, num_classes)
                    Each row has exactly one 1 and rest 0s
        
        Returns:
            loss: Scalar loss value (averaged over batch)
        
        Example:
            predictions = [[0.7, 0.2, 0.1],   # sample 1: predicts class 0
                          [0.1, 0.8, 0.1]]   # sample 2: predicts class 1
            
            targets = [[1, 0, 0],   # sample 1: true class is 0
                      [0, 1, 0]]   # sample 2: true class is 1
            
            loss = -1/2 * (log(0.7) + log(0.8))
                 = -1/2 * (-0.357 + -0.223)
                 = 0.29
        """
        batch_size = predictions.shape[0]
        
        # Add epsilon to prevent log(0) = -infinity
        # This is crucial for numerical stability
        predictions_clipped = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        # Cross-entropy: -sum(targets * log(predictions))
        # Only the log of the correct class contributes (where target = 1)
        loss = -np.sum(targets * np.log(predictions_clipped)) / batch_size
        
        return loss
    
    def compute_gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss with respect to predictions.
        
        This is the gradient that starts backpropagation.
        
        Args:
            predictions: Predicted probabilities from Softmax
                        Shape: (batch_size, num_classes)
            
            targets: One-hot encoded true labels
                    Shape: (batch_size, num_classes)
        
        Returns:
            gradient: Shape (batch_size, num_classes)
        
        Why is the gradient simply (predictions - targets)?
        ====================================================
        
        This comes from combining Softmax and Cross-Entropy:
        
        Let z = logits (pre-softmax), p = softmax(z), y = targets
        
        1. Softmax:      p_i = exp(z_i) / Σexp(z_j)
        2. Cross-Entropy: L = -Σ y_i * log(p_i)
        
        Taking the derivative dL/dz_i involves chain rule through both:
        
            dL/dz_i = Σ_j (dL/dp_j) * (dp_j/dz_i)
        
        After the math (see derivation below), this simplifies to:
        
            dL/dz_i = p_i - y_i
        
        Or in vector form:
            dL/dz = predictions - targets
        
        Derivation (for the curious):
        -----------------------------
        The Jacobian of softmax is complex:
            dp_i/dz_j = p_i(1 - p_i)    if i = j
            dp_i/dz_j = -p_i * p_j      if i ≠ j
        
        The gradient of cross-entropy w.r.t. softmax outputs:
            dL/dp_i = -y_i / p_i
        
        When you multiply these together and simplify:
            dL/dz_i = p_i - y_i
        
        This elegant result is why Softmax + Cross-Entropy is so popular!
        """
        # The combined gradient is simply: predictions - targets
        # No division by batch_size here - that's handled in Layer.backward()
        return predictions - targets
    
    def __repr__(self):
        return f"CrossEntropyLoss(epsilon={self.epsilon})"


class NeuralNetwork:
    """
    A fully-connected neural network for multi-class classification.
    
    Architecture:
        Input → [Layer → ReLU] × (n-1) → Layer → Softmax → Output
        
    The network chains together:
        - Dense layers (linear transformations)
        - ReLU activations (for hidden layers)
        - Softmax activation (for output layer)
    
    Example:
        layer_sizes = [784, 128, 64, 10]
        Creates: Layer(784→128) → ReLU → Layer(128→64) → ReLU → Layer(64→10) → Softmax
    """
    
    def __init__(self, layer_sizes: list):
        """
        Initialize the neural network with specified architecture.
        
        Args:
            layer_sizes: List of layer sizes, e.g., [784, 128, 64, 10]
                        - First element: input size (784 for MNIST)
                        - Last element: output size (10 for MNIST digits)
                        - Middle elements: hidden layer sizes
        
        Example:
            network = NeuralNetwork([784, 128, 64, 10])
            # Creates 3 layers:
            #   Layer 0: 784 → 128 (+ ReLU)
            #   Layer 1: 128 → 64  (+ ReLU)
            #   Layer 2: 64 → 10   (+ Softmax)
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # Number of weight layers
        
        # Create layers and activations
        self.layers = []
        self.activations = []
        
        for i in range(self.num_layers):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            
            # Create dense layer
            self.layers.append(Layer(input_size, output_size))
            
            # Create activation
            if i < self.num_layers - 1:
                # Hidden layers: use ReLU
                self.activations.append(ReLU())
            else:
                # Output layer: use Softmax for classification
                self.activations.append(Softmax())
        
        print(f"Created Neural Network:")
        print(f"  Architecture: {' → '.join(map(str, layer_sizes))}")
        print(f"  Layers: {self.num_layers}")
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            print(f"    [{i}] {layer} → {activation}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: propagate input through all layers.
        
        Flow: X → Layer1 → ReLU → Layer2 → ReLU → ... → LayerN → Softmax → Output
        
        Args:
            X: Input data of shape (batch_size, input_size)
               For MNIST: (batch_size, 784)
        
        Returns:
            predictions: Output probabilities of shape (batch_size, num_classes)
                        For MNIST: (batch_size, 10)
        
        Note:
            Each layer and activation stores its input/output for backward pass.
        """
        # Start with input
        current = X
        
        # Pass through each layer and activation
        for layer, activation in zip(self.layers, self.activations):
            # Linear transformation: z = X · W + b
            z = layer.forward(current)
            
            # Activation: a = activation(z)
            current = activation.forward(z)
        
        # Return final output (probabilities from softmax)
        return current
    
    def backward(self, grad_output: np.ndarray) -> None:
        """
        Backward pass: propagate gradients through all layers (backpropagation).
        
        Flow: grad → Softmax → LayerN → ReLU → ... → Layer2 → ReLU → Layer1
        
        This is the reverse of forward pass. Each component:
            1. Receives gradient from the layer/activation above
            2. Computes its own gradients (for weights if it's a layer)
            3. Passes gradient to the layer/activation below
        
        Args:
            grad_output: Gradient of loss with respect to network output
                        Shape: (batch_size, num_classes)
                        For softmax + cross-entropy: this is (predictions - targets)
        
        Note:
            Gradients for weights/biases are stored in each Layer object.
            Call update_weights() after this to apply the gradients.
        """
        # Start with gradient from loss function
        grad = grad_output
        
        # Propagate backward through layers (in reverse order)
        for i in range(self.num_layers - 1, -1, -1):
            # Backward through activation
            grad = self.activations[i].backward(grad)
            
            # Backward through layer (computes and stores grad_weights, grad_biases)
            grad = self.layers[i].backward(grad)
    
    def update_weights(self, learning_rate: float) -> None:
        """
        Update all weights in the network using gradient descent.
        
        Args:
            learning_rate: Step size for gradient descent (e.g., 0.01, 0.001)
        
        Note:
            Must call backward() first to compute gradients.
        """
        for layer in self.layers:
            layer.update_weights(learning_rate)
    
    def __repr__(self):
        return f"NeuralNetwork(architecture={self.layer_sizes})"
