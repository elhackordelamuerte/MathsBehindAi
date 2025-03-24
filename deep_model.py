import numpy as np
from gethouse import get_houses
import matplotlib.pyplot as plt

class DeepNeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.random.randn(layer_sizes[i+1]))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:  # Linear activation for output layer
                activations.append(z)
            else:  # ReLU activation for hidden layers
                activations.append(self.relu(z))
        return activations
    
    def backward_propagation(self, X, y, activations, learning_rate):
        m = X.shape[0]
        deltas = []
        
        # Output layer error
        delta = activations[-1] - y
        deltas.append(delta)
        
        # Hidden layers error
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(activations[i])
            deltas.insert(0, delta)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(activations[i].T, deltas[i]) / m
            self.biases[i] -= learning_rate * np.mean(deltas[i], axis=0)

def train_deep_model():
    # Get and prepare data
    houses = get_houses()
    
    # Extract and normalize features
    sizes = np.array([house.size for house in houses])
    rooms = np.array([house.num_rooms for house in houses])
    locations = np.array([house.location_quality for house in houses])
    prices = np.array([house.price for house in houses])
    
    # Normalize data
    X = np.column_stack((
        (sizes - np.mean(sizes)) / np.std(sizes),
        (rooms - np.mean(rooms)) / np.std(rooms),
        (locations - np.mean(locations)) / np.std(locations)
    ))
    y = ((prices - np.mean(prices)) / np.std(prices)).reshape(-1, 1)
    
    # Create model with multiple layers [3 -> 64 -> 32 -> 16 -> 1]
    model = DeepNeuralNetwork([3, 64, 32, 16, 1])
    
    # Training parameters
    epochs = 1000
    learning_rate = 0.001
    losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        activations = model.forward_propagation(X)
        prediction = activations[-1]
        
        # Calculate loss
        loss = np.mean((prediction - y) ** 2)
        losses.append(loss)
        
        # Backward pass
        model.backward_propagation(X, y, activations, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot predictions vs actual
    plt.subplot(1, 2, 2)
    plt.scatter(y, prediction, alpha=0.5)
    plt.plot([-3, 3], [-3, 3], 'r--')
    plt.title('Predicted vs Actual Prices')
    plt.xlabel('Actual Price (Normalized)')
    plt.ylabel('Predicted Price (Normalized)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_deep_model()