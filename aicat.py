import numpy as np
from getcat import get_cats
import matplotlib.pyplot as plt

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def simple_cat_network():
    # Get cat data
    cats = get_cats()
    
    # Extract features and price data
    lengths = np.array([cat.length for cat in cats])
    races = np.array([cat.race for cat in cats])
    prices = np.array([cat.price for cat in cats])
    
    # Normalize all data
    X_length = normalize_data(lengths)
    X_race = normalize_data(races)
    
    # Combine features into input matrix
    X = np.column_stack((X_length, X_race))
    y = normalize_data(prices).reshape(-1, 1)
    
    # Initialize weights and bias
    W = np.random.randn(2, 1)  # 2 weights for 2 features
    b = np.random.randn(1)
    
    # Training parameters
    learning_rate = 0.01
    epochs = 1000
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        prediction = np.dot(X, W) + b
        
        # Calculate loss (MSE)
        loss = np.mean((prediction - y) ** 2)
        losses.append(loss)
        
        # Backward pass (gradient descent)
        dW = np.dot(X.T, (prediction - y)) / len(X)
        db = np.mean(prediction - y)
        
        # Update weights and bias
        W -= learning_rate * dW
        b -= learning_rate * db
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # Print feature importance
    feature_importance = np.abs(W.flatten())
    print("\nFeature Importance:")
    print(f"Length: {feature_importance[0]:.4f}")
    print(f"Race: {feature_importance[1]:.4f}")
    
    # Display sample predictions
    print("\nSample Predictions:")
    for i in range(5):
        actual_price = prices[i]
        predicted_price = (prediction[i][0] * (np.max(prices) - np.min(prices))) + np.min(prices)
        
        print(f"\nCat {i+1}:")
        print(f"Length: {lengths[i]:.1f} cm")
        print(f"Race Quality: {races[i]} (1-5)")
        print(f"Actual Price: ${actual_price:.2f}")
        print(f"Predicted Price: ${predicted_price:.2f}")
        print(f"Difference: ${abs(actual_price - predicted_price):.2f}")
    
    # Save generated cats to file
    print("\nSaving cats data to file...")
    with open('c:\\Users\\cmoi\\Documents\\GitHub\\MathsBehindAi\\cats_data.txt', 'w') as f:
        f.write("Length,Race,Price\n")
        for cat in cats:
            f.write(f"{cat.length},{cat.race},{cat.price}\n")
    print("Data saved to cats_data.txt")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot length vs price
    plt.subplot(1, 3, 2)
    plt.scatter(lengths, prices, c=races, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Race Quality')
    plt.title('Length vs Price (colored by Race)')
    plt.xlabel('Length (cm)')
    plt.ylabel('Price ($)')
    
    # Plot predicted vs actual
    plt.subplot(1, 3, 3)
    plt.scatter(y, prediction, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('Predicted vs Actual Prices')
    plt.xlabel('Actual Price (Normalized)')
    plt.ylabel('Predicted Price (Normalized)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simple_cat_network()