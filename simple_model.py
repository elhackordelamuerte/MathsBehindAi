import numpy as np
from gethouse import get_houses
import matplotlib.pyplot as plt

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def simple_neural_network_multi():
    # Get house data
    houses = get_houses()
    
    # Extract multiple features and price data
    sizes = np.array([house.size for house in houses])
    rooms = np.array([house.num_rooms for house in houses])
    locations = np.array([house.location_quality for house in houses])
    prices = np.array([house.price for house in houses])
    
    # Normalize all features
    X_size = normalize_data(sizes)
    X_rooms = normalize_data(rooms)
    X_location = normalize_data(locations)
    
    # Combine features into input matrix
    X = np.column_stack((X_size, X_rooms, X_location))
    y = normalize_data(prices).reshape(-1, 1)
    
    # Initialize weights and bias
    W = np.random.randn(3, 1)  # 3 weights for 3 features
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
    print(f"Size: {feature_importance[0]:.4f}")
    print(f"Rooms: {feature_importance[1]:.4f}")
    print(f"Location: {feature_importance[2]:.4f}")
    
    # Calculate and display predictions for sample houses
    print("\nSample Predictions:")
    for i in range(5):  # Show first 5 houses
        actual_price = prices[i]
        predicted_price = (prediction[i][0] * (np.max(prices) - np.min(prices))) + np.min(prices)
        
        print(f"\nHouse {i+1}:")
        print(f"Features:")
        print(f"  - Size: {sizes[i]:.2f}")
        print(f"  - Rooms: {rooms[i]}")
        print(f"  - Location Quality: {locations[i]}")
        print(f"Actual Price: {actual_price:.2f}")
        print(f"Predicted Price: {predicted_price:.2f}")
        print(f"Difference: {abs(actual_price - predicted_price):.2f}")
    
    # Calculate and display model performance metrics
    mse = np.mean((prediction - y) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(prediction - y))
    
    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    
    # Visualize training results
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot predicted vs actual prices
    plt.subplot(1, 2, 2)
    plt.scatter(y, prediction, color='blue', alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
    plt.title('Predicted vs Actual Prices')
    plt.xlabel('Actual Price (Normalized)')
    plt.ylabel('Predicted Price (Normalized)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simple_neural_network_multi()