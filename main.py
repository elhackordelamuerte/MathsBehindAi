from data_preparation import load_and_prepare_data
from data_preprocess import preprocess_data
from model_architecture import create_model
from model_training import train_and_evaluate
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Load and prepare the sign language dataset
        X_train, y_train, X_test, y_test = load_and_prepare_data()
        
        print("\nDataset loaded successfully!")
        print(f"Number of training samples: {len(X_train)}")
        print(f"Number of test samples: {len(X_test)}")
        
        # Preprocess the data
        train_generator, validation_generator = preprocess_data(X_train, y_train, X_test, y_test)
        
        print("\nData preprocessing completed!")
        print(f"Training generator batches: {len(train_generator)}")
        print(f"Validation generator batches: {len(validation_generator)}")
        
        # Create and compile the model
        model = create_model()
        
        print("\nModel created successfully!")
        print("\nStarting model training...")
        
        # Train and evaluate the model
        history, test_loss, test_accuracy = train_and_evaluate(
            model, 
            train_generator, 
            validation_generator,
            epochs=10
        )
        
        # Plot training history
        plot_training_history(history)
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())