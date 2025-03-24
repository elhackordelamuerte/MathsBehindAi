import tensorflow as tf
from model_architecture import create_model

# Exercice 4: Entraînement et Évaluation du Modèle
# Objectif: Entraîner le modèle et évaluer ses performances.

def train_and_evaluate(model, train_generator, validation_generator, epochs=10):
    # Training parameters
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(validation_generator)
    
    print(f"\nFinal Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    return history, test_loss, test_accuracy