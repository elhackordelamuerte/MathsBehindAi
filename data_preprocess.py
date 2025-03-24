import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Exercice 2: Prétraitement des Données
# Objectif: Apprendre à utiliser des techniques de prétraitement pour améliorer la qualité des données d'entrée.

def preprocess_data(X_train, y_train, X_test, y_test):
    # Convert labels to one-hot encoding
    num_classes = len(set(y_train))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Configure data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create training data generator
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=32
    )

    # Configure validation/test data generator (no augmentation)
    test_datagen = ImageDataGenerator()

    # Create validation data generator
    validation_generator = test_datagen.flow(
        X_test, y_test,
        batch_size=32
    )

    return train_generator, validation_generator