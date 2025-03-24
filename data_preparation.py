import numpy as np
import kagglehub
import csv  # Ajouté pour la lecture des fichiers CSV

# Exercice 1: Introduction et Préparation des Données
# Objectif: Comprendre comment charger et préparer les données pour l'entraînement d'un modèle.

def load_and_prepare_data():
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Download dataset
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")

    # Read training data
    with open(f"{path}/sign_mnist_train.csv", 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            y_train.append(int(row[0]))  # First column is label
            X_train.append([float(pixel) for pixel in row[1:]])  # Rest are pixels

    # Read test data
    with open(f"{path}/sign_mnist_test.csv", 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            y_test.append(int(row[0]))  # First column is label
            X_test.append([float(pixel) for pixel in row[1:]])  # Rest are pixels

    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Reshape images to 28x28
    X_train = X_train.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)

    # Normalize pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print("Data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test