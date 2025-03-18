import numpy as np
from scipy import datasets
import matplotlib.pyplot as plt

tableau = np.array([1, 2, 3, 4, 5])

zeros = np.zeros(5)
uns = np.ones(5)

sequence = np.arange(0, 10, 2)
matrice = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
print("Premiers 3 éléments:", tableau[:3])
print("Forme de la matrice:", matrice.shape)

matrice_aleatoire = np.random.randint(1, 100, size=(3, 4))
print("\nMatrice aléatoire (3x4):")
print(matrice_aleatoire)


def my_full_of_1_arr(lines, columns):
    array_of_ones = np.ones((lines, columns))
    print("\nArray of ones:")
    print(array_of_ones)
    print("\nShape:", array_of_ones.shape)


def super_square(lines, cols):
    matrix = np.zeros((lines, cols))
    
    matrix[0, :] = 1
    matrix[-1, :] = 1
    
    matrix[:, 0] = 1
    matrix[:, -1] = 1
    
    print("\nBorder matrix:")
    print(matrix)
    print("\nShape:", matrix.shape)

def my_zoom():
    face = datasets.face(gray=True)
    height, width = face.shape
    

    center_h = height // 2
    center_w = width // 2
    zoom_size_h = height // 4
    zoom_size_w = width // 4
    
    zoomed_face = face[center_h-zoom_size_h:center_h+zoom_size_h, 
                      center_w-zoom_size_w:center_w+zoom_size_w]
    

    saturated_face = zoomed_face.copy()
    saturated_face[saturated_face < 127] = 0 
    saturated_face[saturated_face >= 127] = 255
    
    # Display both images side by side
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.title('Zoomed Image')
    plt.imshow(zoomed_face, cmap=plt.cm.gray)
    
    plt.subplot(1, 2, 2)
    plt.title('Saturated Image')
    plt.imshow(saturated_face, cmap=plt.cm.gray)
    
    plt.show()

#my_zoom()
super_square(5, 5)
my_full_of_1_arr(5, 4)


def initialisation(lines, cols):
    random_matrix = np.random.randn(lines, cols)
    
    ones_column = np.ones((lines, 1))
    
    result = np.concatenate((random_matrix, ones_column), axis=1)
    
    print("\nInitialized matrix with ones column:")
    print(result)
    print("\nShape:", result.shape)
    
    return result

initialisation(5, 4)

def normalisation(mat):
    matrix = np.array(mat, dtype=float)

    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    normalized_matrix = (matrix - means) / stds
    
    print("\nOriginal matrix:")
    print(matrix)
    print("\nNormalized matrix:")
    print(normalized_matrix)
    print("\nVerification - means of columns:", np.mean(normalized_matrix, axis=0))
    print("Verification - std of columns:", np.std(normalized_matrix, axis=0))
    
    return normalized_matrix


test_matrix = initialisation(5, 4)
normalized_result = normalisation(test_matrix)

def plot_simple_line():
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 25, 30, 40]
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='red', linewidth=2)
    
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Simple Linear Graph')
    
    plt.grid(True)
    plt.show()

plot_simple_line()

def plot_double_line():
    x = [1, 2, 3, 4, 5]
    y1 = [10, 20, 25, 30, 40]
    y2 = [5, 15, 20, 25, 35]
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, color='red', linewidth=2, linestyle='-', label='First curve')
    plt.plot(x, y2, color='blue', linewidth=2, linestyle='--', label='Second curve')
    
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Double Line Graph')
    plt.grid(True)
    plt.legend()
    
    plt.show()

plot_double_line()

def random_dataset(n):
    data_dict = {}
    
    for i in range(n):
        random_data = np.random.normal(loc=0, scale=1, size=100)
        
        key = f"Experience {i+1}"
        data_dict[key] = random_data

    print("\nGenerated Dataset:")
    for key in data_dict:
        print(f"{key}: {len(data_dict[key])} values")
        print(f"Mean: {np.mean(data_dict[key]):.2f}")
        print(f"Std: {np.std(data_dict[key]):.2f}\n")
    
    return data_dict

dataset = random_dataset(4)

def visualize_datasets(data_dict):
    plt.figure(figsize=(12, 6))
    
    # Create box plots for all experiments
    plt.subplot(1, 2, 1)
    plt.boxplot(data_dict.values())
    plt.xticks(range(1, len(data_dict) + 1), data_dict.keys())
    plt.title('Distribution of Experiments')
    plt.grid(True)
    
    # Create histograms for all experiments
    plt.subplot(1, 2, 2)
    for key, data in data_dict.items():
        plt.hist(data, alpha=0.5, label=key, bins=20)
    plt.title('Histogram of All Experiments')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

dataset = random_dataset(4)
visualize_datasets(dataset)

def graphique(data_dict):
    n = len(data_dict)
    cols = 2  
    rows = (n + 1) // 2  

    plt.figure(figsize=(12, 4*rows))
    for i, (key, data) in enumerate(data_dict.items(), 1):
        plt.subplot(rows, cols, i)
        plt.plot(data, 'b-', linewidth=1)
        plt.hist(data, alpha=0.3, color='gray', bins=20)

        plt.title(key)
        plt.xlabel('Index/Frequency')
        plt.ylabel('Value')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

test_data = random_dataset(4)
graphique(test_data)

def plot_normal_distribution():
    data = np.random.normal(loc=0, scale=1, size=1000)
    
    plt.figure(figsize=(10, 6))

    plt.hist(data, bins=30, color='purple', alpha=0.7, 
             edgecolor='black', label='Random Samples')
    
    plt.xlabel('Values (μ=0, σ=1)')
    plt.ylabel('Frequency')
    plt.title('Normal Distribution Histogram\n(1000 samples)')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.show()

plot_normal_distribution()

