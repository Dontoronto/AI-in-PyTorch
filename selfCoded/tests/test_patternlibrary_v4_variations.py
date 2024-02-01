import torch
from itertools import combinations
import math
import os

def initialize_pattern_library(n, k):
    # Berechnung des Binomialkoeffizienten (n über k)
    count = math.comb(n, k)

    # Erstellen Sie alle möglichen Kombinationen von Indizes in einer 3x3-Matrix
    indices = list(range(n))
    combinations_of_indices = list(combinations(indices, k))

    # Erstellen Sie Tensoren für jede Kombination
    _patterns = []
    for comb in combinations_of_indices:
        # Erstellen Sie eine 3x3-Matrix mit Nullen
        matrix = torch.zeros((3, 3))

        # Setzen Sie `k` Einsen an den kombinierten Positionen
        for idx in comb:
            # Umwandeln des flachen Index in zweidimensionale Indizes
            row, col = divmod(idx, 3)
            matrix[row, col] = 1.0

        _patterns.append(matrix)

    # Konvertiere die Liste von Matrizen in einen einzigen Tensor
    patterns = torch.stack(_patterns)

    return patterns

patterns = initialize_pattern_library(9,4)
print(patterns.shape)
def count_unique_tensors(tensor_list):
    unique_tensors = []
    for tensor in tensor_list:
        if not any(torch.all(torch.eq(tensor, unique_tensor)) for unique_tensor in unique_tensors):
            unique_tensors.append(tensor)
    return len(unique_tensors)

# Anzahl der einzigartigen Tensoren in patterns
num_unique_tensors = count_unique_tensors(patterns)
print("Anzahl der einzigartigen Tensoren:", num_unique_tensors)

file_path = "tensor_data.txt"

def save_tensors_to_files(tensor, folder='output'):
    """
    Saves each 3x3 tensor in the given 126x3x3 tensor to individual text files.
    :param tensor: A 126x3x3 PyTorch tensor.
    :param folder: The folder where text files will be saved.
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    file_name = f"{folder}/tensor.txt"
    with open(file_name, 'w') as file:
        #Iterate and save each 3x3 tensor
        for i, matrix in enumerate(tensor):
            for row in matrix:
                file.write(' '.join(map(str, row.tolist())) + '\n')
            file.write('\n')

save_tensors_to_files(patterns)

print(f"All tensors data saved to {file_path}")