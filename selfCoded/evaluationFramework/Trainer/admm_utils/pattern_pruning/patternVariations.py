import torch
from itertools import combinations
import math
import os


def initialize_pattern_library(n, k):
    '''
    Berechnet Binomialkoeffizienten (n über k) und erstellt alle möglichen Kombinationen von Indizes in einer
    quadratischen matrix. n ist die Anzahl der Matrixelemente, k ist die Anzahl der Pattern Felder
    :param n:
    :param k:
    :return:
    '''
    count = math.comb(n, k)

    indices = list(range(n))
    combinations_of_indices = list(combinations(indices, k))

    _patterns = []
    for comb in combinations_of_indices:

        # create matrix with zeros
        matrix = torch.zeros((3, 3))

        # set amount of k ones
        for idx in comb:
            # flat indices to 2d indices
            row, col = divmod(idx, 3)
            matrix[row, col] = 1.0

        _patterns.append(matrix)

    # convert list to tensor of multiple tensors
    patterns = torch.stack(_patterns)

    return patterns


def count_unique_tensors(tensor_list):
    unique_tensors = []
    for tensor in tensor_list:
        if not any(torch.all(torch.eq(tensor, unique_tensor)) for unique_tensor in unique_tensors):
            unique_tensors.append(tensor)
    return len(unique_tensors)


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
