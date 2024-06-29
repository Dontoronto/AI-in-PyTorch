import torch
from itertools import combinations
import math


def initialize_pattern_library(n, k):
    '''
    Berechnet Binomialkoeffizienten (n über k) und erstellt alle möglichen Kombinationen von Indizes in einer
    quadratischen matrix. n ist die Anzahl der Matrixelemente, k ist die Anzahl der Pattern Felder
    :param n: quantity of n elements selectable
    :param k: fixed amount of selected marked values
    :return: a list of unique combinations according to the binomialcoefficient
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

def initialize_elog_based_patterns():
    '''
    Generiert die vier spezifischen Sparse Convolution Patterns (SCPs) und gibt sie als Tensor zurück.
    :return: ein Tensor, der die vier 3x3 SCP-Muster enthält
    '''
    # Definieren der vier SCP-Muster
    scp1 = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0]
    ]

    scp2 = [
        [0, 1, 0],
        [1, 1, 0],
        [0, 1, 0]
    ]

    scp3 = [
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]

    scp4 = [
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 0]
    ]

    # Konvertiere die Muster in Torch-Tensoren
    scp1_tensor = torch.tensor(scp1, dtype=torch.float32)
    scp2_tensor = torch.tensor(scp2, dtype=torch.float32)
    scp3_tensor = torch.tensor(scp3, dtype=torch.float32)
    scp4_tensor = torch.tensor(scp4, dtype=torch.float32)

    # Erstellen einer Liste der Tensoren
    # _patterns = [scp1_tensor, scp2_tensor, scp3_tensor, scp4_tensor]
    _patterns = (scp1_tensor, scp2_tensor, scp3_tensor, scp4_tensor)

    # Konvertieren der Liste in einen Tensor
    patterns = torch.stack(_patterns)

    return patterns




