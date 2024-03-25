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




