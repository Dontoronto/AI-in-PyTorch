import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats


def gaussian_filter(kernel_size=3, sigma=1):
    # Erzeugen eines Gauß-Filters
    interval = (2 * sigma + 1.) / kernel_size
    x = np.linspace(-sigma - interval / 2., sigma + interval / 2., kernel_size + 1)
    kern1d = np.diff(stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return torch.from_numpy(kernel).clone().detach().float()

def laplacian_of_gaussian(kernel_size=3, sigma=1):
    # Erzeugen eines Laplacian-of-Gaussian Filters
    gaussian = gaussian_filter(kernel_size, sigma)
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    log = F.conv2d(gaussian.unsqueeze(0).unsqueeze(0), laplacian.unsqueeze(0).unsqueeze(0), padding=1)
    return log.squeeze()

def enhanced_laplacian_of_gaussian(kernel_size=3, sigma=1, n_interpolations=8, p=0.75):
    # Erzeugen eines Enhanced Laplacian-of-Gaussian Filters
    log = laplacian_of_gaussian(kernel_size, sigma)
    elog = log
    for _ in range(n_interpolations - 1):
        elog += log * p
    elog /= n_interpolations
    return elog

def initialize_pattern_library(K):
    patterns = []

    # Teilen Sie K auf die verschiedenen Filtertypen auf
    num_gaussian = num_log = num_elog = K // 3

    # Gauß-Filter erstellen
    for _ in range(num_gaussian):
        patterns.append(gaussian_filter())

    # LoG-Filter erstellen
    for _ in range(num_log):
        patterns.append(laplacian_of_gaussian())

    # ELoG-Filter erstellen
    for _ in range(num_elog):
        patterns.append(enhanced_laplacian_of_gaussian())

    # Stapeln Sie alle Muster in einem Tensor
    patterns = torch.stack(patterns)

    return patterns

# Initialisieren der Musterbibliothek mit K = 126
K = 126
patterns = initialize_pattern_library(K)
print(torch.unique(patterns))
print(patterns)