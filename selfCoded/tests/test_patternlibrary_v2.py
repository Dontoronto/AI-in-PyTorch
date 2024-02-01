import os

import torch
import numpy as np
from scipy import stats
import torch.nn.functional as F

def set_top_four_to_one(tensor):
    # Flatten the tensor and get the indices of the top 4 values
    flat_tensor = tensor.flatten()
    _, indices = torch.topk(flat_tensor, 4)

    # Create a mask with ones at the top indices and zeros elsewhere
    mask = torch.zeros_like(flat_tensor)
    mask[indices] = 1

    # Reshape the mask to the original tensor shape
    return mask.reshape(tensor.shape)
def gaussian_filter(sigma):
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    interval = (2 * sigma + 1.) / kernel_size
    x = np.linspace(-sigma - interval / 2., sigma + interval / 2., kernel_size + 1)
    kern1d = np.diff(stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()

    # Zuschneiden des Kernels auf 3x3, falls erforderlich
    if kernel.shape[0] > 3:
        center = kernel.shape[0] // 2
        kernel = kernel[center - 1:center + 2, center - 1:center + 2]

    return torch.from_numpy(kernel).float()

def laplacian_of_gaussian(sigma):
    gaussian = gaussian_filter(sigma)
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    log = F.conv2d(gaussian.unsqueeze(0).unsqueeze(0), laplacian.unsqueeze(0).unsqueeze(0), padding=1)
    return log.squeeze()

def enhanced_laplacian_of_gaussian(sigma, n_interpolations=8, p=0.75):
    log = laplacian_of_gaussian(sigma)
    elog = log
    for _ in range(n_interpolations - 1):
        elog += log * p
    elog /= n_interpolations
    return elog

def initialize_pattern_library(K):
    patterns = []
    sigmas = [0.5, 1.0, 1.5]  # Reduzierte Liste von Standardabweichungen

    # Erstellen von Filtern für jede Standardabweichung
    for sigma in sigmas:
        # patterns.append(gaussian_filter(sigma))
        # patterns.append(laplacian_of_gaussian(sigma))
        # patterns.append(enhanced_laplacian_of_gaussian(sigma))
        patterns.append(set_top_four_to_one(gaussian_filter(sigma)))
        patterns.append(set_top_four_to_one(laplacian_of_gaussian(sigma)))
        patterns.append(set_top_four_to_one(enhanced_laplacian_of_gaussian(sigma)))

    # Füllen Sie die restlichen Muster auf, um K zu erreichen
    while len(patterns) < K:
        random_pattern = torch.rand(3, 3)
        patterns.append(set_top_four_to_one(random_pattern))

    # Stapeln Sie alle Muster in einem Tensor
    patterns = torch.stack(patterns[:K])

    return patterns

# Initialisieren der Musterbibliothek mit K = 126
K = 126
patterns = initialize_pattern_library(K)
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

#np.savetxt('my_file.txt', patterns.numpy())

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

