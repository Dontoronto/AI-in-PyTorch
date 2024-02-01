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

    if kernel.shape[0] > 3:
        center = kernel.shape[0] // 2
        kernel = kernel[center - 1:center + 2, center - 1:center + 2]

    # Set the top four values to one and the rest to zero
    return set_top_four_to_one(kernel)

# ... Analogous changes for laplacian_of_gaussian and enhanced_laplacian_of_gaussian ...

def initialize_pattern_library(K):
    patterns = []
    sigmas = [0.5, 1.0, 1.5]  # Reduced list of standard deviations

    for sigma in sigmas:
        patterns.append(gaussian_filter(sigma))
        patterns.append(laplacian_of_gaussian(sigma))
        patterns.append(enhanced_laplacian_of_gaussian(sigma))

    while len(patterns) < K:
        random_pattern = torch.rand(3, 3)
        patterns.append(set_top_four_to_one(random_pattern))

    patterns = torch.stack(patterns[:K])

    return patterns

K = 126
patterns = initialize_pattern_library(K)

print(patterns)
def count_unique_tensors(tensor_list):
    unique_tensors = []
    for tensor in tensor_list:
        if not any(torch.all(torch.eq(tensor, unique_tensor)) for unique_tensor in unique_tensors):
            unique_tensors.append(tensor)
    return len(unique_tensors)

# Anzahl der einzigartigen Tensoren in patterns
num_unique_tensors = count_unique_tensors(patterns)
print("Anzahl der einzigartigen Tensoren:", num_unique_tensors)
