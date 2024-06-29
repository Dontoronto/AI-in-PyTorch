import numpy as np
import torch
import torchvision.transforms as T

# Example NumPy array (3 channels, height=224, width=224)
# original_numpy_array = np.random.rand(224, 224, 3).astype(np.float32)
original_numpy_array = np.zeros((244,244,3)).astype(np.float32)

# Normalize parameters
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Transform: NumPy array to Tensor and normalize
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

# Convert NumPy array (HWC) to Tensor (CHW) and normalize
tensor = transform(original_numpy_array)

# Reverse normalization
for t, m, s in zip(tensor, mean, std):
    t.mul_(s).add_(m)

# Convert Tensor back to NumPy array
reconstructed_numpy_array = tensor.numpy()

# Transpose from CHW to HWC
reconstructed_numpy_array = np.transpose(reconstructed_numpy_array, (1, 2, 0))

# Calculate the difference
difference = original_numpy_array - reconstructed_numpy_array

# Print out the results
print("Original NumPy Array:")
print(np.abs(original_numpy_array))
print("\nReconstructed NumPy Array:")
print(np.abs(reconstructed_numpy_array))
print("\nDifference Array:")
print(difference)
print("\nMaximum difference:")
print(np.max(np.abs(difference)))

# L1 norm
l1_norm = np.sum(np.abs(difference))

# L2 norm
l2_norm = np.sqrt(np.sum(np.square(difference)))

# L-infinity norm
l_inf_norm = np.max(np.abs(difference))

# Print out the results
print("L1 norm:", l1_norm)
print("L2 norm:", l2_norm)
print("L-infinity norm:", l_inf_norm)

#%%
