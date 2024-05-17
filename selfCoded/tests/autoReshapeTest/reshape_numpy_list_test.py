# Example 2D list of shape (3, 10)
list_2d = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
           [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
           [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]

# Transpose the list to (10, 3)
reshaped_list = list(map(list, zip(*list_2d)))

print(reshaped_list)

import numpy as np

# Example 2D array of shape (3, 10)
array = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                  [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])

# Reshape the array to (10, 3)
reshaped_array = array.T

print(reshaped_array)

print("new test")
import numpy as np

def swap_first_two_dimensions(array):
    if array.ndim == 2:
        # For 2D array, swap the first two dimensions
        return array.transpose()
    elif array.ndim == 3:
        # For 3D array, swap the first two dimensions
        return array.transpose(1, 0, 2)
    else:
        raise ValueError("This function only supports 2D and 3D arrays")

# Example usage for a 2D array
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
reshaped_2d = swap_first_two_dimensions(array_2d)
print("Original 2D shape:", array_2d.shape)
print("Reshaped 2D shape:", reshaped_2d.shape)
print(reshaped_2d)

# Example usage for a 3D array
array_3d = np.random.rand(3, 10, 2)  # using random values for demonstration
reshaped_3d = swap_first_two_dimensions(array_3d)
print("Original 3D shape:", array_3d.shape)
print("Reshaped 3D shape:", reshaped_3d.shape)
print(reshaped_3d)

#%%
