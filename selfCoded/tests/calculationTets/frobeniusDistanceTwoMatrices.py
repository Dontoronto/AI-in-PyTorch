import torch

# Define two matrices
matrix1 = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]])
matrix2 = torch.tensor([[9.0, 8.0, 7.0],
                        [6.0, 5.0, 4.0],
                        [3.0, 2.0, 1.0]])

# Calculate the difference between the matrices
difference = matrix1 - matrix2

# Calculate the Frobenius norm of the difference, i.e., the Frobenius distance
frobenius_distance = torch.norm(difference, p='fro')

print(f"Frobenius Distance between the two matrices: {frobenius_distance}")

#%%
