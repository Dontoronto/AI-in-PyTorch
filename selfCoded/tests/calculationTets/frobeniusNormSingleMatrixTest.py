import torch

# Define a matrix
matrix = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]])

# Calculate the Frobenius norm
frobenius_norm = torch.norm(matrix, p='fro')

print(f"Frobenius Norm of the matrix: {frobenius_norm}")

#%%
