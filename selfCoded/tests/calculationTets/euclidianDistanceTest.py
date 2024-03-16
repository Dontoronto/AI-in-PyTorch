import torch

# Example tensors of the same shape
tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([4.0, 5.0, 6.0])

# Step 1: Calculate the difference between the tensors
difference = tensor1 - tensor2

# Step 2: Square the differences
squared_difference = difference ** 2

# Step 3: Sum the squared differences
sum_squared_difference = torch.sum(squared_difference)

# Step 4: Take the square root of the sum to get the Euclidean distance
euclidean_distance = torch.sqrt(sum_squared_difference)

print(f"Euclidean Distance: {euclidean_distance}")
#%%
