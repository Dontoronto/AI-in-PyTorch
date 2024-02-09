import torch

mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)

print(mean.shape)