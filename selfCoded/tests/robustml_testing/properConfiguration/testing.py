import robustml
import torch
from torchvision import transforms
import numpy as np


dataset = robustml.provider.CIFAR10("/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/dataset/cifar-10/cifar-10-batches-py/test_batch")
# '/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/dataset/cifar-10'

x,y = dataset[10]

print(x.shape)
print(type(y))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

x_tensor = transform(x)

print(x)
print(x_tensor)
print(x_tensor.shape)
#%%
