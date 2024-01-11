
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import h5py
import pandas, numpy, random
import matplotlib.pyplot as plt

# sourcecode: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/deepfool.html#DeepFool

batch_size = 10

test_tensor = torch.tensor([True] * batch_size)

print("Here is a test for multiplying a tensor value inside of a tensor")
print(test_tensor)
#%%


#### Here we test some slicing methods

import numpy as np

test_array = torch.randn((1,10))
print("This is the auto-generated array of random numbers")
print(test_array)
#%%
slicing_array = []

for i in range(len(test_array)):
    _ = test_array[i:i+1].clone().detach()
    if i == 0:
        print("here is the single sliced object of array")
        print(_)
    slicing_array.append(_)

# Here the difference is that we have single tensors inside of a list
print("Here comes the sliced array")
print(slicing_array)


# Testing the difference between [i:i+1] and [i]
test_array = torch.tensor([1, 2, 3, 4, 5])  # Example tensor

# Output for the first code snippet
for i in range(len(test_array)):
    result = test_array[i:i+1].clone().detach()
    print(f"Output for index {i}: {result} of type{type(result)}")

# Output for the second code snippet
for i in range(len(test_array)):
    result = test_array[i].clone().detach()
    print(f"Output for index {i}: {result} of type{type(result)}")
#%%
