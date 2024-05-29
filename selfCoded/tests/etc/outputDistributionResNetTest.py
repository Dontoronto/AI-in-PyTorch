import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Load a pretrained model
model = models.resnet18(pretrained=True).eval()

# Preprocess the input image
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = '/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/selfCoded/DeepFoolPreTrained_v2/testImages/adv_desk.jpeg'
img = Image.open(img_path).convert('RGB')
input_tensor = preprocess(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)

# Normalize the outputs to sum to 1 using softmax
output = torch.nn.functional.softmax(output, dim=1)
model_outputs = output.squeeze()  # Remove the batch dimension

print(model_outputs.max())

# Sort the model outputs in descending order
sorted_outputs = torch.sort(model_outputs, descending=True).values

# Select the highest 5% of the outputs
num_top_outputs = int(len(sorted_outputs) * 0.05)
top_outputs = sorted_outputs[:num_top_outputs]

# Sort the top outputs in ascending order
sorted_top_outputs = torch.sort(top_outputs).values

# Reorder the top outputs to mimic a normal distribution
n = len(sorted_top_outputs)
ordered_outputs = torch.zeros(n)

# Place the largest values in the middle
left, right = 0, n - 1
for i in range(n):
    if i % 2 == 0:
        ordered_outputs[right] = sorted_top_outputs[i]
        right -= 1
    else:
        ordered_outputs[left] = sorted_top_outputs[i]
        left += 1

# Plot the reordered top 5% outputs
x_values = np.arange(n)
plt.figure(figsize=(10, 6))
plt.plot(x_values, ordered_outputs.numpy(), label='Ordered Top 5% Model Outputs')
plt.xlabel('Output Index')
plt.ylabel('Probability Value')
plt.title('Probability Distribution of Top 5% ResNet18 Model Outputs for a Single Image')
plt.legend()
plt.show()




# import numpy as np
# import torch
# import torchvision.models as models
# import torchvision.transforms as T
# from PIL import Image
# import matplotlib.pyplot as plt
#
# # Load a pretrained model
# model = models.resnet18(pretrained=True).eval()
#
# # Preprocess the input image
# preprocess = T.Compose([
#     T.Resize(256),
#     T.CenterCrop(224),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# img_path = '/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/selfCoded/DeepFoolPreTrained_v2/testImages/adv_desk.jpeg'
# img = Image.open(img_path).convert('RGB')
# input_tensor = preprocess(img).unsqueeze(0)
#
#
# with torch.no_grad():
#     output = model(input_tensor)
#
#     # Step 3: Normalize the outputs to sum to 1
# output = torch.nn.functional.softmax(output, dim=1)
# model_outputs = output.squeeze()  # Remove the batch dimension
#
# # Step 2: Sort the model outputs
# sorted_outputs = torch.sort(model_outputs).values
#
# # Step 3: Reorder the sorted outputs to mimic a normal distribution
# n = len(sorted_outputs)
# ordered_outputs = torch.zeros(n)
#
# # Place the largest values in the middle
# left, right = 0, n - 1
# for i in range(n):
#     if i % 2 == 0:
#         ordered_outputs[right] = sorted_outputs[i]
#         right -= 1
#     else:
#         ordered_outputs[left] = sorted_outputs[i]
#         left += 1
#
# # Step 4: Plot the reordered outputs
# x_values = np.arange(n)
# plt.figure(figsize=(10, 6))
# plt.plot(x_values, ordered_outputs.numpy(), label='Ordered Model Outputs')
# plt.xlabel('Output Index')
# plt.ylabel('Probability Value')
# plt.title('Probability Distribution of ResNet18 Model Outputs for a Single Image')
# plt.legend()
# plt.show()
# #%%

#%%
