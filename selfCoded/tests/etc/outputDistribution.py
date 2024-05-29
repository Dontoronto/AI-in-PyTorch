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

# Sort the model outputs in descending order
sorted_outputs = torch.sort(model_outputs, descending=True).values

# Select the highest 5% of the outputs
num_top_outputs = int(len(sorted_outputs) * 0.05)
top_outputs = sorted_outputs[:num_top_outputs]

# Normalize the top 5% outputs by dividing by the maximum value
max_value = top_outputs.max()
normalized_top_outputs = top_outputs / max_value

# Sort the normalized top outputs in ascending order
sorted_top_outputs = torch.sort(normalized_top_outputs).values

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
plt.ylabel('Normalized Probability Value')
plt.title('Probability Distribution of Top 5% ResNet18 Model Outputs for a Single Image')
plt.legend()
plt.show()





# import torch
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Step 1: Generate or load model outputs
# # For demonstration, we'll use a random tensor and normalize it to sum to 1
# model_outputs = torch.abs(torch.randn(1000))
# model_outputs = model_outputs / model_outputs.sum()
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
# plt.title('Probability Distribution of Model Outputs')
# plt.legend()
# plt.show()



# import torch
# import torch.distributions as dist
# import matplotlib.pyplot as plt
#
# # Step 1: Generate or load model outputs
# # For demonstration, we'll use a random tensor
# model_outputs = torch.randn(1000)
#
# # Step 2: Sort the outputs
# sorted_outputs = torch.sort(model_outputs).values
#
# # Step 3: Fit a probability distribution
# # Example: fitting a normal distribution
# mean = torch.mean(sorted_outputs)
# std = torch.std(sorted_outputs)
# normal_dist = dist.Normal(mean, std)
#
# # Generate values for the x-axis
# x_values = torch.linspace(sorted_outputs.min(), sorted_outputs.max(), 1000)
#
# # Calculate the probability density function (pdf)
# pdf_values = normal_dist.log_prob(x_values).exp()
#
# # Step 4: Plot the distribution
# plt.figure(figsize=(10, 6))
# plt.plot(x_values.numpy(), pdf_values.numpy(), label='Fitted Normal Distribution')
# plt.hist(sorted_outputs.numpy(), bins=30, density=True, alpha=0.6, label='Model Outputs (Histogram)')
# plt.xlabel('Model Output Values')
# plt.ylabel('Probability Density')
# plt.title('Probability Distribution of Model Outputs')
# plt.legend()
# plt.show()

#%%
