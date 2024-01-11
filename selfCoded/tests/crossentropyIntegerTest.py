import torch
import torch.nn as nn

# Define a simple neural network with a single output neuron
class SingleOutputNet(nn.Module):
    def __init__(self):
        super(SingleOutputNet, self).__init__()
        self.fc = nn.Linear(2, 1)  # Input size: 2, Output size: 1 (single output neuron)

    def forward(self, x):
        x = self.fc(x)
        return x

# Input tensor
input_tensor = torch.tensor([0.3, 0.4])

# Target label (output) - Use a float between 0 and 1 for the single output neuron
target_label = torch.tensor([0.75])  # Example: Single output neuron value (0.0 to 1.0)

# Instantiate the network
model = SingleOutputNet()

# Define the loss function for single output neuron
criterion = nn.BCEWithLogitsLoss()

# Forward pass
output = model(input_tensor)  # Unsqueeze to add batch dimension

# Calculate the loss
loss = criterion(output, target_label)
print(loss)
