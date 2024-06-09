import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DepthwiseLeNet(nn.Module):
    def __init__(self):
        super(DepthwiseLeNet, self).__init__()
        self.conv1 = DepthwiseSeparableConv(1, 6, kernel_size=3, padding=1)  # Input channels = 1 (grayscale image), Output channels = 6
        self.conv2 = DepthwiseSeparableConv(6, 16, kernel_size=3, padding=1)  # Input channels = 6, Output channels = 16
        self.fc1 = nn.Linear(16 * 7 * 7, 120)  # Fully connected layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Output 10 classes for MNIST

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Data loading and transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/dataset/mnist/", train=True, transform=transform)
test_dataset = datasets.MNIST(root="/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/dataset/mnist/", train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Model, loss function, optimizer
model = DepthwiseLeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Testing loop
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)

print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')

# Display weights of the convolutional layers
def display_weights(conv_layer):
    depthwise_weight = conv_layer.depthwise.weight.data.cpu().numpy()
    pointwise_weight = conv_layer.pointwise.weight.data.cpu().numpy()

    print(f"Depthwise weights shape: {depthwise_weight.shape}")
    print(depthwise_weight)

    print(f"Pointwise weights shape: {pointwise_weight.shape}")
    print(pointwise_weight)

print("\nWeights of conv1 layer:")
display_weights(model.conv1)

print("\nWeights of conv2 layer:")
display_weights(model.conv2)
