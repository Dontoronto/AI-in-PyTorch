
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D as ax
import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Input layer with 2 neurons, hidden layer with 4 neurons
        self.relu = nn.Sigmoid()       # ReLU activation function
        self.fc2 = nn.Linear(2, 1)  # Output layer with 1 neuron

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
        #return torch.softmax(x)

# Define the AND function dataset
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
targets = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

# Instantiate the model, loss function, and optimizer
model = MLP()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification problems
optimizer = optim.SGD(model.parameters(), lr=0.0663)

# Training loop
epochs = 30000
for epoch in range(epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test the trained model
with torch.no_grad():
    test_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    predictions = model(test_inputs)
    print("\nTest Predictions:")
    for i in range(len(test_inputs)):
        print(f'Input: {test_inputs[i].tolist()}, Prediction: {predictions[i].item():.4f}')


plt.figure(figsize=(8, 6))

x_min, x_max = 0, 1
y_min, y_max = 0, 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
print(xx.shape)
print(yy)
grid_inputs = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
grid_inputs.requires_grad_(True)  # Enable gradient tracking for inputs
grid_outputs = model(grid_inputs)
np_outputs = grid_outputs.detach().numpy().reshape(xx.shape)


ax = plt.axes(projection='3d')

ax.plot_surface(xx, yy, np_outputs, cmap='viridis')
ax.set_title('MLP probability for AND Function')
ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Output probability')
ax.legend()

plt.show()


# Plot the decision boundary
plt.figure(figsize=(8, 6))
num_out_points = len(grid_outputs)
np_norm_derivatives = np.zeros(num_out_points)

#for out_point in range(num_out_points):
#    derivative = torch.autograd.grad(grid_outputs[out_point], grid_inputs,retain_graph=True)[0][out_point]
#    np_norm_derivatives[out_point] = torch.linalg.vector_norm(derivative,ord=2).detach().numpy()


derivative = torch.autograd.grad(grid_outputs,grid_inputs,retain_graph=True,grad_outputs=torch.ones_like(grid_outputs))[0]

norms = torch.linalg.vector_norm(derivative, ord=2, dim=1, keepdim=True).detach().numpy()


#normalized_tensor = derivative / norms
#print(normalized_tensor)


ax2 = plt.axes(projection='3d')

ax2.plot_surface(xx, yy, norms.reshape(xx.shape), cmap='viridis')
ax2.set_title('Decision Boundary of MLP for AND Function')
ax2.set_xlabel('Input 1')
ax2.set_ylabel('Input 2')
ax2.set_zlabel('Norm of the MLP gradient')
ax2.legend()

plt.show()



#%%
