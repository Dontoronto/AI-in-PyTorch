# Inspirationen für die Umsetzung von ADMM

## Based on Systematic Weight Pruning Paper

```
import torch
import torch.nn as nn
import torch.optim as optim

# Assuming `model` is your PyTorch model and `dataloader` is your dataset loader

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ADMM hyperparameters
rho = 1e-3 # Penalty parameter, adjust based on your problem
num_iterations = 10 # Number of ADMM iterations, adjust based on convergence

# Initialization
Z = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
U = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

for iteration in range(num_iterations):
    # Step 1: Update weights W using SGD while considering U and Z
    for epoch in range(num_epochs): # Number of epochs for the SGD step
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Add the ADMM regularization term to the loss
            admm_reg_term = sum((param - Z[name] + U[name]).pow(2).sum() for name, param in model.named_parameters())
            total_loss = loss + (rho / 2) * admm_reg_term
            total_loss.backward()
            optimizer.step()
    
    # Step 2: Update Z (proximal operator step, i.e., weight pruning)
    with torch.no_grad():
        for name, param in model.named_parameters():
            # This is a simplified representation. Implement your own projection function based on your sparsity constraints.
            Z[name] = project_to_sparsity_constraints(param + U[name], sparsity_level)
    
    # Step 3: Update dual variables U
    with torch.no_grad():
        for name, param in model.named_parameters():
            U[name] += param - Z[name]

# Final step: Retrain pruned model to restore accuracy if necessary
```
Der ADMM Loss wird hier über jeden Batch auf den normalen Loss drauf gerechnet.
Der Algorithmus durchläuft den ganzen Datensatz und dann die festgelegte Anzahl
an Epochen. Nachdem die Epochen abgelaufen sind aktualisiert er die Z- und U- 
Variablen. Diese Abfolge durchläuft er so oft bis die Anzahl an ADMM-Iterationen
abgelaufen sind.

## Based on pconv-appendix paper

```
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 100)  # Beispiel für MNIST
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def update_z(u, w, rho):
    # Beispiel für eine einfache Update-Regel für z (Sparsity-Förderung)
    # Dies könnte eine spezifischere Logik enthalten, die auf deine Anforderungen zugeschnitten ist
    return torch.sign(w) * torch.max(torch.abs(w) - u / rho, torch.tensor(0.0))

def admm_train(model, loss_fn, data_loader, rho, lambda_l1, iterations):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    u = {name: torch.zeros_like(param, requires_grad=False) for name, param in model.named_parameters()}
    z = {name: torch.zeros_like(param, requires_grad=False) for name, param in model.named_parameters()}

    for it in range(iterations):
        # Schritt 1: Optimiere W bezüglich der ursprünglichen Verlustfunktion plus einem ADMM-Term
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            admm_loss = sum(torch.norm(param - z[name] + u[name], p=2) for name, param in model.named_parameters())
            total_loss = loss + rho/2 * admm_loss
            total_loss.backward()
            optimizer.step()

        # Schritt 2: Update z unter Berücksichtigung der Sparsity-Beschränkungen
        with torch.no_grad():
            for name, param in model.named_parameters():
                z[name] = update_z(u[name], param, rho)

        # Schritt 3: Update Dualvariablen u
        with torch.no_grad():
            for name, param in model.named_parameters():
                u[name] += param - z[name]

# Beispielverwendung
model = SimpleNN()
loss_fn = nn.CrossEntropyLoss()
# Angenommen, data_loader ist definiert und liefert Trainingsdaten
# admm_train(model, loss_fn, data_loader, rho=1.0, lambda_l1=1e-5, iterations=10)

```