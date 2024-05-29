import math
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))

def calculate_ssim(img1, img2, win_size=7):
    img1 = img1.permute(0, 2, 3, 1).cpu().squeeze(3).numpy()
    img2 = img2.permute(0, 2, 3, 1).cpu().squeeze(3).numpy()
    ssim_val = 0.0
    for i in range(img1.shape[0]):
        ssim_val += ssim(img1[i], img2[i], multichannel=True, win_size=3, data_range=1.0)
    return ssim_val / img1.shape[0]

# Überarbeiteter Training- und Evaluierungscode

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DenoisingLeNet(nn.Module):
    def __init__(self):
        super(DenoisingLeNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Datenvorbereitung
transform = transforms.Compose([
    transforms.ToTensor()
])
# Laden des MNIST-Datensatzes
train_dataset = datasets.MNIST(root="/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/dataset/mnist/",
                               train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.MNIST(root="/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/dataset/mnist/",
                              train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modell, Verlustfunktion und Optimierer
model = DenoisingLeNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, _ in train_loader:
        # Hinzufügen von künstlichem Rauschen zu den Eingabebildern
        noisy_inputs = inputs + 0.5 * torch.randn_like(inputs)
        noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

        optimizer.zero_grad()
        outputs = model(noisy_inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Evaluierung
def evaluate(model, dataloader):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for inputs, _ in dataloader:
            noisy_inputs = inputs + 0.5 * torch.randn_like(inputs)
            noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

            outputs = model(noisy_inputs)
            # Berechnen von PSNR und SSIM
            psnr = calculate_psnr(outputs, inputs)
            ssim_val = calculate_ssim(outputs, inputs, win_size=7)
            total_psnr += psnr * inputs.size(0)
            total_ssim += ssim_val * inputs.size(0)

    avg_psnr = total_psnr / len(dataloader.dataset)
    avg_ssim = total_ssim / len(dataloader.dataset)
    return avg_psnr, avg_ssim

avg_psnr, avg_ssim = evaluate(model, test_loader)
print(f'PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')