import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

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

def calculate_psnr(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 20 * np.log10(pixel_max / np.sqrt(mse.item()))

def calculate_ssim(img1, img2, win_size=7):
    img1 = img1.permute(0, 2, 3, 1).cpu().squeeze(3).detach().numpy()
    img2 = img2.permute(0, 2, 3, 1).cpu().squeeze(3).detach().numpy()
    ssim_val = 0.0
    for i in range(img1.shape[0]):
        ssim_val += ssim(img1[i], img2[i], multichannel=False, win_size=win_size, data_range=1.0)
    return ssim_val / img1.shape[0]

def save_images(original, noisy, denoised, epoch):
    fig, axes = plt.subplots(3, 5, figsize=(12, 6))
    for i in range(5):
        axes[0, i].imshow(original[i].cpu().detach().numpy().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original {i+1}')

        axes[1, i].imshow(noisy[i].cpu().detach().numpy().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Noisy {i+1}')

        axes[2, i].imshow(denoised[i].cpu().detach().numpy().squeeze(), cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_title(f'Denoised {i+1}')

    plt.tight_layout()
    plt.savefig(f'denoising_epoch_{epoch}.png')
    plt.close(fig)

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
num_epochs = 35

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

    # Speichern von Bildern nach jedem Epoch
    if epoch % 5 == 0:
        save_images(inputs[:5], noisy_inputs[:5], outputs[:5], epoch)

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
            ssim_val = calculate_ssim(outputs, inputs)
            total_psnr += psnr * inputs.size(0)
            total_ssim += ssim_val * inputs.size(0)

    avg_psnr = total_psnr / len(dataloader.dataset)
    avg_ssim = total_ssim / len(dataloader.dataset)
    return avg_psnr, avg_ssim

avg_psnr, avg_ssim = evaluate(model, test_loader)
print(f'PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')
#%%
