import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
# from sewar.full_ref import fsim
import torchvision.transforms as T
from torchvision.transforms import ToPILImage

import lenet
import modelWrapper

def mnist_transformer() -> T.Compose:
    transformator = T.Compose([
        T.ToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=[0.1307], std=[0.3081])
    ])

    return transformator

def preprocessBackwardsBatched(batch):
    # TODO: evtl. müssen wir nicht image tensoren sondern auch batch tensoren zurück umwandeln. Hier
    # TODO: testen und evtl. anpassen damit automatisch erkannt wird was gefordert ist
    tensors = batch.clone().detach()
    image_list = list()
    for tensor in tensors:
        meanBack = torch.tensor([0.1307]).view(-1, 1, 1)
        stdBack = torch.tensor([0.3081]).view(-1, 1, 1)
        tensor = tensor * stdBack + meanBack
        tensorBack = torch.clamp(tensor, 0, 1)
        image_list.append(tensorBack)
    return torch.stack(image_list)
class DenoisingLeNet(nn.Module):
    def __init__(self, pretrained_lenet):
        super(DenoisingLeNet, self).__init__()
        self.pretrained_lenet = pretrained_lenet
        self.encoder = nn.Sequential(
            pretrained_lenet.conv1,
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Output: 14x14
            pretrained_lenet.conv2,
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # Output: 7x7
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 28x28
            nn.Tanh()
        )
        self.freeze_encoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

class DenoisingLeNetDefault(nn.Module):
    def __init__(self):
        super(DenoisingLeNetDefault, self).__init__()
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
    mse_val = nn.functional.mse_loss(img1, img2).item()
    if mse_val == 0:
        return 100
    pixel_max = 1.0
    return 20 * np.log10(pixel_max / np.sqrt(mse_val))

def calculate_ssim(img1, img2, win_size=7):
    img1 = img1.permute(0, 2, 3, 1).cpu().squeeze(3).detach().numpy()
    img2 = img2.permute(0, 2, 3, 1).cpu().squeeze(3).detach().numpy()
    ssim_val = 0.0
    for i in range(img1.shape[0]):
        ssim_val += ssim(img1[i], img2[i], multichannel=False, win_size=win_size, data_range=1.0)
    return ssim_val / img1.shape[0]

def calculate_mse(img1, img2):
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()
    return mse(img1, img2)

def calculate_rmse(img1, img2):
    return np.sqrt(calculate_mse(img1, img2))


def calculate_snr(img1, img2):
    signal = torch.sum(img1 ** 2)
    noise = torch.sum((img1 - img2) ** 2)
    return 10 * torch.log10(signal / noise).item()


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
transform = mnist_transformer()

# Laden des MNIST-Datensatzes
train_dataset = datasets.MNIST(root="/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/dataset/mnist/",
                               train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.MNIST(root="/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/dataset/mnist/",
                              train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# =========

# Initialisieren der vortrainierten LeNet und der erweiterten Klasse
pretrained_lenet = lenet.LeNet()

#torch.save(pretrained_lenet.state_dict(), 'model_parameters.pth')



pretrained_lenet.load_state_dict(torch.load('model_parameters.pth'))

denoising_model_pretrained = DenoisingLeNet(pretrained_lenet)
denoising_model_pretrained.load_state_dict(torch.load('modified_model_parameter.pth'))
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, denoising_model.parameters()), lr=1e-3)


# ========

denoising_model = DenoisingLeNetDefault()



# Modell, Verlustfunktion und Optimierer
criterion = nn.MSELoss()
optimizer = optim.Adam(denoising_model.parameters(), lr=1e-3)

# Training
num_epochs = 35

# ============================== Note: just retraining pretrained Model


# for epoch in range(num_epochs):
#     denoising_model.train()
#     denoising_model.freeze_encoder()
#     running_loss = 0.0
#     for inputs, _ in train_loader:
#
#         optimizer.zero_grad()
#         outputs = denoising_model(inputs)
#         loss = criterion(outputs, inputs)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * inputs.size(0)
#
#     epoch_loss = running_loss / len(train_loader.dataset)
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
#
# torch.save(denoising_model.state_dict(), 'modified_model_parameter.pth')
#
# exit()

# ==============================


# Note: for testing backwards preprocessing
for epoch in range(num_epochs):
    denoising_model.train()
    running_loss = 0.0
    for inputs, _ in train_loader:
        # Hinzufügen von künstlichem Rauschen zu den Eingabebildern
        noisy_inputs = inputs + 0.5 * torch.randn_like(inputs)
        noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

        optimizer.zero_grad()
        outputs = denoising_model(noisy_inputs)
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
# def evaluate(model, dataloader, epo):
#     model.eval()
#     total_psnr = 0.0
#     total_ssim = 0.0
#     with torch.no_grad():
#         for inputs, _ in dataloader:
#             noisy_inputs = inputs + 0.5 * torch.randn_like(inputs)
#             noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)
#
#             outputs = model(noisy_inputs)
#             # Berechnen von PSNR und SSIM
#             psnr = calculate_psnr(outputs, inputs)
#             ssim_val = calculate_ssim(outputs, inputs)
#             total_psnr += psnr * inputs.size(0)
#             total_ssim += ssim_val * inputs.size(0)
#
#
#     inp = preprocessBackwardsBatched(inputs[:5])
#     nois = preprocessBackwardsBatched(noisy_inputs[:5])
#     out = preprocessBackwardsBatched(outputs[:5])
#
#     save_images(inp, nois, out, epo)
#
#     avg_psnr = total_psnr / len(dataloader.dataset)
#     avg_ssim = total_ssim / len(dataloader.dataset)
#     return avg_psnr, avg_ssim

# Evaluierung
def evaluate_extended(model, dataloader, epo):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    total_rmse = 0.0
    #total_fsim = 0.0
    total_snr = 0.0
    with torch.no_grad():
        for inputs, _ in dataloader:
            noisy_inputs = inputs + 0.5 * torch.randn_like(inputs)
            noisy_inputs = torch.clamp(noisy_inputs, 0., 1.)

            outputs = model(noisy_inputs)
            # Berechnen von Metriken
            psnr = calculate_psnr(outputs, inputs)
            ssim_val = calculate_ssim(outputs, inputs)
            mse_val = calculate_mse(outputs, inputs)
            rmse_val = calculate_rmse(outputs, inputs)
            #fsim_val = calculate_fsim(outputs, inputs)
            snr_val = calculate_snr(outputs, inputs)

            total_psnr += psnr * inputs.size(0)
            total_ssim += ssim_val * inputs.size(0)
            total_mse += mse_val * inputs.size(0)
            total_rmse += rmse_val * inputs.size(0)
            #total_fsim += fsim_val * inputs.size(0)
            total_snr += snr_val * inputs.size(0)

    inp = preprocessBackwardsBatched(inputs[:5])
    nois = preprocessBackwardsBatched(noisy_inputs[:5])
    out = preprocessBackwardsBatched(outputs[:5])

    save_images(inp, nois, out, epo)

    avg_psnr = total_psnr / len(dataloader.dataset)
    avg_ssim = total_ssim / len(dataloader.dataset)
    avg_mse = total_mse / len(dataloader.dataset)
    avg_rmse = total_rmse / len(dataloader.dataset)
    #avg_fsim = total_fsim / len(dataloader.dataset)
    avg_snr = total_snr / len(dataloader.dataset)

    return avg_psnr, avg_ssim, avg_mse, avg_rmse, avg_snr


# avg_psnr, avg_ssim = evaluate(denoising_model, test_loader, 88)
# print(f'Default: PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')
# avg_psnr, avg_ssim = evaluate(denoising_model_pretrained, test_loader, 99)
# print(f'Retrained: PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')

avg_psnr, avg_ssim, avg_mse, avg_rmse, avg_snr = evaluate_extended(denoising_model, test_loader, 88)
print(f'Default: PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}, SNR: {avg_snr:.4f}')

avg_psnr, avg_ssim, avg_mse, avg_rmse, avg_snr = evaluate_extended(denoising_model_pretrained, test_loader, 99)
print(f'Retrained: PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}, SNR: {avg_snr:.4f}')
