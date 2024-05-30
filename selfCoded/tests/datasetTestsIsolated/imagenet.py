import os
from multiprocessing import freeze_support

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



# Funktion zum Anzeigen einiger Informationen über die Daten
def show_batch(loader):
    for images, labels in loader:
        print(f'Batch Größe: {images.size()}')
        print(f'Label: {labels}')
        break  # Nur den ersten Batch anzeigen





if __name__ == '__main__':
    freeze_support()# Definiere die Transformationskette
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("loading dataset path")
    # Pfad zu den ImageNet-Daten
    data_dir = "D:\\imagenet"  # Ändere diesen Pfad entsprechend

    # Erstelle das ImageNet Dataset
    print("loading dataset")
    imagenet_dataset = datasets.ImageNet(root=data_dir, split='train', transform=transform,
                                         is_valid_file=lambda path: not os.path.basename(path).startswith("._"))

    # Erstelle den DataLoader
    print("loading dataloader")
    dataloader = DataLoader(imagenet_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Zeige einen Batch an
    show_batch(dataloader)
    print("show batch")
