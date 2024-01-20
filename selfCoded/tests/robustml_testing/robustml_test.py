import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torchattacks
import robustml
# Load a pre-trained PyTorch model
model = resnet18(pretrained=True).eval()
# Prepare the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/dataset/cifar-10',
                                       train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

def evaluate_model(model, testloader):
    for images, labels in testloader:
        # Forward pass
        print(images.shape)
        outputs = model(images).squeeze(0)
        # Calculate accuracy
        predicted = outputs.argmax().item()
        #_, predicted = torch.max(outputs.data, 0)
        print(outputs)
        print(predicted)
        break

    return predicted

accuracy = evaluate_model(model, testloader)
#%%
