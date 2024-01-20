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
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
# Set up a simple attack from torchattacks (e.g., PGD)
attack = torchattacks.PGD(model, eps=0.3, alpha=2/255, steps=40)
# Evaluate the model's robustness
def evaluate_model(model, testloader, attack):
    correct = 0
    total = 0
    for images, labels in testloader:
        # Apply the attack
        adv_images = attack(images, labels)
        # Forward pass
        outputs = model(adv_images)
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
accuracy = evaluate_model(model, testloader, attack)
print(f'Accuracy under attack: {accuracy}%')
#%%
