import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Load a pretrained model
model = models.resnet18(pretrained=True).eval()



def hook_fn(module, input, output):
    feature_maps.append(output)

# Function to hook feature maps
feature_maps = []
# Register hooks to all convolutional layers
hooks = []
for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        hooks.append(layer.register_forward_hook(hook_fn))

# Preprocess the input image
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# # Load an example image
# img_path = '/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/selfCoded/DeepFoolPreTrained_v2/testImages/adv_desk.jpeg'
# img = Image.open(img_path).convert('RGB')
# input_tensor = preprocess(img).unsqueeze(0)  # Create a mini-batch as expected by the model
input_tensor = torch.randn(32,3,244,244)
print(input_tensor)


# Forward pass to extract feature maps
with torch.no_grad():
    res = model(input_tensor)

# Plot feature maps
def plot_feature_maps(feature_maps, layer_names, num_columns=8):
    for fmap, layer_name in zip(feature_maps, layer_names):
        num_kernels = fmap.shape[1]
        num_rows = num_kernels // num_columns + (num_kernels % num_columns > 0)
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 2))
        fig.suptitle(f'Feature maps of layer: {layer_name}')
        for i in range(num_kernels):
            ax = axes[i // num_columns, i % num_columns]
            ax.imshow(fmap[0, i].cpu().numpy(), cmap='viridis')
            ax.axis('off')
        plt.show()
        plt.close(fig)
        return

# Get names of convolutional layers
layer_names = [name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Conv2d)]

# Visualize the feature maps
plot_feature_maps(feature_maps, layer_names)

# Remove hooks
for hook in hooks:
    hook.remove()

#%%
