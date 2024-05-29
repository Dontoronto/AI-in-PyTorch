import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.use('Agg')  # Use a non-interactive backend

def find_largest_divisor(n):
    for i in range(10, 1, -1):
        if n % i == 0:
            return i
    return 8


def extract_single_feature_map(model, batch, conv_name):

    def hook_fn(module, input, output):
        feature_map.append(output)

    # Function to hook feature maps
    feature_map = []
    # Register hooks to all convolutional layers
    hooks = []

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            if name == conv_name:
                hooks.append(layer.register_forward_hook(hook_fn))

    # Forward pass to extract features
    with torch.no_grad():
        model(batch)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return feature_map[0], conv_name


def extract_all_feature_maps(model, batch):

    def hook_fn(module, input, output):
        feature_maps.append(output)

    # List to save the layer names
    conv_names = []
    # Function to hook feature maps
    feature_maps = []
    # Register hooks to all convolutional layers
    hooks = []

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn))
            conv_names.append(name)


    # Forward pass to extract features
    with torch.no_grad():
        model(batch)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return feature_maps, conv_names


def plot_single_feature_map(feature_maps, layer_name):
    num_kernels = feature_maps.shape[1]
    num_columns = find_largest_divisor(num_kernels)
    num_rows = num_kernels // num_columns + (num_kernels % num_columns > 0)
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 2))
    fig.suptitle(f'Feature maps of layer: {layer_name}')
    for i in range(num_kernels):
        ax = axes[i // num_columns, i % num_columns]
        ax.imshow(feature_maps[0, i].cpu().numpy(), cmap='viridis')
        ax.axis('off')
    current_figure = plt.gcf()
    plt.show()
    plt.close(fig)

    return current_figure


def plot_all_feature_maps(feature_maps, layer_names):
    layer_figures = []
    for fmap, layer_name in zip(feature_maps, layer_names):
        fig = plot_single_feature_map(fmap, layer_name)
        layer_figures.append(fig)

    return layer_figures

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = '/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/selfCoded/DeepFoolPreTrained_v2/testImages/adv_desk.jpeg'
img = Image.open(img_path).convert('RGB')
input_tensor = preprocess(img).unsqueeze(0)

# input_tensor = torch.randn(1,3,224,224)

# ---------

model = models.resnet18(pretrained=True).eval()

# features, lnames = extract_all_feature_maps(model, input_tensor)
#
# figure_list = plot_all_feature_maps(features, lnames)
#
# for i, figure in enumerate(figure_list):
#     print(f"Saving the {i} image")
#     figure.savefig(f"testSaveSpot/{i}_img.png", dpi=400)
#     plt.close(figure)

# --------

features, lname = extract_single_feature_map(model, input_tensor, "conv1")

gcf_data = plot_single_feature_map(features, lname)

#print(gcf_data)

#gcf_data.savefig("test.png")
#plt.close(gcf_data)




#%%
