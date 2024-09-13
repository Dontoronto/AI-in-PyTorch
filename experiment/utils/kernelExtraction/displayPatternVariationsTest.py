from collections import defaultdict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import modelWrapper
import kernelDrawer
import kernelDrawerNegative
from torchvision.models import resnet18, ResNet18_Weights


def get_conv_layer_names(model):
    conv_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
            conv_layer_names.append(name)
    return conv_layer_names


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)  # Input channels = 1 (grayscale image), Output channels = 6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)  # Input channels = 6, Output channels = 16
        self.fc1 = nn.Linear(16 * 7 * 7, 120)  # Fully connected layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Output 10 classes for MNIST

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)



# Function to extract non-zero positions from convolutional layers
def extract_non_zero_positions(layer):
    #non_zero_positions = []
    explored_set = set()
    with torch.no_grad():
        i = 0
        j = 0
        for kernel in layer.weight:
            kernel_positions = []
            i +=1
            for channel in kernel:
                j +=1
                intmed_step = channel.flatten() != 0
                positions = intmed_step.cpu().nonzero(as_tuple=True)[0].numpy()
                explored_set.add(tuple(positions))
                #explored_set = {tuple(row) for row in explored}
                #kernel_positions.append(positions)
            #non_zero_positions.append(kernel_positions)
    print(f"filters: {i}")
    print(f"kernels: {j}")
    return explored_set

def get_nonzero_tuples(model_wrapped):

    layer_names = get_conv_layer_names(model_wrapped)
    nonzero_tuples = set()

    for name in layer_names:
        layer = dict(model_wrapped.named_modules())[name]
        if isinstance(layer, nn.Conv2d):
            non_zero_positions = extract_non_zero_positions(layer)
            nonzero_tuples |= non_zero_positions

    nonzero_tuples = tuple(nonzero_tuples)
    print(f"Unique Tuples identified: \n{nonzero_tuples}")
    return nonzero_tuples #tuple(nonzero_tuples)

def find_matching_pattern(non_zero_positions, patterns_set):
    """
    Find the index of the pattern in the set that matches the given non-zero positions.
    """
    #patterns_list = list(patterns_set)
    for i, pattern in enumerate(patterns_set):
        if non_zero_positions == pattern:
            return i
    return None

def save_negative_positions(model, patterns_set):
    """
    Iterate over layers, match kernel patterns, and save positions of negative values in a dictionary.
    """
    layer_names = get_conv_layer_names(model)
    negative_positions_dict = {i: [] for i in range(len(patterns_set))}

    for name in layer_names:
        layer = dict(model.named_modules())[name]
        if isinstance(layer, nn.Conv2d):
            weight = layer.weight.data.cpu()
            for c in range(weight.shape[0]):
                for k in range(weight.shape[1]):
                    kernel = weight[c, k]
                    #non_zero_positions = set(tuple(idx) for idx in np.argwhere(kernel != 0))
                    intmed_step = kernel.flatten() != 0
                    positions = intmed_step.cpu().nonzero(as_tuple=True)[0].numpy()
                    pattern_index = find_matching_pattern(tuple(positions), patterns_set)
                    if pattern_index is not None:
                        #negative_positions = [tuple(idx) for idx in np.argwhere(kernel < 0)]
                        negative_step = kernel.flatten() < 0
                        positions = negative_step.cpu().nonzero(as_tuple=True)[0].numpy()
                        negative_positions_dict[pattern_index].append(tuple(positions))

    return negative_positions_dict

def get_negative_distribution(negative_positions):
    neg_counts = {key: defaultdict(float) for key in negative_positions.keys()}

    # Iterate over each key and its list of tuples
    for key, tuples_list in negative_positions.items():
        # Iterate over each tuple in the list
        number_pattern = len(tuples_list)
        for tpl in tuples_list:
            # Count each element in the tuple
            for element in tpl:
                neg_counts[key][element] += (1/number_pattern)

    # Print the resulting counts dictionary
    for key, count_dict in neg_counts.items():
        print(f"Pattern {key}:")
        for element, count in count_dict.items():
            print(f"  Negative Weight at pos. {element} occurs {count:.2f}%")
            count_dict[element] = round(count,2)

    return neg_counts

# Load the model (adjust the path and model definition as necessary)
# Model, Loss, Optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ========================  Note: LeNet Pattern Distribution

# model = LeNet()#.to(device=device)
# model_wrapped = modelWrapper.ModelWrapper(model)
# model_wrapped.load_state_dict(torch.load("models/LeNet_elog_adv.pth", map_location=device))
# # model_wrapped.load_state_dict(torch.load("elog_adv/LeNet_admm_retrain.pth", map_location=device))
# # model_wrapped.load_state_dict(torch.load("elog_adv/LeNet.pth", map_location=device))
# model_wrapped.to(device)
#
# nonzero_tuples = get_nonzero_tuples(model_wrapped)
# neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
# counts = get_negative_distribution(neg_positions)
#
# combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
# combined.save("LeNet_elog_adv_kernel_sparsity.png")
#
# combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
# combined_neg.save("LeNet_elog_adv_kernel_sparsity_distribution.png")
#
# # -------------------
#
# model = LeNet()#.to(device=device)
# model_wrapped = modelWrapper.ModelWrapper(model)
# model_wrapped.load_state_dict(torch.load("models/LeNet_elog_default.pth", map_location=device))
# # model_wrapped.load_state_dict(torch.load("elog_adv/LeNet_admm_retrain.pth", map_location=device))
# # model_wrapped.load_state_dict(torch.load("elog_adv/LeNet.pth", map_location=device))
# model_wrapped.to(device)
#
# nonzero_tuples = get_nonzero_tuples(model_wrapped)
# neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
# counts = get_negative_distribution(neg_positions)
#
# combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
# combined.save("LeNet_elog_default_kernel_sparsity.png")
#
# combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
# combined_neg.save("LeNet_elog_default_kernel_sparsity_distribution.png")
#
# # -------------------
#
# model = LeNet()#.to(device=device)
# model_wrapped = modelWrapper.ModelWrapper(model)
# model_wrapped.load_state_dict(torch.load("models/LeNet_all_adv.pth", map_location=device))
# # model_wrapped.load_state_dict(torch.load("elog_adv/LeNet_admm_retrain.pth", map_location=device))
# # model_wrapped.load_state_dict(torch.load("elog_adv/LeNet.pth", map_location=device))
# model_wrapped.to(device)
#
# nonzero_tuples = get_nonzero_tuples(model_wrapped)
# neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
# counts = get_negative_distribution(neg_positions)
#
# combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
# combined.save("LeNet_trivial_adv_kernel_sparsity.png")
#
# combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
# combined_neg.save("LeNet_trivial_adv_kernel_sparsity_distribution.png")
#
# # -------------------
#
# model = LeNet()#.to(device=device)
# model_wrapped = modelWrapper.ModelWrapper(model)
# model_wrapped.load_state_dict(torch.load("models/LeNet_all_default.pth", map_location=device))
# # model_wrapped.load_state_dict(torch.load("elog_adv/LeNet_admm_retrain.pth", map_location=device))
# # model_wrapped.load_state_dict(torch.load("elog_adv/LeNet.pth", map_location=device))
# model_wrapped.to(device)
#
# nonzero_tuples = get_nonzero_tuples(model_wrapped)
# neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
# counts = get_negative_distribution(neg_positions)
#
# combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
# combined.save("LeNet_trivial_default_kernel_sparsity.png")
#
# combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
# combined_neg.save("LeNet_trivial_default_kernel_sparsity_distribution.png")

# -------------------

model = LeNet()#.to(device=device)
model_wrapped = modelWrapper.ModelWrapper(model)
model_wrapped.load_state_dict(torch.load("conn_lenet/Trivial Default Conn.pth", map_location=device))
# model_wrapped.load_state_dict(torch.load("elog_adv/LeNet_admm_retrain.pth", map_location=device))
# model_wrapped.load_state_dict(torch.load("elog_adv/LeNet.pth", map_location=device))
model_wrapped.to(device)

nonzero_tuples = get_nonzero_tuples(model_wrapped)
neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
counts = get_negative_distribution(neg_positions)

combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
combined.save("LeNet_trivial_default_conn_kernel_sparsity.png")

combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
combined_neg.save("LeNet_trivial_default_conn_kernel_sparsity_distribution.png")


# -------------------

model = LeNet()#.to(device=device)
model_wrapped = modelWrapper.ModelWrapper(model)
model_wrapped.load_state_dict(torch.load("conn_lenet/Trivial Adv Conn.pth", map_location=device))
# model_wrapped.load_state_dict(torch.load("elog_adv/LeNet_admm_retrain.pth", map_location=device))
# model_wrapped.load_state_dict(torch.load("elog_adv/LeNet.pth", map_location=device))
model_wrapped.to(device)

nonzero_tuples = get_nonzero_tuples(model_wrapped)
neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
counts = get_negative_distribution(neg_positions)

combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
combined.save("LeNet_trivial_adv_conn_kernel_sparsity.png")

combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
combined_neg.save("LeNet_trivial_adv_conn_kernel_sparsity_distribution.png")

# -------------------

model = LeNet()#.to(device=device)
model_wrapped = modelWrapper.ModelWrapper(model)
model_wrapped.load_state_dict(torch.load("conn_lenet/SCP Default Conn.pth", map_location=device))
# model_wrapped.load_state_dict(torch.load("elog_adv/LeNet_admm_retrain.pth", map_location=device))
# model_wrapped.load_state_dict(torch.load("elog_adv/LeNet.pth", map_location=device))
model_wrapped.to(device)

nonzero_tuples = get_nonzero_tuples(model_wrapped)
neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
counts = get_negative_distribution(neg_positions)

combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
combined.save("LeNet_elog_default_conn_kernel_sparsity.png")

combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
combined_neg.save("LeNet_elog_default_conn_kernel_sparsity_distribution.png")

# -------------------

model = LeNet()#.to(device=device)
model_wrapped = modelWrapper.ModelWrapper(model)
model_wrapped.load_state_dict(torch.load("conn_lenet/SCP Adv Conn.pth", map_location=device))
# model_wrapped.load_state_dict(torch.load("elog_adv/LeNet_admm_retrain.pth", map_location=device))
# model_wrapped.load_state_dict(torch.load("elog_adv/LeNet.pth", map_location=device))
model_wrapped.to(device)

nonzero_tuples = get_nonzero_tuples(model_wrapped)
neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
counts = get_negative_distribution(neg_positions)

combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
combined.save("LeNet_elog_adv_conn_kernel_sparsity.png")

combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
combined_neg.save("LeNet_elog_adv_conn_kernel_sparsity_distribution.png")

# ======================= Note: ResNet18 Pattern Distribution

# model = resnet18()
# model_wrapped = modelWrapper.ModelWrapper(model)
# model_wrapped.load_state_dict(torch.load("models/ResNet_elog_adv.pth", map_location=device))
# model_wrapped.to(device)
#
# nonzero_tuples = get_nonzero_tuples(model_wrapped)
# neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
# counts = get_negative_distribution(neg_positions)
#
# combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
# combined.save("ResNet18_elog_adv_kernel_sparsity.png")
#
# combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
# combined_neg.save("ResNet18_elog_adv_kernel_sparsity_distribution.png")
#
# # -------------------
#
# model = resnet18()
# model_wrapped = modelWrapper.ModelWrapper(model)
# model_wrapped.load_state_dict(torch.load("models/ResNet_elog_default.pth", map_location=device))
# model_wrapped.to(device)
#
# nonzero_tuples = get_nonzero_tuples(model_wrapped)
# neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
# counts = get_negative_distribution(neg_positions)
#
# combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
# combined.save("ResNet18_elog_default_kernel_sparsity.png")
#
# combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
# combined_neg.save("ResNet18_elog_default_kernel_sparsity_distribution.png")
#
# # -------------------
#
# model = resnet18()
# model_wrapped = modelWrapper.ModelWrapper(model)
# model_wrapped.load_state_dict(torch.load("models/ResNet_all_default.pth", map_location=device))
# model_wrapped.to(device)
#
# nonzero_tuples = get_nonzero_tuples(model_wrapped)
# neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
# counts = get_negative_distribution(neg_positions)
#
# combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
# combined.save("ResNet18_trivial_default_kernel_sparsity.png")
#
# combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
# combined_neg.save("ResNet18_trivial_default_kernel_sparsity_distribution.png")
#
# # -------------------
#
# model = resnet18()
# model_wrapped = modelWrapper.ModelWrapper(model)
# model_wrapped.load_state_dict(torch.load("models/ResNet_all_adv.pth", map_location=device))
# model_wrapped.to(device)
#
# nonzero_tuples = get_nonzero_tuples(model_wrapped)
# neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
# counts = get_negative_distribution(neg_positions)
#
# combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
# combined.save("ResNet18_trivial_adv_kernel_sparsity.png")
#
# combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
# combined_neg.save("ResNet18_trivial_adv_kernel_sparsity_distribution.png")

# -------------------

model = resnet18()
model_wrapped = modelWrapper.ModelWrapper(model)
model_wrapped.load_state_dict(torch.load("conn_res/Trivial Default Conn.pth", map_location=device))
model_wrapped.to(device)

nonzero_tuples = get_nonzero_tuples(model_wrapped)
neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
counts = get_negative_distribution(neg_positions)

combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
combined.save("ResNet18_trivial_default_conn_kernel_sparsity.png")

combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
combined_neg.save("ResNet18_trivial_default_conn_kernel_sparsity_distribution.png")

# -------------------

model = resnet18()
model_wrapped = modelWrapper.ModelWrapper(model)
model_wrapped.load_state_dict(torch.load("conn_res/Trivial Adv Conn.pth", map_location=device))
model_wrapped.to(device)

nonzero_tuples = get_nonzero_tuples(model_wrapped)
neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
counts = get_negative_distribution(neg_positions)

combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
combined.save("ResNet18_trivial_adv_conn_kernel_sparsity.png")

combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
combined_neg.save("ResNet18_trivial_adv_conn_kernel_sparsity_distribution.png")

# -------------------

model = resnet18()
model_wrapped = modelWrapper.ModelWrapper(model)
model_wrapped.load_state_dict(torch.load("conn_res/SCP Default Conn.pth", map_location=device))
model_wrapped.to(device)

nonzero_tuples = get_nonzero_tuples(model_wrapped)
neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
counts = get_negative_distribution(neg_positions)

combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
combined.save("ResNet18_elog_default_conn_kernel_sparsity.png")

combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
combined_neg.save("ResNet18_elog_default_conn_kernel_sparsity_distribution.png")

# -------------------

model = resnet18()
model_wrapped = modelWrapper.ModelWrapper(model)
model_wrapped.load_state_dict(torch.load("conn_res/SCP Adv Conn.pth", map_location=device))
model_wrapped.to(device)

nonzero_tuples = get_nonzero_tuples(model_wrapped)
neg_positions = save_negative_positions(model_wrapped, nonzero_tuples)
counts = get_negative_distribution(neg_positions)

combined = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,25)
combined.save("ResNet18_elog_adv_conn_kernel_sparsity.png")

combined_neg = kernelDrawerNegative.combine_images_with_positions(nonzero_tuples,counts,4,25)
combined_neg.save("ResNet18_elog_adv_conn_kernel_sparsity_distribution.png")





# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # _weights = ResNet18_Weights.IMAGENET1K_V1
# # _model = resnet18(_weights)
# _model = resnet18()
# Model = modelWrapper.ModelWrapper(_model)
# Model.load_state_dict(torch.load("ResNet_admm_retrain.pth", map_location=device))
#
# nonzero_tuples = get_nonzero_tuples(Model)
# print(nonzero_tuples)
#
#
# combined = combined_image = kernelDrawer.combine_images_with_positions(nonzero_tuples,4,5)
# combined.save("kernel_sparsity.png")





