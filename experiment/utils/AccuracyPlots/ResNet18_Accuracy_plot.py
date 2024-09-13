import matplotlib.pyplot as plt

def plot_prune_performance(prune_rate, model_names, model_data, dataset_name, y_min, y_max):
    """
    Plots the performance of different models under various pruning rates for baseline, phase 1, and phase 2.

    Parameters:
    - prune_rate: List of prune rates.
    - model_names: List of model names.
    - model_data: List of lists where each sublist contains accuracy values for a single model across different phases.
                  Format: [[baseline, phase1, phase2], ...]
    - dataset_name: Name of the dataset (e.g., 'Cifar-10' or 'ImageNet').
    - y_min: Minimum value of y-axis (for cropping the y-scale).
    - y_max: Maximum value of y-axis (for cropping the y-scale).
    """

    plt.figure(figsize=(5, 6))
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define a list of colors
    # colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
    colors = ['#1f77b4', '#d95f02', '#003f5c', '#e7298a', '#66a61e', '#e6ab02']

    for i, model in enumerate(model_names):
        baseline_data = model_data[i][0]
        phase1_data = model_data[i][1]
        phase2_data = model_data[i][2]

        color = colors[i % len(colors)]  # Use color from the list, cycle if more models than colors

        plt.plot(prune_rate[0], baseline_data, '.', color=color)
        plt.plot(prune_rate[1], phase1_data, 's', color=color)
        plt.plot(prune_rate[2], phase2_data, 'o', color=color)
        plt.plot(prune_rate, model_data[i], '-', color=color, label=f'{model}', linewidth=2.5)

    # plt.xlabel('Optimization Phase')
    # plt.ylabel('Accuracy (%)')
    # Increase font sizes
    plt.title('ResNet18', fontsize=15)
    # plt.xlabel('Optimization Phase', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title(dataset_name)
    plt.ylim(y_min, y_max)
    plt.margins(0.15)
    plt.legend()
    # plt.grid(True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    fig.savefig("ResNet18_Accuracy.png", dpi=300)

# Example data
# prune_rate = [1.0, 2.0, 3.0]  # Example prune rates
optimization_phase = ['Base Model', 'ADMM Model', 'Retrained Model']  # Example prune rates
model_names = [
    "Unstructured Pruning",
    "Trivial Pattern Pruning",
    "SCP Pattern Pruning",
    "Adv. Unstructured Pruning",
    "Adv. Trivial Pattern Pruning",
    "Adv. SCP Pattern Pruning"
    # "Trivial Adv. Conn",
    # "Trivial Default Conn",
    # "SCP Adv. Conn",
    # "SCP Default Conn"
]

# model_data = [
#     [0.960, 0.958, 0.955],  # VGG-16: [Baseline, Phase 1, Phase 2]
#     [0.955, 0.953, 0.950]   # ResNet-18: [Baseline, Phase 1, Phase 2]
# ]

# Plot CIFAR-10 data
# plot_prune_performance(prune_rate, model_names, model_data, 'Cifar-10', 0.94, 0.97)

# Example data for ImageNet
accuracy = [
    [68.66, 69.35, 69.19],
    [68.66, 67.82, 58.84],
    [68.66, 67.10, 67.27],
    [68.66, 69.00, 68.90],
    [68.66, 67.39, 60.30],
    [68.66, 65.71, 66.96]
    # [68.69, 65.64, 55.40],
    # [68.69, 66.29, 55.46],
    # [68.69, 64.35, 64.65],
    # [68.69, 65.09, 65.24]
]

# Plot ImageNet data
plot_prune_performance(optimization_phase, model_names, accuracy, 'ImageNet', 58.6, 69.5)

#%%
