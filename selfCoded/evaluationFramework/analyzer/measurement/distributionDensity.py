import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.cm as cm


# def calculate_distribution_density(model):
#     """
#     Berechnet die Verteilungsdichte der Gewichte eines PyTorch-Modells.
#
#     Args:
#     - model (torch.nn.Module): Ein PyTorch-Modell.
#
#     Returns:
#     - numpy.ndarray: Die Werte der Bins.
#     - numpy.ndarray: Die Kanten der Bins.
#     """
#     # Extrahiere alle Gewichte des Modells
#     weights = []
#     for param in model.parameters():
#         weights += param.cpu().detach().numpy().flatten().tolist()
#     weights = np.array(weights)
#
#     # Berechne die Verteilungsdichte
#     density, bin_edges = np.histogram(weights, bins=30, density=True)
#     return density, bin_edges
#
# def plot_distribution_density(density, bin_edges):
#     """
#     Stellt die Verteilungsdichte grafisch dar.
#
#     Args:
#     - density (numpy.ndarray): Die Werte der Bins.
#     - bin_edges (numpy.ndarray): Die Kanten der Bins.
#     """
#     # Mittelpunkte der Bins berechnen für die Darstellung
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#
#     # Erstelle das Diagramm
#     plt.figure(figsize=(10, 6))
#     plt.bar(bin_centers, density, width=bin_edges[1] - bin_edges[0], color='blue', alpha=0.7)
#     plt.xlabel('Gewichtswert')
#     plt.ylabel('Dichte')
#     plt.title('Verteilungsdichte der Modellgewichte')
#     plt.show()

def calculate_optimal_density_range(model):
    """
    Berechnet die Verteilungsdichte der Gewichte eines PyTorch-Modells.

    Args:
    - model (torch.nn.Module): Ein PyTorch-Modell.
    - bins (int): Die Anzahl der Bins für das Histogramm.
    - density_range (tuple): Ein Tupel (min, max) zur Beschränkung des Wertebereichs.

    Returns:
    - numpy.ndarray: Die Werte der Bins.
    - numpy.ndarray: Die Kanten der Bins.
        """
    weights = []
    for param in model.parameters():
        weights += param.cpu().detach().numpy().flatten().tolist()
    weights = np.array(weights)

    mean_weight = np.mean(weights)
    std_weight = np.std(weights)
    density_range = (mean_weight - 3*std_weight, mean_weight + 3*std_weight)

    return density_range

def calculate_distribution_density(model, bins=None, density_range=None):
    """
    Berechnet die Verteilungsdichte der Gewichte eines PyTorch-Modells.

    Args:
    - model (torch.nn.Module): Ein PyTorch-Modell.
    - bins (int): Die Anzahl der Bins für das Histogramm.
    - density_range (tuple): Ein Tupel (min, max) zur Beschränkung des Wertebereichs.

    Returns:
    - numpy.ndarray: Die Werte der Bins.
    - numpy.ndarray: Die Kanten der Bins.
    """
    weights = []
    for param in model.parameters():
        weights += param.cpu().detach().numpy().flatten().tolist()
    weights = np.array(weights)

    if bins is None:
        n = len(weights)
        bins = int(np.ceil(2 * n**(1/3)))

    if density_range is None:
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        density_range = (mean_weight - 3*std_weight, mean_weight + 3*std_weight)

    density, bin_edges = np.histogram(weights, bins=bins, range=density_range, density=True)
    return density, bin_edges

def plot_distribution_density(density, bin_edges, log_scale=False):
    """
    Stellt die Verteilungsdichte grafisch dar.

    Args:
    - density (numpy.ndarray): Die Werte der Bins.
    - bin_edges (numpy.ndarray): Die Kanten der Bins.
    - log_scale (bool): Wenn True, wird die y-Achse logarithmisch skaliert.
    """
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normierung der Dichtewerte für die Farbskala
    #norm = plt.Normalize(density.min()*0.3, density.max()*0.3)
    # norm = plt.Normalize(vmin=0.2, vmax=0.8)
    # colors = cm.viridis(norm(density))
    #colors = cm.winter(norm(density))


    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, density, width=bin_edges[1] - bin_edges[0], color='blue', alpha=0.7)
    # plt.bar(bin_centers, density, width=bin_edges[1] - bin_edges[0], color='blue', alpha=0.7)
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.title('Distribution Density of model weights')
    if log_scale:
        plt.yscale('log')

    #plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis), label='Density')
    plt.show()
    fig_density = plt.gcf()
    plt.show()
    plt.close(fig_density)

    return fig_density
