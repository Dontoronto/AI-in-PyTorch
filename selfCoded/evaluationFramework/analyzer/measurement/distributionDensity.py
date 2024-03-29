import numpy as np
import matplotlib.pyplot as plt
import torch

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

def calculate_distribution_density(model, bins=100, density_range=None):
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
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, density, width=bin_edges[1] - bin_edges[0], color='blue', alpha=0.7)
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.title('Distribution Density of model weights')
    if log_scale:
        plt.yscale('log')
    plt.show()
