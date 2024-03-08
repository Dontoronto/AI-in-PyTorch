import torch
import numpy as np

def generate_pruning_mask(tensor, ratio):
    # Schritt 1: Berechnung der Anzahl der zu behaltenden Elemente
    total_elements = tensor.numel()
    num_elements_to_keep = round(total_elements * (1 - ratio))

    # Schritt 2: Flatten des Tensors und Sortierung der absoluten Werte
    flat_tensor = tensor.flatten().abs()
    sorted_values, _ = torch.sort(flat_tensor, descending=True)

    # Schritt 3: Bestimmung der Pruning-Schwelle
    threshold = sorted_values[num_elements_to_keep - 1] if num_elements_to_keep > 0 else sorted_values[0]

    # Schritt 4: Erstellen der Maske
    mask = tensor.abs() >= threshold

    return mask.type_as(tensor)

# Test der Funktion
tensor = torch.randn(1, 1, 3, 3) # Erzeugen eines Tensors mit zufälligen Werten
pruning_mask = generate_pruning_mask(tensor, 0.7) # Generieren der Pruning-Maske mit einem Verhältnis von 70%

