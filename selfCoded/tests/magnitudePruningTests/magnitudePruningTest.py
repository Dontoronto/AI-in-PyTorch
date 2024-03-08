import torch


# Note: Mask and Z-layer can be pruned this way
def prune_tensor(tensor, pruning_ratio):
    # Schritt 1: Flatten des Tensors
    flat_tensor = tensor.flatten()

    # Schritt 2: Bestimmung der Anzahl der zu prunenden Elemente
    num_elements_to_prune = round(len(flat_tensor) * pruning_ratio)

    # Schritt 3: Finden des Schwellenwerts
    # Hinweis: argsort gibt Indizes der sortierten Elemente zurück, nicht die sortierten Elemente selbst
    # torch.abs(flat_tensor) gibt ein Tensor mit den absoluten Werten zurück
    # [:num_elements_to_prune] schneidet die ersten 'num_elements_to_prune' Elemente ab
    threshold_idx = torch.argsort(torch.abs(flat_tensor))[:num_elements_to_prune]
    print(f"Number of pruned elements: {len(threshold_idx)}")
    # Schritt 4: Setzen der Werte unter dem Schwellenwert auf Null
    flat_tensor[threshold_idx] = 0

    # Schritt 5: Reshape des geflatteten Arrays zurück in seine ursprüngliche Form
    pruned_tensor = flat_tensor.reshape(tensor.shape)

    return pruned_tensor

# Beispiel
tensor = torch.randn(1, 1, 3, 3)  # Ein Tensor mit zufälligen Werten
pruning_ratio = 0.62  # 70% der Werte sollen auf Null gesetzt werden

print(tensor)
pruned_tensor = prune_tensor(tensor, pruning_ratio)
print(pruned_tensor)