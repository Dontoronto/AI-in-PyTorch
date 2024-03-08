import torch

def create_magnitude_pruned_mask(tensor, ratio):
    '''
    :param tensor: source tensor which needed to be masked
    :param ratio: pruning ratio
    :return: mask of pruned weights just 0 on pruned weights and 1 on non-pruned
    '''
    # Gesamtzahl der Elemente
    total_elements = tensor.numel()
    # Berechnung der Anzahl der zu behaltenden Elemente
    num_elements_to_keep = round(total_elements * (1 - ratio))

    # Bestimmung des k-ten kleinsten Wertes als Schwelle
    if num_elements_to_keep == 0:
        threshold = tensor.max() + 1  # Setzt alle auf 0, wenn alles geprunt wird
    else:
        flat_tensor = tensor.view(-1).abs()
        # torch.kthvalue findet den k-ten kleinsten Wert in einem Tensor
        # Wir verwenden total_elements - num_elements_to_keep als k, um den Schwellenwert zu finden
        threshold = torch.kthvalue(flat_tensor, k=total_elements - num_elements_to_keep + 1).values

    # Erstellen der Maske
    mask = tensor.abs() >= threshold

    return mask