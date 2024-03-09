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


def add_tensor_lists_inplace(target_list: list, listA: list, listB: list):
    '''
    This method iterates over 3 Lists(of tensors) on same index level and adds them
    :param target_list: List of tensors which receive the equation
    :param listA:  list of tensors which are term of summation
    :param listB:  list of tensors which are term of summation
    '''
    for target_tensor, tensor1, tensor2 in zip(target_list, listA, listB):
        add_tensors_inplace(target_tensor, tensor1, tensor2)


def add_tensors_inplace(target: torch.Tensor, term1: torch.Tensor, term2: torch.Tensor):
    '''
    efficient way to add and assign tensors
    !!! Keep in mind not to mixup argument sequence, second argument in this function will be copyed into target
        and then get added by the third argument!!! (mixing up results in wrong values)
    :param target: placed sum placed inside of this tensor
    :param term1: first tensor to add
    :param term2: second tesnor to add
    '''
    target.copy_(term1).add_(term2)

def subtract_tensors_inplace(target: torch.Tensor, term1: torch.Tensor, term2: torch.Tensor):
    '''
    Efficient way to subtract tensors and assign the result.
    !!! Keep in mind not to mixup argument sequence, second argument in this function will be copyed into target
        and then get subtracted by the third argument!!! (mixing up results in wrong values)
    :param target: The tensor where the result will be placed (modified in-place).
    :param term1: The tensor from which to subtract.
    :param term2: The tensor to subtract.
    '''
    target.copy_(term1).sub_(term2)

def scale_and_add_tensors_inplace(target: torch.Tensor, alpha: float, x: torch.Tensor, y: torch.Tensor):
    '''
    Simulates the caffe_gpu_axpy function for PyTorch tensors.
    Performs the operation alpha * x + y and stores the result in target tensor.

    !!! Keep in mind the order of arguments to avoid confusion:
    - target is the tensor where the result will be stored.
    - alpha is the scalar by which x is scaled.
    - x is the first tensor to be scaled by alpha.
    - y is the tensor to be added to the scaled x.

    :param target: The tensor where the result will be placed (modified in-place).
    :param alpha: The scalar to scale x.
    :param x: The tensor to be scaled by alpha.
    :param y: The tensor to add to the scaled x.
    '''
    target.copy_(y).add_(x, alpha=alpha)
