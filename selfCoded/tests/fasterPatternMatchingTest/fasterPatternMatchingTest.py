import torch


def initialize_elog_based_patterns():
    '''
    Generiert die vier spezifischen Sparse Convolution Patterns (SCPs) und gibt sie als Tensor zurück.
    :return: ein Tensor, der die vier 3x3 SCP-Muster enthält
    '''
    # Definieren der vier SCP-Muster
    scp1 = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0]
    ]

    scp2 = [
        [0, 1, 0],
        [1, 1, 0],
        [0, 1, 0]
    ]

    scp3 = [
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]

    scp4 = [
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 0]
    ]

    # Konvertiere die Muster in Torch-Tensoren
    scp1_tensor = torch.tensor(scp1, dtype=torch.int64)
    scp2_tensor = torch.tensor(scp2, dtype=torch.int64)
    scp3_tensor = torch.tensor(scp3, dtype=torch.int64)
    scp4_tensor = torch.tensor(scp4, dtype=torch.int64)

    # Erstellen einer Liste der Tensoren
    _patterns = [scp1_tensor, scp2_tensor, scp3_tensor, scp4_tensor]

    # Konvertieren der Liste in einen Tensor
    patterns = torch.stack(_patterns)

    return patterns

class PatternMatcher:
    def __init__(self, pattern_library, available_patterns_indices):
        # Stack the pattern library and extract available patterns
        #self.pattern_library = torch.stack(pattern_library)  # Shape: (n, 3, 3, 3)
        self.pattern_library = pattern_library  # Shape: (n, 3, 3, 3)
        self.available_patterns = self.pattern_library[available_patterns_indices]  # Shape: (4, 3, 3, 3)
        print(f"availabe patterns = {self.available_patterns.shape[0]}")
        #self.available_patterns_indices_tensor = torch.tensor([1,5,7,8])
        self.available_patterns_indices_tensor = torch.tensor(available_patterns_indices)

    def _choose_best_patterns(self, layer):
        output, input_channels, height, width = layer.shape
        # Reshape layer for broadcasting with available patterns
        layer = layer.view(output * input_channels, 1, height, width)  # Shape: (output*input_channels, 1, 3, 3)
        available_patterns = self.available_patterns.view(1, self.available_patterns.shape[0], height, width)  # Shape: (1, 4, 3, 3)

        # Compute dot products
        # dot_products = torch.sum(layer * available_patterns, dim=(2, 3))  # Shape: (output*input_channels, 4)
        prod = torch.abs(layer * available_patterns)
        dot_products = torch.sum(prod, dim=(2, 3))

        # Find indices of the max dot products
        max_indices = torch.argmax(dot_products, dim=1)  # Shape: (output*input_channels)
        mapped_indices = self.available_patterns_indices_tensor[max_indices]
        #print(test_slice)

        # Reshape to (output, input_channels)
        mapped_indices = mapped_indices.view(output, input_channels)

        return mapped_indices

    def process_layers(self, layer_list):
        result = []
        for layer in layer_list:
            best_patterns_indices = self._choose_best_patterns(layer)
            result.append(best_patterns_indices)
        return result

# def get_pattern_masks(tensor_assignment_list, pattern_library):
#     # Gibt eine Liste zurück, die für jeden Index in tensor_assignments das entsprechende Muster enthält
#     #return convert_to_tensors(self.tensor_assignments, self.pattern_library)
#     return [convert_to_single_tensor(tensor_assignment_list, pattern_library, i)
#             for i in range(len(tensor_assignment_list))]
#
# def convert_to_single_tensor(tensor_assignments, pattern_library, layer_index):
#     # Directly access the specific layer's indices
#     sublist = tensor_assignments[layer_index]
#
#     # Convert indices to a PyTorch tensor for advanced indexing
#     if len(sublist[0]) == 1:  # Shape (1,1,3,3)
#         # Flatten the sublist since it contains single-item lists
#         indices = [idx[0] for idx in sublist]
#         tensor = torch.stack([pattern_library[i] if i is not None else torch.zeros(3, 3)
#                               for i in indices]).unsqueeze(1)
#     else:  # Shape (6,16,3,3)
#         # For more complex case, use advanced indexing if possible
#         layer_tensors = []
#         for group in sublist:
#             # Convert group to tensor for advanced indexing
#             group_indices = [idx for idx in group]#torch.tensor(group)
#             group_tensor = torch.stack([pattern_library[i] if i is not None else torch.zeros(3, 3)
#                                         for i in group_indices])
#             layer_tensors.append(group_tensor)
#         tensor = torch.stack(layer_tensors)
#
#     return tensor

def get_pattern_masks(tensor_assignment_list, pattern_library):
    """
    Returns a list of tensors, each containing the corresponding patterns for each index in tensor_assignment_list.
    """
    return [convert_to_single_tensor(tensor_assignment_list[layer_idx], pattern_library) for layer_idx in range(len(tensor_assignment_list))]

# def convert_to_single_tensor(layer_assignments, pattern_library):
#     """
#     Convert tensor assignments to a tensor containing the corresponding patterns from the pattern library.
#     """
#     # Flatten the layer assignments to create a 1D tensor of indices
#     indices = torch.tensor([item for sublist in layer_assignments for item in sublist], dtype=torch.long)
#
#     # Use advanced indexing to gather patterns and reshape to the original structure
#     pattern_shape = pattern_library.shape[1:]  # Get shape of a single pattern
#     gathered_patterns = pattern_library[indices].reshape(*indices.shape, *pattern_shape)
#
#     return gathered_patterns

def convert_to_single_tensor(layer_assignments, pattern_library):
    """
    Convert tensor assignments to a tensor containing the corresponding patterns from the pattern library.
    Replace indices of -1 with a 3x3 kernel filled with zeros.
    """
    # Convert layer assignments to a tensor of indices
    indices = layer_assignments.detach()
    #indices = torch.tensor(layer_assignments, dtype=torch.int32)

    # Create a tensor of zeros with the same shape as a pattern
    zero_kernel = torch.zeros(pattern_library.shape[1:], dtype=pattern_library.dtype, device=pattern_library.device)

    # Initialize the result tensor
    result = torch.zeros((*indices.shape, *pattern_library.shape[1:]), dtype=pattern_library.dtype, device=pattern_library.device)

    # Fill the result tensor with patterns from the library or zero kernels
    mask = indices != -1
    result[mask] = pattern_library[indices[mask]]
    result[~mask] = zero_kernel

    return result

# Example usage:
#pattern_library = [torch.randn(3, 3) for _ in range(10)]  # Replace with your actual patterns
pattern_library = initialize_elog_based_patterns()
available_patterns_indices = [0, 1, 2, 3]  # Example indices

matcher = PatternMatcher(pattern_library, available_patterns_indices)

layer_list = [torch.randn(6, 1, 3, 3)]  # Replace with your actual data
#print(layer_list)
#print(matcher.available_patterns)
result = matcher.process_layers(layer_list)
#print(f"result shape={result[0].shape}")

#result[0][0][0]=-1
# Print results
# for i, res in enumerate(result):
#     print(f"Layer {i}:")
#     print(res)
#


# test = get_pattern_masks(result, pattern_library)
# for i, res in enumerate(test):
#     print(f"Layer {i}:")
#     print(res.shape)




#
# for i, res in enumerate(layer_list[1]):
#     print(f"Layer {i}:")
#     print(res)



# def assign_connectivity_pruned_kernel(indices_list, layer_list, mask_list, threshold_ratios):
#     assert len(layer_list) == len(threshold_ratios), "Mismatch between number of layers and number of threshold ratios."
#
#     # comment this line out if you want ot enable inline configuration
#     updated_indices_list = indices_list #copy.deepcopy(indices_list)
#
#     # Process each layer with its corresponding pruning threshold
#     for layer_idx, (layer, threshold_ratio) in enumerate(zip(layer_list, threshold_ratios)):
#         norms = []
#
#         # Collect Frobenius norms for all kernels in the layer
#         for channel_idx, kernels in enumerate(layer):
#             for kernel_idx, _ in enumerate(kernels):
#                 mask_idx = indices_list[layer_idx][channel_idx][kernel_idx]
#                 if mask_idx != -1:  # Ensure the kernel hasn't been pruned already
#                     mask = mask_list[mask_idx]
#                     norm = torch.norm(kernels[kernel_idx] * mask, p='fro').item()
#                     norms.append((norm, layer_idx, channel_idx, kernel_idx))
#
#         # Determine how many kernels to prune based on the specified ratio
#         num_to_prune = round(len(norms) * threshold_ratio)
#         # Sort norms to identify which kernels have the lowest norms
#         norms_sorted = sorted(norms, key=lambda x: x[0])
#
#         # Prune the specified percentage of kernels with the lowest norms
#         for _, l_idx, c_idx, k_idx in norms_sorted[:num_to_prune]:
#             updated_indices_list[l_idx][c_idx][k_idx] = -1
#
#
#     return updated_indices_list

def assign_connectivity_pruned_kernel(indices_list, layer_list, mask_list, threshold_ratios):
    assert len(layer_list) == len(threshold_ratios), "Mismatch between number of layers and number of threshold ratios."

    updated_indices_list = indices_list  # Use original list for inline modification

    # Process each layer with its corresponding pruning threshold
    for layer_idx, (layer, threshold_ratio) in enumerate(zip(layer_list, threshold_ratios)):
        #layer = layer  # Ensure layer tensor is on GPU for efficient computation
        #mask_list = mask_list  # Ensure mask tensor is on GPU for efficient computation

        # Collect Frobenius norms for all kernels in the layer
        masks = mask_list[indices_list[layer_idx].flatten()].reshape_as(layer)
        prod = torch.abs(layer * masks)
        norms = torch.sum(prod, dim=(2, 3))

        # Flatten norms and get the corresponding indices
        norms_flat = norms.flatten()
        sorted_indices = torch.argsort(norms_flat)

        # Determine how many kernels to prune based on the specified ratio
        num_to_prune = round(len(norms_flat) * threshold_ratio)

        # Prune the specified percentage of kernels with the lowest norms
        prune_indices = sorted_indices[:num_to_prune]
        prune_indices_unraveled = torch.unravel_index(prune_indices, norms.shape)

        updated_indices_list[layer_idx][prune_indices_unraveled[0], prune_indices_unraveled[1]] = -1

    return updated_indices_list


con_test = assign_connectivity_pruned_kernel(result, layer_list, pattern_library,
                                             [0.5])

test = get_pattern_masks(con_test, pattern_library)
for i, res in enumerate(test):
    print(f"Layer {i}:")
    print(res.shape)

print(type(pattern_library))
print(pattern_library.shape)


# from collections import deque
# def count_occurrences_iterative(tensor_assignments, index):
#     '''
#     counts the occurence of value inside of the nested_list and returns the count
#     '''
#     count = 0
#     queue = deque([tensor_assignments])
#
#     while queue:
#         current = queue.popleft()
#         for item in current:
#             if item == index:
#                 count += 1
#             elif isinstance(item, list):
#                 queue.append(item)
#     return count
#
# def _count_pattern_assignments(pattern_count_tensor, tensor_assignments):
#     '''
#     aggregates the number of assigned patterns per pattern
#     :return: list of counted pattern distribution
#     '''
#
#     for i in range(len(pattern_count_tensor)):
#         pattern_count_tensor[i] = count_occurrences_iterative(tensor_assignments, i)
#
#     return pattern_count_tensor


def count_occurrences(tensor_assignments, pattern_count_tensor):
    """
    Counts the occurrences of each pattern index in tensor_assignments while excluding -1 values.
    """
    # Flatten the tensor_assignments and filter out -1 values
    flat_indices = torch.cat([ta.flatten() for ta in tensor_assignments])
    valid_indices = flat_indices[flat_indices != -1]

    # Count occurrences of each pattern index
    #pattern_count_tensor = torch.zeros(pattern_library_size, dtype=torch.long, device=valid_indices.device)
    unique_indices, counts = valid_indices.unique(return_counts=True)
    pattern_count_tensor[unique_indices] = counts

    return pattern_count_tensor


pattern_count_tensor_instance = torch.zeros(pattern_library.shape[0], dtype=pattern_library.dtype,
                                            device=pattern_library.device)

#counts = _count_pattern_assignments(pattern_count_tensor, con_test)
counts = count_occurrences(con_test, pattern_count_tensor_instance)
# print(f"pattern_count_tensor: {pattern_count_tensor}")
print(f"con_test: {con_test}")
print(counts)

# def _track_pattern_impact(tensor, index, pattern_library, abs_impact_pattern_tensor):
#     frob_distance = torch.norm(tensor * pattern_library[index], p='fro')
#     abs_impact_pattern_tensor[index] += float(frob_distance)
#
#
# def _calc_abs_pattern_impact(tensor_list, tensor_assignments, pattern_library, abs_impact_pattern_tensor):
#     '''
#     calculates the absoulte value of the kernel*mask frobenius norms aggregated
#     :param tensor_list: list of layers (type tensor)
#     '''
#     for layer_idx, layer in enumerate(tensor_list):
#         for channel_idx, channel in enumerate(layer):
#             for kernel_idx, kernel in enumerate(channel):
#                 mask_idx = tensor_assignments[layer_idx][channel_idx][kernel_idx]
#                 if mask_idx is not None:  # Ensure the kernel hasn't been pruned already
#                     _track_pattern_impact(kernel, mask_idx, pattern_library, abs_impact_pattern_tensor)


# Note: not right
# def _track_pattern_impact(layer, mask, pattern_library):
#     """
#     Calculates the Frobenius norm of each kernel multiplied by the corresponding pattern and aggregates the absolute impact.
#     """
#     # Compute the Frobenius norm for each kernel
#     frob_distances = torch.norm(layer * mask.unsqueeze(1), p='fro', dim=(2, 3))
#
#     # Sum the norms for each pattern index
#     impact = frob_distances.sum(dim=0)
#
#     return impact
#
# def _calc_abs_pattern_impact(tensor_list, tensor_assignments, pattern_library, abs_impact_pattern_tensor):
#     """
#     Calculates the absolute value of the kernel*mask Frobenius norms aggregated.
#     """
#     abs_impact_pattern_tensor *= 0
#     # abs_impact_pattern_tensor = torch.zeros(pattern_library.shape[0], dtype=tensor_list[0].dtype, device=tensor_list[0].device)
#
#     for layer, assignments in zip(tensor_list, tensor_assignments):
#         # Convert assignments to tensor
#         assignments_tensor = assignments.detach()
#
#         # Create a mask to exclude -1 values
#         valid_mask = assignments_tensor != -1
#         valid_assignments = assignments_tensor[valid_mask]
#
#         # Skip the calculation if there are no valid assignments
#         if valid_assignments.numel() == 0:
#             continue
#
#         # Create a mask by gathering patterns from the library for valid assignments
#         valid_patterns = pattern_library[valid_assignments]
#
#         # Initialize the mask tensor with zeros
#         mask = torch.zeros_like(layer)
#
#         # Apply the valid patterns to the mask tensor
#         mask[valid_mask] = valid_patterns.permute(1, 0, 2, 3)
#
#         # Calculate the impact for this layer
#         layer_impact = _track_pattern_impact(layer, mask, pattern_library)
#
#         # Aggregate the impacts
#         abs_impact_pattern_tensor += layer_impact
#
#     return abs_impact_pattern_tensor

def _calc_abs_pattern_impact(tensor_list, tensor_assignments, pattern_library, abs_impact_pattern_tensor):
    """
    Calculates the absolute value of the kernel*mask Frobenius norms aggregated.
    """
    abs_impact_pattern_tensor *= 0
    # abs_impact_pattern_tensor = torch.zeros(pattern_library.shape[0], dtype=tensor_list[0].dtype, device=tensor_list[0].device)

    for pattern_idx in range(pattern_library.shape[0]):
        pattern = pattern_library[pattern_idx]

        # Calculate Frobenius norm for each layer's tensor assignment with the current pattern
        for layer, assignments in zip(tensor_list, tensor_assignments):
            layer = layer#.to('cuda')  # Ensure layer tensor is on GPU for efficient computation
            assignments_tensor = assignments.detach()#torch.tensor(assignments, dtype=torch.long, device=layer.device)

            # Mask to exclude -1 values
            valid_mask = (assignments_tensor == pattern_idx)

            # Skip if no valid assignments
            if not valid_mask.any():
                continue

            # Apply the pattern to valid assignments
            valid_layer = layer[valid_mask].reshape(-1, *pattern.shape)
            valid_masked_layer = valid_layer * pattern

            # Calculate Frobenius norm and sum it up
            frob_norms = torch.norm(valid_masked_layer, p='fro', dim=(1, 2))
            abs_impact_pattern_tensor[pattern_idx] += frob_norms.sum().item()

    return abs_impact_pattern_tensor

abs_impact_pattern_tensor_instance = torch.zeros(pattern_library.shape[0], dtype=torch.float32,
                                            device=layer_list[0].device)

# print(f"layer_list: {layer_list}")
# print(f"con_test: {con_test}")
# print(f"pattern_library: {pattern_library}")
# print(f"abs_impact_pattern_tensor_instance: {abs_impact_pattern_tensor_instance}")

abs_impact = _calc_abs_pattern_impact(layer_list, con_test, pattern_library, abs_impact_pattern_tensor_instance)

print(abs_impact)

def _calc_avg_pattern_impact(pattern_count_tensor, abs_impact_pattern_tensor, avg_impact_pattern_tensor):
    """
    Calculates the average impact for each pattern.
    """
    # Initialize the average impact tensor
    avg_impact_pattern_tensor *= 0

    # Avoid division by zero
    valid_mask = pattern_count_tensor != 0

    # Calculate the average impact
    avg_impact_pattern_tensor[valid_mask] = abs_impact_pattern_tensor[valid_mask] / pattern_count_tensor[valid_mask]

    return avg_impact_pattern_tensor

avg_impact_pattern_tensor_instance = torch.zeros(pattern_library.shape[0], dtype=torch.float32,
                                                 device=layer_list[0].device)

avg_test = _calc_avg_pattern_impact(pattern_count_tensor_instance, abs_impact_pattern_tensor_instance,
                         avg_impact_pattern_tensor_instance)

print(f"avg_impact: {avg_test}")

import libraryReduction

def reduce_available_patterns(min_amount_indices, available_patterns_indices, pattern_counts, abs_impact_patterns):
    '''
    interface method for selectable pattern reduction algorithms.
    This can be influence algo behavior. Many options possible
    :returns indexes which are removed from available patterns list
    '''
    # return libraryReduction.fixed_reduction_rate(self.available_patterns_indices,
    #                                              self.pattern_counts, min_amount_indices)
    return libraryReduction.impact_based_reduction_rate(available_patterns_indices,
                                       pattern_counts, abs_impact_patterns,
                                       min_amount_indices)


print(available_patterns_indices)
min_index_reduce = reduce_available_patterns(2, available_patterns_indices, pattern_count_tensor_instance,
                                             abs_impact_pattern_tensor_instance)
min_index_reduce = reduce_available_patterns(2, available_patterns_indices, pattern_count_tensor_instance,
                                             abs_impact_pattern_tensor_instance)

print(available_patterns_indices)
print(min_index_reduce)
print(available_patterns_indices)
