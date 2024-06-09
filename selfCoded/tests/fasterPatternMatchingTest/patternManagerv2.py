import sys, os
sys.path.append(os.getcwd())
import torch

from patternVariations import initialize_pattern_library, initialize_elog_based_patterns
from collections import deque

#import utils
from utils import save_tensors_to_files, count_unique_tensors
#import libraryReduction
from libraryReduction import impact_based_reduction_rate


# TODO: noch logger schreiben damit die gröbsten Schritte geloggt werden können
class PatternManager:

    # list saves available patterns
    pattern_library = []

    # list saves assignments of patterns according to the weights
    tensor_assignments = []

    # list of same length as patter_library list. Saves amount of assigned patterns at same index
    pattern_counts = []

    # list saves indexes of patterns from pattern_library which are selectable
    available_patterns_indices = []

    # TODO: avg impact calculation
    abs_impact_patterns = []
    avg_impact_patterns = []

    def __init__(self):
        '''
        Example shapes of weights torch.Size([6, 1, 3, 3]) or torch.Size([16, 6, 3, 3])
        and for fc: torch.Size([120, 784])
        '''

        self.connectivityPruningEnabled = True

        self._noReduction = False

        # creating pattern library
        self.pattern_library = self.create_pattern_library(elog_patterns=True)
        #self.pattern_library = self.create_pattern_library(elog_patterns=False)

        # initializes a list of available patterns with indices of pattern library in list format
        self.available_patterns_indices = self.initialize_available_patterns()
        #saves the available pattern (3x3) in tensor format according to indices of self.available_patterns
        self.available_patterns = None
        # saves the available indices in tensor format
        self.available_patterns_indices_tensor = None

        self.pattern_count_tensor = torch.zeros(self.pattern_library.shape[0], dtype=self.pattern_library.dtype,
                                                    device=self.pattern_library.device)

        self.abs_impact_pattern_tensor = torch.zeros(self.pattern_library.shape[0], dtype=torch.float32,
                                                         device=self.pattern_library.device)

        self.avg_impact_pattern_tensor = torch.zeros(self.pattern_library.shape[0], dtype=torch.float32,
                                                         device=self.pattern_library.device)

    def initialize_available_patterns(self):
        return list(range(len(self.pattern_library)))


    def setConnectivityPruning(self, flag):
        '''
        With this setter you can turn on the connectivity pruning
        :param flag: bool
        '''
        self.connectivityPruningEnabled = flag

    def create_pattern_library(self, elog_patterns=False):
        '''
        This method is for creation of a list of different patterns.
        Elements of list are in type tensor
        :return: list of unique patterns for example n=9, k=4 -> 3x3 Tensors with 4 Fields set to 1 rest to 0
        '''

        if elog_patterns:
            tensor_list = initialize_elog_based_patterns()
            self._noReduction = True
        else:
            #test if this will work out
            tensor_list = initialize_pattern_library(9,4)

        #pattern_library = list(tensor_list.unbind(dim=0))

        return tensor_list

    def assign_patterns_to_tensors(self, tensor_list, pruning_ratio_list=None):
        """
        Assigns a pattern to a specific tensor.
        Platzhalter für das Zuweisen des am besten passenden Patterns zu jedem Tensor
        Args:
        tensor_index (int): The index of the tensor in the tensor list.
        pattern (str): The pattern to assign to the tensor.
        """
        #self.tensor_assignments = []
        # TODO: not creating every time list of same length avg_impact...
        self.abs_impact_pattern_tensor *= 0
        self.avg_impact_pattern_tensor *= 0
        self.pattern_count_tensor *= 0

        self.available_patterns = self.pattern_library[self.available_patterns_indices]
        #saves the available pattern (3x3) in tensor format according to indices of self.available_patterns
        self.available_patterns_indices_tensor = torch.tensor(self.available_patterns_indices,
                                                              dtype=self.pattern_library.dtype,
                                                              device=self.pattern_library.device)

        self.tensor_assignments = self.process_layers(tensor_list)

        # here is the logic for connectivity pruning
        if self.connectivityPruningEnabled is True:
            threshold_ratios = pruning_ratio_list
            self.tensor_assignments = assign_connectivity_pruned_kernel(self.tensor_assignments, tensor_list,
                                                                        self.pattern_library, threshold_ratios)

        if self._noReduction is False:
            # tracks counts of all patterns
            self.count_occurrences()

            # tracks the impact of all patterns in aggregated form
            self.calc_abs_pattern_impact(tensor_list)

            # tracks the impact of all patterns in average form
            self.calc_avg_pattern_impact()

    def process_layers(self, layer_list):
        result = []
        for layer in layer_list:
            best_patterns_indices = self._choose_best_patterns(layer)
            result.append(best_patterns_indices)
        return result

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

        # Reshape to (output, input_channels)
        mapped_indices = mapped_indices.view(output, input_channels)

        return mapped_indices



    def count_occurrences(self):
        """
        Counts the occurrences of each pattern index in tensor_assignments while excluding -1 values.
        """
        # Flatten the tensor_assignments and filter out -1 values
        flat_indices = torch.cat([ta.flatten() for ta in self.tensor_assignments])
        valid_indices = flat_indices[flat_indices != -1]

        # Count occurrences of each pattern index
        #pattern_count_tensor = torch.zeros(pattern_library_size, dtype=torch.long, device=valid_indices.device)
        unique_indices, counts = valid_indices.unique(return_counts=True)
        self.pattern_count_tensor[unique_indices] = counts

        return self.pattern_count_tensor

    def calc_abs_pattern_impact(self, tensor_list):
        """
        Calculates the absolute value of the kernel*mask Frobenius norms aggregated.
        """
        self.abs_impact_pattern_tensor *= 0
        # abs_impact_pattern_tensor = torch.zeros(pattern_library.shape[0], dtype=tensor_list[0].dtype, device=tensor_list[0].device)

        for pattern_idx in range(self.pattern_library.shape[0]):
            pattern = self.pattern_library[pattern_idx]

            # Calculate Frobenius norm for each layer's tensor assignment with the current pattern
            for layer, assignments in zip(tensor_list, self.tensor_assignments):
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
                self.abs_impact_pattern_tensor[pattern_idx] += frob_norms.sum().item()

        return self.abs_impact_pattern_tensor

    def calc_avg_pattern_impact(self):
        """
        Calculates the average impact for each pattern.
        """
        # Initialize the average impact tensor
        self.avg_impact_pattern_tensor *= 0

        # Avoid division by zero
        valid_mask = self.pattern_count_tensor != 0

        # Calculate the average impact
        self.avg_impact_pattern_tensor[valid_mask] = (
                self.abs_impact_pattern_tensor[valid_mask] / self.pattern_count_tensor[valid_mask])

        return self.avg_impact_pattern_tensor


    def reduce_available_patterns(self, min_amount_indices):
        '''
        interface method for selectable pattern reduction algorithms.
        This can be influence algo behavior. Many options possible
        :returns indexes which are removed from available patterns list
        '''
        # return libraryReduction.fixed_reduction_rate(self.available_patterns_indices,
        #                                              self.pattern_counts, min_amount_indices)
        return impact_based_reduction_rate(self.available_patterns_indices,
                                           self.pattern_count_tensor, self.abs_impact_pattern_tensor,
                                           min_amount_indices)


    def get_pattern_masks(self):
        """
        Returns a list of tensors, each containing the corresponding patterns for each index in tensor_assignment_list.
        """
        return [convert_to_single_tensor(self.tensor_assignments[layer_idx],
                                         self.pattern_library) for layer_idx in range(len(self.tensor_assignments))]

    def update_pattern_assignments(self, tensor_list, min_amount_indices=12, pruning_ratio_list=None):
        # Führt die erforderlichen Methoden nacheinander aus, um die Musterzuweisungen zu aktualisieren
        if self._noReduction == False:
            self.reduce_available_patterns(min_amount_indices=min_amount_indices)  # Reduziert die Liste der verfügbaren Muster
        self.assign_patterns_to_tensors(tensor_list, pruning_ratio_list)  # Weist Muster den Tensoren erneut zu

    # TODO: maybe this won't work because of import error for testmain.py
    def save_pattern_library(self, folder_path):
        """
        Saves each 3x3 tensor in the given 126x3x3 tensor to individual text files.
        :param tensor: A 126x3x3 PyTorch tensor.
        :param folder: The folder where text files will be saved.
        """
        save_tensors_to_files(self.pattern_library, folder=folder_path)

    def save_available_patterns(self, folder_path):
        '''
        Saves the available patterns for current pruning
        :param folder_path: folder and filename to save file at
        '''
        available_patterns = self._resolute_available_tensors()
        save_tensors_to_files(available_patterns, folder=folder_path)

    def _resolute_available_tensors(self):
        '''
        helper method to resolve indexes of available patterns with pattern_library
        :return:
        '''
        temp = list()
        for index in self.available_patterns_indices:
            temp.append(self.pattern_library[index])

        return temp

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
#%%
