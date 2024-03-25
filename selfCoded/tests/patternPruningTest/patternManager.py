import sys, os
sys.path.append(os.getcwd())
import torch

import patternVariations
import collections

import utils
import libraryReduction

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

        # creating pattern library
        self.pattern_library = self.create_pattern_library()

        # initializes a list of available patterns with indices of pattern library
        self.available_patterns_indices = self.initialize_available_patterns()

        self.connectivityPruningEnabled = False

    def get_tensor_assignment(self, tensor_index):
        """
        Retrieves the pattern assigned to a specific tensor.

        Args:
        tensor_index (int): The index of the tensor.

        Returns:
        str: The pattern assigned to the tensor.
        """
        return self.tensor_assignments.get(tensor_index)

    def load_pattern_library(cls):
        # hier können pattern librarys reingeladen werden
        pass

    def load_available_patterns(self):
        # hier können die available patterns aktualisiert werden
        pass

    def setConnectivityPruning(self, flag):
        '''
        With this setter you can turn on the connectivity pruning
        :param flag: bool
        '''
        self.connectivityPruningEnabled = flag


    @classmethod
    def create_pattern_library(cls):
        '''
        This method is for creation of a list of different patterns.
        Elements of list are in type tensor
        :return: list of unique patterns for example n=9, k=4 -> 3x3 Tensors with 4 Fields set to 1 rest to 0
        '''
        tensor_list = patternVariations.initialize_pattern_library(9,4)

        pattern_library = list(tensor_list.unbind(dim=0))

        return pattern_library

    def initialize_available_patterns(self):
        return list(range(len(self.pattern_library)))

    def assign_patterns_to_tensors(self, tensor_list):
        """
        Assigns a pattern to a specific tensor.
        Platzhalter für das Zuweisen des am besten passenden Patterns zu jedem Tensor
        Args:
        tensor_index (int): The index of the tensor in the tensor list.
        pattern (str): The pattern to assign to the tensor.
        """
        self.tensor_assignments = []
        # TODO: not creating every time list of same length avg_impact...
        self.abs_impact_patterns = [0] * len(self.pattern_library)
        self.avg_impact_patterns = [0] * len(self.pattern_library)
        self.pattern_counts = [0] * len(self.pattern_library)
        for layer in tensor_list:
            temp_layer = []
            for tensor in layer:
                layer_tensor = [self._choose_best_pattern(tensor=single_tensor)
                                             for single_tensor in tensor]
                temp_layer.append(layer_tensor)
            self.tensor_assignments.append(temp_layer)

        # here is the logic for connectivity pruning
        if self.connectivityPruningEnabled is True:
            threshold_ratios = [0.8, 0.5, 0.5]
            assign_connectivity_pruned_kernel(self.tensor_assignments, tensor_list, self.pattern_library,
                                              threshold_ratios)

        # tracks counts of all patterns
        self._count_pattern_assignments()

        # tracks the impact of all patterns in aggregated form
        self._calc_abs_pattern_impact(tensor_list)

        # tracks the impact of all patterns in average form
        self._calc_avg_pattern_impact()

    def _choose_best_pattern(self, tensor):
        min = 1000
        min_index = None
        for i in self.available_patterns_indices:
            frob_distance = torch.norm(torch.abs(tensor) - self.pattern_library[i], p='fro')
            if frob_distance < min:
                min = frob_distance
                min_index = i

        return min_index

    # TODO: reducing impact aggregating



    # TODO: optimierung pattern_counts soll einmal erstellt werden und nur neu belegt werden
    def _count_pattern_assignments(self):
        '''
        aggregates the number of assigned patterns per pattern
        :return: list of counted pattern distribution
        '''

        for i in range(len(self.pattern_counts)):
            self.pattern_counts[i] = count_occurrences_iterative(self.tensor_assignments, i)

        return self.pattern_counts

    def _track_pattern_impact(self, tensor, index):
        frob_distance = torch.norm(tensor * self.pattern_library[index], p='fro')
        self.abs_impact_patterns[index] += float(frob_distance)

    def _calc_abs_pattern_impact(self, tensor_list):
        '''
        calculates the absoulte value of the kernel*mask frobenius norms aggregated
        :param tensor_list: list of layers (type tensor)
        '''
        for layer_idx, layer in enumerate(tensor_list):
            for channel_idx, channel in enumerate(layer):
                for kernel_idx, kernel in enumerate(channel):
                    mask_idx = self.tensor_assignments[layer_idx][channel_idx][kernel_idx]
                    if mask_idx is not None:  # Ensure the kernel hasn't been pruned already
                        self._track_pattern_impact(kernel, mask_idx)

    def _calc_avg_pattern_impact(self):

        for i in range(len(self.pattern_library)):
            if self.pattern_counts[i] == 0:
                self.avg_impact_patterns[i] = 0
            else:
                temp_avg = self.abs_impact_patterns[i]/self.pattern_counts[i]
                self.avg_impact_patterns[i] = float(temp_avg)


    def reduce_available_patterns(self, min_amount_indices):
        '''
        interface method for selectable pattern reduction algorithms.
        This can be influence algo behavior. Many options possible
        :returns indexes which are removed from available patterns list
        '''
        # return libraryReduction.fixed_reduction_rate(self.available_patterns_indices,
        #                                              self.pattern_counts, min_amount_indices)
        return libraryReduction.impact_based_reduction_rate(self.available_patterns_indices,
                                                            self.pattern_counts, self.abs_impact_patterns,
                                                            min_amount_indices)

    def get_single_pattern_mask(self, layer_index):
        # Gibt eine Liste zurück, die für jeden Index in tensor_assignments das entsprechende Muster enthält
        return convert_to_single_tensor(self.tensor_assignments, self.pattern_library, layer_index)

    def get_pattern_masks(self):
        # Gibt eine Liste zurück, die für jeden Index in tensor_assignments das entsprechende Muster enthält
        #return convert_to_tensors(self.tensor_assignments, self.pattern_library)
        return [convert_to_single_tensor(self.tensor_assignments, self.pattern_library, i)
                for i in range(len(self.tensor_assignments))]

    def update_pattern_assignments(self, tensor_list, min_amount_indices=12):
        # Führt die erforderlichen Methoden nacheinander aus, um die Musterzuweisungen zu aktualisieren
        self.reduce_available_patterns(min_amount_indices=min_amount_indices)  # Reduziert die Liste der verfügbaren Muster
        self.assign_patterns_to_tensors(tensor_list)  # Weist Muster den Tensoren erneut zu

    # TODO: maybe this won't work because of import error for testmain.py
    def save_pattern_library(self, folder_path):
        """
        Saves each 3x3 tensor in the given 126x3x3 tensor to individual text files.
        :param tensor: A 126x3x3 PyTorch tensor.
        :param folder: The folder where text files will be saved.
        """
        utils.save_tensors_to_files(self.pattern_library, folder=folder_path)

    def save_available_patterns(self, folder_path):
        '''
        Saves the available patterns for current pruning
        :param folder_path: folder and filename to save file at
        '''
        available_patterns = self._resolute_available_tensors()
        utils.save_tensors_to_files(available_patterns, folder=folder_path)

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

    # comment this line out if you want ot enable inline configuration
    updated_indices_list = indices_list #copy.deepcopy(indices_list)

    # Process each layer with its corresponding pruning threshold
    for layer_idx, (layer, threshold_ratio) in enumerate(zip(layer_list, threshold_ratios)):
        norms = []

        # Collect Frobenius norms for all kernels in the layer
        for channel_idx, kernels in enumerate(layer):
            for kernel_idx, _ in enumerate(kernels):
                mask_idx = indices_list[layer_idx][channel_idx][kernel_idx]
                if mask_idx is not None:  # Ensure the kernel hasn't been pruned already
                    mask = mask_list[mask_idx]
                    norm = torch.norm(kernels[kernel_idx] * mask, p='fro').item()
                    norms.append((norm, layer_idx, channel_idx, kernel_idx))

        # Determine how many kernels to prune based on the specified ratio
        num_to_prune = round(len(norms) * threshold_ratio)
        # Sort norms to identify which kernels have the lowest norms
        norms_sorted = sorted(norms, key=lambda x: x[0])

        # Prune the specified percentage of kernels with the lowest norms
        for _, l_idx, c_idx, k_idx in norms_sorted[:num_to_prune]:
            updated_indices_list[l_idx][c_idx][k_idx] = None


    return updated_indices_list

def count_occurrences_iterative(nested_list, value):
    '''
    counts the occurence of value inside of the nested_list and returns the count
    '''
    count = 0
    queue = collections.deque([nested_list])

    while queue:
        current = queue.popleft()
        for item in current:
            if item == value:
                count += 1
            elif isinstance(item, list):
                queue.append(item)
    return count

def convert_to_tensors(nested_indices, tensor_list):
    result_tensors = []
    for sublist in nested_indices:
        # Pre-fetch tensors based on indices to minimize repetitive access
        fetched_tensors = [tensor_list[idx] for idx in sum(sublist, [])]

        if len(sublist[0]) == 1:  # Shape (6,1,3,3)
            # Use torch.stack directly on pre-fetched tensors with adjusted dimensions
            tensor = torch.stack(fetched_tensors).view(len(sublist), 1, 3, 3)
        else:  # Shape (16,6,3,3)
            # Calculate sizes for reshaping
            num_layers = len(sublist)
            tensors_per_layer = len(sublist[0])
            # Reshape to match (16,6,3,3), assuming uniform size across sublist
            tensor = torch.stack(fetched_tensors).view(num_layers, tensors_per_layer, 3, 3)

        result_tensors.append(tensor)
    return result_tensors

def convert_to_single_tensor(tensor_assignments, pattern_library, layer_index):
    # Directly access the specific layer's indices
    sublist = tensor_assignments[layer_index]

    # Convert indices to a PyTorch tensor for advanced indexing
    if len(sublist[0]) == 1:  # Shape (1,1,3,3)
        # Flatten the sublist since it contains single-item lists
        indices = [idx[0] for idx in sublist]
        tensor = torch.stack([pattern_library[i] if i is not None else torch.zeros(3, 3)
                              for i in indices]).unsqueeze(1)
    else:  # Shape (6,16,3,3)
        # For more complex case, use advanced indexing if possible
        layer_tensors = []
        for group in sublist:
            # Convert group to tensor for advanced indexing
            group_indices = [idx for idx in group]#torch.tensor(group)
            group_tensor = torch.stack([pattern_library[i] if i is not None else torch.zeros(3, 3)
                                        for i in group_indices])
            layer_tensors.append(group_tensor)
        tensor = torch.stack(layer_tensors)

    return tensor
#%%
