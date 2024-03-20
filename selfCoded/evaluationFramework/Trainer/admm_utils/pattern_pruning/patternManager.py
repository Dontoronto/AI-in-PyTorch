import os
import torch

#from .patternVariations import initialize_pattern_library

import patternVariations
import collections
import operator

class PatternManager:
    pattern_library = []               # Speichert die verfügbaren Muster
    tensor_assignments = []            # Speichert die Zuweisungen von Mustern zu Tensoren
    pattern_counts = []                # Speichert die Anzahl der Zuweisungen für jedes Muster
    available_patterns_indices = []    # Speichert die Indizes der Muster, die noch zur Auswahl stehen

    def __init__(self):
        '''
        Example shapes of weights torch.Size([6, 1, 3, 3]) or torch.Size([16, 6, 3, 3])
        and for fc: torch.Size([120, 784])
        '''

        # creating pattern library
        self.pattern_library = self.create_pattern_library()

        # initializes a list of available patterns with indices of pattern library
        self.available_patterns_indices = self.initialize_available_patterns()

        # Weist Muster den Tensoren zu
        #self.assign_patterns_to_tensors()

        # Zählt, wie oft jedes Muster zugewiesen wird
        #self.count_pattern_assignments()

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
        for layer in tensor_list:
            temp_layer = []
            for tensor in layer:
                #print(f"shape of temp layer = {tensor.shape}")
                layer_tensor = [self._choose_best_pattern(tensor=single_tensor)
                                             for single_tensor in tensor]
                #print(f"shape of layer-tensor = {layer_tensor}")
                temp_layer.append(layer_tensor)
            self.tensor_assignments.append(temp_layer)

        self.count_pattern_assignments()


        # if pattern not in self.patterns:
        #     raise ValueError(f"Pattern '{pattern}' is not recognized.")
        # self.tensor_assignments[tensor_index] = pattern

    def _choose_best_pattern(self, tensor):
        min = 1000
        min_index = None
        for i in self.available_patterns_indices:
            frob_distance = torch.norm(torch.abs(tensor) - self.pattern_library[i], p='fro')
            if frob_distance < min:
                min = frob_distance
                min_index = i

        return min_index

    def count_pattern_assignments(self):
        # Zählt, wie oft jedes Muster in tensor_assignments zugewiesen wird
        self.pattern_counts = [0] * len(self.pattern_library)

        for i in range(len(self.pattern_counts)):
            self.pattern_counts[i] = count_occurrences_iterative(self.tensor_assignments, i)

        return self.pattern_counts

    def reduce_available_patterns(self):
        # Platzhaltermethode zur Reduzierung der Liste der auswählbaren Muster
        if len(self.available_patterns_indices) > 12:
            sliced_list = operator.itemgetter(*self.available_patterns_indices)(self.pattern_counts)

            lowest_index = index_of_lowest_value(sliced_list)

            self.available_patterns_indices.pop(lowest_index)

            return lowest_index

    def get_single_pattern_mask(self, layer_index):
        # Gibt eine Liste zurück, die für jeden Index in tensor_assignments das entsprechende Muster enthält
        return convert_to_single_tensor(self.tensor_assignments, self.pattern_library, layer_index)

    def get_pattern_masks(self):
        # Gibt eine Liste zurück, die für jeden Index in tensor_assignments das entsprechende Muster enthält
        return convert_to_tensors(self.tensor_assignments, self.pattern_library)

    def update_pattern_assignments(self, tensor_list):
        # Führt die erforderlichen Methoden nacheinander aus, um die Musterzuweisungen zu aktualisieren
        self.reduce_available_patterns()  # Reduziert die Liste der verfügbaren Muster
        self.assign_patterns_to_tensors(tensor_list)  # Weist Muster den Tensoren erneut zu

    def save_pattern_library(self, folder_path):
        """
        Saves each 3x3 tensor in the given 126x3x3 tensor to individual text files.
        :param tensor: A 126x3x3 PyTorch tensor.
        :param folder: The folder where text files will be saved.
        """
        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)

        file_name = f"{folder_path}/tensor.txt"
        with open(file_name, 'w') as file:
            #Iterate and save each 3x3 tensor
            for i, matrix in enumerate(tensor):
                for row in matrix:
                    file.write(' '.join(map(str, row.tolist())) + '\n')
                file.write('\n')

@staticmethod
def count_occurrences_iterative(nested_list, value):
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

@staticmethod
def index_of_lowest_value(lst):
    return lst.index(min(lst)) if lst else None

@staticmethod
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

@staticmethod
def convert_to_single_tensor(tensor_assignments, pattern_library, layer_index):
    # Directly access the specific layer's indices
    sublist = tensor_assignments[layer_index]

    # Convert indices to a PyTorch tensor for advanced indexing
    if len(sublist[0]) == 1:  # Shape (6,1,3,3)
        # Flatten the sublist since it contains single-item lists
        indices = torch.tensor([idx[0] for idx in sublist])
        tensor = torch.stack([pattern_library[i] for i in indices]).unsqueeze(1)
    else:  # Shape (16,6,3,3)
        # For more complex case, use advanced indexing if possible
        layer_tensors = []
        for group in sublist:
            # Convert group to tensor for advanced indexing
            group_indices = torch.tensor(group)
            group_tensor = torch.stack([pattern_library[i] for i in group_indices])
            layer_tensors.append(group_tensor)
        tensor = torch.stack(layer_tensors)

    return tensor
#%%
