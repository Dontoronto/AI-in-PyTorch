import time
import torch
from concurrent.futures import ProcessPoolExecutor

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
    scp1_tensor = torch.tensor(scp1, dtype=torch.float32)
    scp2_tensor = torch.tensor(scp2, dtype=torch.float32)
    scp3_tensor = torch.tensor(scp3, dtype=torch.float32)
    scp4_tensor = torch.tensor(scp4, dtype=torch.float32)

    # Erstellen einer Liste der Tensoren
    _patterns = [scp1_tensor, scp2_tensor, scp3_tensor, scp4_tensor]

    # Konvertieren der Liste in einen Tensor
    patterns = torch.stack(_patterns)

    return patterns

class PatternMatcher:
    def __init__(self, pattern_library, available_patterns_indices):
        self.pattern_library = pattern_library
        self.available_patterns_indices = available_patterns_indices

    def _choose_best_pattern(self, tensor):
        min_frobenius_distance = float('inf')
        min_index = None
        for i in self.available_patterns_indices:
            frob_distance = torch.norm(torch.abs(tensor) - self.pattern_library[i], p='fro')
            if frob_distance < min_frobenius_distance:
                min_frobenius_distance = frob_distance
                min_index = i
        return min_index

    def process_layer(self, tensor_list):
        tensor_assignments = []
        for layer in tensor_list:
            temp_layer = [[None] * layer.size(1) for _ in range(layer.size(0))]
            with ProcessPoolExecutor() as executor:
                futures = {}
                for i in range(layer.size(0)):
                    for j in range(layer.size(1)):
                        futures[(i, j)] = executor.submit(self._choose_best_pattern, layer[i, j])

                for (i, j), future in futures.items():
                    temp_layer[i][j] = future.result()

            tensor_assignments.append(temp_layer)
        return tensor_assignments

    def regularCodeSnippet(self, tensor_list):
        tensor_assignments = []
        for layer in tensor_list:
            temp_layer = []
            for tensor in layer:
                layer_tensor = [self._choose_best_pattern(tensor=single_tensor)
                                for single_tensor in tensor]
                temp_layer.append(layer_tensor)
            tensor_assignments.append(temp_layer)

        return tensor_assignments

if __name__ == '__main__':

    # Example usage
    pattern_library = initialize_elog_based_patterns()
    available_patterns_indices = range(pattern_library.size(0))
    matcher = PatternMatcher(pattern_library, available_patterns_indices)

    layer = [torch.randn(256, 384, 3, 3) for _ in range(2)]

    start_time = time.time()
    processed_layer = matcher.process_layer(layer)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time with multiprocessing: {elapsed_time:.6f} seconds")

    start_time = time.time()
    normal_processed_layer = matcher.regularCodeSnippet(layer)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time with normal approach: {elapsed_time:.6f} seconds")
