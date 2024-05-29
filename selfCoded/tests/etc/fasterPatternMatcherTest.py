import torch
import time

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

    def _choose_best_pattern_frobenius(self, tensor):
        min_distance = float('inf')
        min_index = None
        for i in self.available_patterns_indices:
            frob_distance = torch.norm(torch.abs(tensor) - self.pattern_library[i], p='fro')
            if frob_distance < min_distance:
                min_distance = frob_distance
                min_index = i
        return min_index

    def _choose_best_pattern_flattening(self, tensor):
        max_dot_product = float('-inf')
        max_index = None
        tensor_flattened = tensor.view(-1)
        for i in self.available_patterns_indices:
            pattern_tensor_flattened = self.pattern_library[i].view(-1)
            dot_product = torch.dot(tensor_flattened, pattern_tensor_flattened)
            if dot_product > max_dot_product:
                max_dot_product = dot_product
                max_index = i
        return max_index

    def _choose_best_pattern_elementwise(self, tensor):
        max_dot_product = float('-inf')
        max_index = None
        for i in self.available_patterns_indices:
            pattern_tensor = self.pattern_library[i]
            dot_product = torch.sum(tensor * pattern_tensor)
            if dot_product > max_dot_product:
                max_dot_product = dot_product
                max_index = i
        return max_index

# Example tensors
N = 4
raw_pattern_library = initialize_elog_based_patterns()
pattern_library = [tensor for tensor in raw_pattern_library]
# pattern_library = [torch.rand(3, 3) for _ in range(N)]
available_patterns_indices = list(range(N))
test_tensor = torch.rand(3, 3)

matcher = PatternMatcher(pattern_library, available_patterns_indices)

layer = torch.rand(56,94,3,3)

# Benchmark original method
start_time = time.time()
test_list = []
for tensor in layer:
    layer_tensor = [matcher._choose_best_pattern_frobenius(tensor=single_tensor)
                    for single_tensor in tensor]
    test_list.append(layer_tensor)
print(test_list)
#matcher._choose_best_pattern_frobenius(test_tensor)
frobenius_time = time.time() - start_time

# Benchmark flattening method
start_time = time.time()
test_list = []
for tensor in layer:
    layer_tensor = [matcher._choose_best_pattern_flattening(tensor=single_tensor)
                    for single_tensor in tensor]
    test_list.append(layer_tensor)
print(test_list)
# matcher._choose_best_pattern_flattening(test_tensor)
flattening_time = time.time() - start_time

# Benchmark element-wise method
start_time = time.time()
test_list = []
for tensor in layer:
    layer_tensor = [matcher._choose_best_pattern_elementwise(tensor=single_tensor)
                    for single_tensor in tensor]
    test_list.append(layer_tensor)
print(test_list)
# matcher._choose_best_pattern_elementwise(test_tensor)
elementwise_time = time.time() - start_time

print(f"Frobenius norm method time: {frobenius_time:.6f} seconds")
print(f"Flattening approach time: {flattening_time:.6f} seconds")
print(f"Element-wise approach time: {elementwise_time:.6f} seconds")
