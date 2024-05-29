import torch

def test_initialize_elog_based_patterns():
    # Initialize patterns
    patterns = initialize_elog_based_patterns()

    # Process the output with pattern_library
    pattern_library = list(patterns.unbind(dim=0))

    # Check if all tensors are on CUDA
    all_on_cuda = all(tensor.is_cuda for tensor in pattern_library)

    if all_on_cuda:
        print("All tensors are successfully transferred to CUDA and remain on CUDA.")
    else:
        print("Some tensors are not on CUDA.")





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

    # Konvertiere die Muster in Torch-Tensoren und verschiebe sie auf CUDA
    scp1_tensor = torch.tensor(scp1, dtype=torch.float32).cuda()
    scp2_tensor = torch.tensor(scp2, dtype=torch.float32).cuda()
    scp3_tensor = torch.tensor(scp3, dtype=torch.float32).cuda()
    scp4_tensor = torch.tensor(scp4, dtype=torch.float32).cuda()

    # Erstellen einer Liste der Tensoren
    _patterns = [scp1_tensor, scp2_tensor, scp3_tensor, scp4_tensor]

    # Konvertieren der Liste in einen Tensor
    patterns = torch.stack(_patterns)

    return patterns


# Run the test
test_initialize_elog_based_patterns()
