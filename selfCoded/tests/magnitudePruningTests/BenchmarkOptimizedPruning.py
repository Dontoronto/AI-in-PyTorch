import torch
import cProfile
from timeit import timeit
from memory_profiler import memory_usage

def generate_pruning_mask_optimized(tensor, ratio):
    total_elements = tensor.numel()
    num_elements_to_keep = round(total_elements * (1 - ratio))
    if num_elements_to_keep == 0:
        threshold = tensor.max() + 1
    else:
        flat_tensor = tensor.view(-1).abs()
        threshold = torch.kthvalue(flat_tensor, k=total_elements - num_elements_to_keep + 1).values
    mask = tensor.abs() >= threshold
    return mask

def main():
    # Zeitmessung mit timeit
    timeit_setup = '''
import torch
from __main__ import generate_pruning_mask_optimized
tensor = torch.randn(1000000, 1, 3, 3)  # Tensor wird innerhalb des Setup-Strings definiert
'''
    timeit_code = 'generate_pruning_mask_optimized(tensor, 0.7)'
    execution_time = timeit(stmt=timeit_code, setup=timeit_setup, number=100)
    print(f"Die Ausführungszeit beträgt (durchschnittlich über 100 Durchläufe): {execution_time / 100} Sekunden.")

    # Speichermessung mit memory_profiler
    tensor = torch.randn(1000000, 1, 3, 3)  # Tensor für memory_usage und cProfile
    mem_usage = memory_usage((generate_pruning_mask_optimized, (tensor, 0.7)), max_iterations=1, retval=True)
    print(f"Maximaler Speicherverbrauch: {max(mem_usage[0])} MiB")

    # Performance-Analyse mit cProfile
    cProfile.runctx("generate_pruning_mask_optimized(tensor, 0.7)", globals(), locals())

if __name__ == "__main__":
    main()
