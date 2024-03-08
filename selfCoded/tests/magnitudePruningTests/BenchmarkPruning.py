import torch
import cProfile
from timeit import timeit
from memory_profiler import memory_usage

def generate_pruning_mask_optimized(tensor, ratio):
    # Schritt 1: Berechnung der Anzahl der zu behaltenden Elemente
    total_elements = tensor.numel()
    num_elements_to_keep = round(total_elements * (1 - ratio))

    # Schritt 2: Flatten des Tensors und Sortierung der absoluten Werte
    flat_tensor = tensor.flatten().abs()
    sorted_values, _ = torch.sort(flat_tensor, descending=True)

    # Schritt 3: Bestimmung der Pruning-Schwelle
    threshold = sorted_values[num_elements_to_keep - 1] if num_elements_to_keep > 0 else sorted_values[0]

    # Schritt 4: Erstellen der Maske
    mask = tensor.abs() >= threshold

    return mask.type_as(tensor)

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
