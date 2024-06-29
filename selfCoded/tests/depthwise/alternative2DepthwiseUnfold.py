import torch
import torch.nn as nn

# Eingabebild (Batchgröße, Kanäle, Höhe, Breite)
input = torch.randn(1, 3, 5, 5)  # Beispielbild

# Faltungsparameter
in_channels = 3
out_channels = 2
kernel_size = 3
stride = 1
padding = 1

# Definiere die 2D-Faltung
conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

# Anwenden der 2D-Faltung
output_conv2d = conv2d(input)

# Extrahieren der Faltungsgewichte und des Bias
weight = conv2d.weight
bias = conv2d.bias

# Unfolding des Eingangsbildes
unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
input_unfolded = unfold(input)

# Reshape der Faltungsgewichte für die Batch-Verarbeitung
weight_reshaped = weight.view(out_channels, -1)

# Anzahl der Patches
num_patches = input_unfolded.size(-1)

# Bereite die Ausgabematrix vor
output_unfolded = torch.zeros((input.size(0), out_channels, num_patches))

# Berechne jedes Patch separat
for i in range(num_patches):
    patch = input_unfolded[..., i]  # Patch extrahieren
    patch = patch.view(input.size(0), -1)  # Reshape des Patches
    output_unfolded[..., i] = (weight_reshaped @ patch.T).T  # Matrixmultiplikation

# Hinzufügen des Bias
output_unfolded += bias.view(1, -1, 1)

# Reshape der Ausgabe in die gewünschte Form (Batchgröße, Kanäle, Höhe, Breite)
output_height = (input.size(2) + 2 * padding - kernel_size) // stride + 1
output_width = (input.size(3) + 2 * padding - kernel_size) // stride + 1
output = output_unfolded.view(input.size(0), out_channels, output_height, output_width)

# Aufsummierung der gefalteten Teile
output_folded = torch.zeros_like(output_conv2d)
for i in range(output_height):
    for j in range(output_width):
        patch_index = i * output_width + j
        output_folded[:, :, i, j] = output_unfolded[:, :, patch_index]

# Vergleichen der Ergebnisse
print("Output mit conv2d:\n", output_conv2d)
print("Output mit Unfold und Aufsummierung:\n", output_folded)
print("Sind die Ausgaben gleich?:", torch.allclose(output_conv2d, output_folded, atol=1e-6))
