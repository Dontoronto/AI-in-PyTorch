import torch

import patternManagerv2

layer_list = [torch.randn(6, 1, 3, 3), torch.randn(16, 6, 3, 3), torch.randn(1, 6, 3, 3)]

pruning_ratio = [0.5, 0.7, 0.8]

patternManger = patternManagerv2.PatternManager()

patternManger.assign_patterns_to_tensors(layer_list, pruning_ratio)

masks = patternManger.get_pattern_masks()

for mask in masks:
    print(mask.shape)

for i in range(140):
    patternManger.update_pattern_assignments(layer_list,4, pruning_ratio)

masks = patternManger.get_pattern_masks()

for mask in masks:
    print(mask.shape)

patternManger.save_available_patterns("available_patterns.txt")

