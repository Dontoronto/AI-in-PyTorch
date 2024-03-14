import logging

logger = logging.getLogger(__name__)

import torch

@staticmethod
def pruningCounter(model):
    # zeros_count = (tensor == 0).sum().item()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            # Count zeros and total weights
            zeros_count = torch.eq(parameter, 0).sum().item()
            total_weights = parameter.numel()
            zero_weights_percentage = (zeros_count / total_weights) * 100

            # Print layer information
            logger.info(f"Layer: {name}, Zero weights: {total_weights}/{zeros_count} "
                        f"({zero_weights_percentage:.2f}%)")