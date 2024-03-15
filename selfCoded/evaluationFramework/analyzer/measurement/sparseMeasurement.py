import logging

logger = logging.getLogger(__name__)

import torch

@staticmethod
def pruningCounter(model):
    # zeros_count = (tensor == 0).sum().item()
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
            # Access the weight tensor directly
            weight_tensor = module.weight.data

            # Count zeros and total weights in the weight tensor
            zeros_count = torch.eq(weight_tensor, 0).sum().item()
            total_weights = weight_tensor.numel()
            zero_weights_percentage = (zeros_count / total_weights) * 100

            # Print module information along with zero weight analysis
            logger.info(f"Module: {name}, Zero weights: {zeros_count}/{total_weights} "
                        f"({zero_weights_percentage:.2f}%)")
    # for name, parameter in model.named_parameters():
    #     if parameter.requires_grad:
    #         # Count zeros and total weights
    #         zeros_count = torch.eq(parameter, 0).sum().item()
    #         total_weights = parameter.numel()
    #         zero_weights_percentage = (zeros_count / total_weights) * 100
    #
    #         # Print layer information
    #         logger.info(f"Layer: {name}, Zero weights: {total_weights}/{zeros_count} "
    #                     f"({zero_weights_percentage:.2f}%)")