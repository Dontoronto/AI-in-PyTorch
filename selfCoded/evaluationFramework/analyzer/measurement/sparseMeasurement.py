import logging

logger = logging.getLogger(__name__)

import torch

@staticmethod
def pruningCounter(model):

    # holds all pruning data of every layer
    pruning_list = dict()

    for name, module in model.named_modules():


        if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:

            temp = dict()

            # Access the weight tensor directly
            weight_tensor = module.weight.data

            # Count zeros and total weights in the weight tensor
            zeros_count = torch.eq(weight_tensor, 0).sum().item()
            total_weights = weight_tensor.numel()
            zero_weights_percentage = (zeros_count / total_weights) * 100

            temp['zeros'] = zeros_count
            temp['total_weights'] = total_weights
            temp['zero_weights_percentage'] = zero_weights_percentage
            pruning_list[name] = temp
            #temp = None

            # Print module information along with zero weight analysis
            logger.info(f"Module: {name}, Zero weights: {zeros_count}/{total_weights} "
                        f"({zero_weights_percentage:.2f}%)")

    return pruning_list
