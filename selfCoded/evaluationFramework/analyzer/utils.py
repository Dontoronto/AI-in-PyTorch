import torch
import logging


logger = logging.getLogger(__name__)

def weight_export(model, layer_name, path):

    for name, module in model.named_modules():
        #name == layer_name and

        if (hasattr(module, 'weight') and
                module.weight is not None and module.weight.requires_grad):

            # Access the weight tensor directly
            weight_tensor = module.weight.data

            torch.save(weight_tensor, path)
            #print(f"shape of layers = {weight_tensor.shape}")

            return


