import os
from torch.nn import Conv2d, Linear
import logging
import json

logger = logging.getLogger(__name__)


class ModelArchitectureExtractor():

    @staticmethod
    def extractLayers(model, folderName):
        # Check if the folder exists, create it if not
        # folderName = "configs/preOptimizingTuning/model_architecture"
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        config_list = []
        for name, module in model.named_modules():
            if isinstance(module, Conv2d):
                config_list.append({
                    'sparsity': None,
                    'op_types': 'Conv2d',
                    'op_names': name
                })
                logger.info(f"Layer Name: {name} was extracted.")
            elif isinstance(module, Linear):
                config_list.append({
                    'sparsity': None,
                    'op_types': 'Linear',
                    'op_names': name
                })
                logger.info(f"Layer Name: {name} was extracted.")
        with open(folderName + "/ADMModelArchitectureTest.json", 'w') as file:
            json.dump(config_list, file, indent=4)
        logger.info(f"Architecture extracted to folder: {folderName}")
        logger.info(f"Architecture file in {folderName} need to be extended with sparsity and moved to upper folder")