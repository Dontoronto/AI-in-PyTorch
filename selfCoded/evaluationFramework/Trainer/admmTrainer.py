import copy

from .defaultTrainer import DefaultTrainer
from torch.utils.data import DataLoader

import torch
import os
import json
import torch.nn as nn
import logging
import functools
logger = logging.getLogger(__name__)




# TODO: Abhängigkeiten noch nicht fix
# TODO: Analyzer evtl. übergeben mit Preconfigured settings oder
# TODO: DefaultTrainer(..,Analyzer.defaultTrainingMode() -> "Configured Analyzer",..)


# TODO: this class doesn't need changes atm because there are no biases used inside of the layer just in batchnormal
class LayerInfo:
    def __init__(self, name, module, param):
        self.module = module
        self.name = name
        self.param = param
        self.type = type(module).__name__

        # TODO: entscheiden ob ich Gewicht speichern will/ wir hier gewicht extrahiert oder referenz
        # TODO: soll ich flag machen für das Prüfen ob weight neu gesetzt wurde und deshalb aktuallisiert werden muss
        # TODO: oder weglassen und jedes mal neu laden...
        self.W = self.module.weight

    # this method is shit. Thinking about the situation but even though shit
    # def _setattr(model, name, module):
    #     name_list = name.split(".")
    #     for name in name_list[:-1]:
    #         model = getattr(model, name)
    #     setattr(model, name_list[-1], module)

# class LayerInfoGrad:
#     def __init__(self, name, module, param):
#         self.module = module
#         self.name = name
#         self.type = type(module).__name__
#         self.grad = param.grad
#
#     def _setattr(model, name, module):
#         name_list = name.split(".")
#         for name in name_list[:-1]:
#             model = getattr(model, name)
#         setattr(model, name_list[-1], module)

class ADMMTrainer(DefaultTrainer):
    def __init__(self,
                 model,
                 dataHandler,
                 loss,
                 optimizer,
                 epoch=1
                 ):
        super().__init__(model, dataHandler, loss, optimizer, epoch)

        # Variables for ADMM config
        self.admmConfig = None

        # Variables for ADMM Architecture Config
        self.admmArchitectureConfig = None

        # layers to be pruned
        self.pruningLayers = []

        # main iterations
        self.main_iterations = None

        # admm itertations for updates of u and z
        self.admm_iterations = None

    def setADMMConfig(self, kwargs):
        logger.info("ADMM Config was loaded into ADMMTrainer")
        self.admmConfig = kwargs
        # logger.critical(kwargs['trainer'])
        if kwargs.get("trainer").get("main_iterations") is not None:
            self.main_iterations = kwargs.get("trainer").get("main_iterations")
        if kwargs.get("trainer").get("admm_iterations") is not None:
            self.admm_iterations = kwargs.get("trainer").get("admm_iterations")

    def setADMMArchitectureConfig(self, kwargs):
        logger.info("ADMM Architecture Config was loaded into ADMMTrainer")
        self.admmArchitectureConfig = kwargs
        for val in self.admmArchitectureConfig:
            if val['sparsity'] != None:
                for name_module, module in self.model.named_modules():
                    if name_module == val['op_names']:
                        for name_param, param in self.model.named_parameters():
                            if name_param == name_module + ".weight":
                                self.pruningLayers.append(LayerInfo(name_module, module, param))
                                break
                        break

            else:
                continue
        logger.critical(self.pruningLayers)

    # def testGradientModification(self):
    #     for name, param in self.model.named_parameters():
    #         if name == "model.conv1.weight":
    #             before = copy.deepcopy(param.data)
    #             for module_name, module in self.model.named_modules():
    #                 if module_name == "model.conv1":
    #                     test_obj = LayerInfoGrad(name, module, param)
    #                     test_obj.module.weight.data += 0.1
    #                     logger.critical(torch.unique(before - test_obj.module.weight.data))
    #                     logger.critical(test_obj.grad.shape)
    #                     logger.critical(test_obj.module.weight.data.shape)
    #                     return

    # TODO: needs to be deleted at the end
    def testZCopy(self):
        test_z = copy.deepcopy(self.pruningLayers)
        for module_name, module in self.model.named_modules():
            if module_name == "model.conv1":
                logger.critical(self.pruningLayers[0].module.weight.data)
                #self.pruningLayers[0].module.weight.data += 0.1
                test_z[0].module.weight.data += 0.1
                logger.critical(torch.unique(module.weight.data - self.pruningLayers[0].module.weight.data))
                logger.critical(torch.unique(test_z[0].module.weight.data - self.pruningLayers[0].module.weight.data))
                logger.critical(torch.unique(test_z[0].module.weight.data - module.weight.data))

    def layerArchitectureExtractor(self):
        # Check if the folder exists, create it if not
        folderName = "configs/preOptimizingTuning/model_architecture"
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        config_list = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                config_list.append({
                    'sparsity': None,
                    'op_types': 'Conv2d',
                    'op_names': name
                })
                logger.info(f"Layer Name: {name} was extracted.")
        with open(folderName + "/ADMMModelArchitecture.json", 'w') as file:
            json.dump(config_list, file, indent=4)
        logger.info(f"Architecture extracted to folder: {folderName}")
        logger.info(f"Architecture file in {folderName} need to be extended with sparsity and moved to upper folder")

    def admmFilter(self):
        pass

    def train(self, test = False):
        self.preTrainingChecks()
        dataloader = self.createDataLoader(self.dataset)
        self.model.train()

        count = 0
        for i in range(self.epoch):
            for batch, (X, y) in enumerate(dataloader):

                if count == self.main_iterations:
                    return

                # remove existing settings
                self.optimizer.zero_grad()

                # Compute prediction and loss
                pred = self.model(X)
                loss = self.loss(pred, y)

                # Backpropagation
                loss.backward()

                # here should the logic of admm cycle be located
                self.admmFilter()

                # Apply optimization with gradients
                self.optimizer.step()

                if batch % 2 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"Epoch number: {i}")
                    print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")

                # if it hits main_iterations count it will end the admm training
                count += 1


        if test is True:
            self.test()

        pass

