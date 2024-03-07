import copy
from enum import Enum, auto

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


class ADMMVariable(Enum):
    '''
    input for Layerclass to choose which variables should be initialized/ removed
    '''
    W = auto()
    Z = auto()
    U = auto()


# TODO: when iteration needs update of variables U and Z we need
# TODO: to create special functions for this be aware of references
class LayerInfo:
    '''
    This class is a wrapper for a layer which should be modifyed with admm.
    Wrapper is the same for every class. Depending on the enum value other variables will be named
    For example W, U and Z (for ADMM)
    '''
    def __init__(self, name, module, param):
        self.module = module
        self.name = name
        self.param = param
        self.type = type(module).__name__

        # TODO: entscheiden ob ich Gewicht speichern will/ wir hier gewicht extrahiert oder referenz
        # TODO: soll ich flag machen für das Prüfen ob weight neu gesetzt wurde und deshalb aktuallisiert werden muss
        # TODO: oder weglassen und jedes mal neu laden...
        #self.W = None
        #self.dW = None
        self.U = None
        self.Z = None
        self.state = None

    # TODO: implement initialization of all variables
    # TODO: think about a system to easily automate the set process for a list of these classes
    def set_admm_vars(self, ADMM_ENUM: Enum):
        # Reset variables to None before initialization
        #self.W = None
        #self.dW = None
        self.U = None
        self.Z = None
        self.state = ADMM_ENUM

        if ADMM_ENUM == ADMMVariable.W:
            logger.info(f"Layer was set as Weight Layer 'W' with Weight and Gradient")
        elif ADMM_ENUM == ADMMVariable.U:
            self.U = self.module.weight.data.clone().detach().zero_()
            logger.info(f"Layer was set as Dual Variable Layer 'U' with same shape as weights")
        elif ADMM_ENUM == ADMMVariable.Z:
            self.Z = self.module.weight.data.clone().detach()
            logger.info(f"Layer was set as Auxiliary Variable Layer 'Z' copy of Weights")

    # W and dW properties
    @property
    def W(self):
        if self.state == ADMMVariable.W:
            return self.module.weight.data
        else:
            logger.warning(f"GET not possible instance is not configured as Weight Layer: {self}")
            return None

    @W.setter
    def W(self, value):
        if self.state == ADMMVariable.W:
            self.module.weight.data = value
        else:
            logger.warning(f"SET not possible instance is not configured as Weight Layer: {self}")

    @property
    def dW(self):
        if self.state == ADMMVariable.W:
            return self.param.grad
        else:
            logger.warning(f"GET not possible instance is not configured as Weight Layer: {self}")
            return None

    @dW.setter
    def dW(self, value):
        if self.state == ADMMVariable.W:
            self.param.grad = value
        else:
            logger.warning(f"SET not possible instance is not configured as Weight Layer: {self}")

    def make_copy(self):
        '''
        Use copy.copy to create a shallow copy of the instance
        :return: copy object without references to the model layers
        '''
        return copy.copy(self)

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
        #logger.critical(self.pruningLayers)
        #logger.critical(self.pruningLayers[0].module.weight.data)

    # TODO: soll ich classen variable machen oder einfach so zurück geben?
    # TODO: soll das in der Klasse hier sein oder eine methode außerhalb der Klasse?
    def create_copies(self):
        """
        Create two lists containing shallow copies of instances from the original list.

        :param original_list: List of instances with a make_copy method.
        :return: Two lists, each containing copies of the original instances.
        """
        # Create two lists using list comprehensions
        list_copy1 = [copy.copy(instance) for instance in self.pruningLayers]
        list_copy2 = [copy.copy(instance) for instance in self.pruningLayers]

        # Return both lists
        return list_copy1, list_copy2

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
                    #'configuration': str(module)
                })
                logger.info(f"Layer Name: {name} was extracted.")
            elif isinstance(module, nn.Linear):
                config_list.append({
                    'sparsity': None,
                    'op_types': 'Linear',
                    'op_names': name
                    #'configuration': str(module)
                })
                logger.info(f"Layer Name: {name} was extracted.")
        with open(folderName + "/ADMMModelArchitecture.json", 'w') as file:
            json.dump(config_list, file, indent=4)
        logger.info(f"Architecture extracted to folder: {folderName}")
        logger.info(f"Architecture file in {folderName} need to be extended with sparsity and moved to upper folder")

    def admmFilter(self):
        pass

    def train(self, test=False):
        self.model.train()
        self.preTrainingChecks()
        dataloader = self.createDataLoader(self.dataset)
        if test is True:
            self.prepareDataset(testset=True)
            test_loader = self.createDataLoader(self.testset)
            test_loader.shuffle = False
        for epo in range(self.epoch):
            for batch_idx, (data, target) in enumerate(dataloader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                break
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    print(f'Train Epoch: {epo} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

        for i in self.pruningLayers:
            logger.critical(f"Instance: {i} -> W: {(i.W is not None)}, dW: {(i.dW is not None)}, U: {(i.U is not None)}, Z: {(i.Z is not None)}")
            i.set_admm_vars(ADMMVariable.Z)
            logger.critical(f"Instance: {i} -> W: {(i.W is not None)}, dW: {(i.dW is not None)}, U: {(i.U is not None)}, Z: {(i.Z is not None)}")

    # def train(self, test = False):
    #     self.preTrainingChecks()
    #     dataloader = self.createDataLoader(self.dataset)
    #     self.model.train()
    #
    #     count = 0
    #     for i in range(self.epoch):
    #         for batch, (X, y) in enumerate(dataloader):
    #
    #             if count == self.main_iterations:
    #                 return
    #
    #             # remove existing settings
    #             self.optimizer.zero_grad()
    #
    #             # Compute prediction and loss
    #             pred = self.model(X)
    #             loss = self.loss(pred, y)
    #
    #             # Backpropagation
    #             loss.backward()
    #
    #             # here should the logic of admm cycle be located
    #             self.admmFilter()
    #
    #             # Apply optimization with gradients
    #             self.optimizer.step()
    #
    #             if batch % 2 == 0:
    #                 loss, current = loss.item(), batch * len(X)
    #                 print(f"Epoch number: {i}")
    #                 print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")
    #
    #             # if it hits main_iterations count it will end the admm training
    #             count += 1
    #
    #
    #     if test is True:
    #         self.test()
    #
    #     pass

