import copy

from .defaultTrainer import DefaultTrainer
from .admm_utils.utils import (create_magnitude_pruned_mask, add_tensors_inplace, subtract_tensors_inplace,
                               scale_and_add_tensors_inplace)
from .admm_utils.layerInfo import LayerInfo, ADMMVariable

import torch
import os
import json
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import logging

logger = logging.getLogger(__name__)




# TODO: Abhängigkeiten noch nicht fix
# TODO: Analyzer evtl. übergeben mit Preconfigured settings oder
# TODO: DefaultTrainer(..,Analyzer.defaultTrainingMode() -> "Configured Analyzer",..)

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

        # penalty constant for admm
        self.rho = None

        # clip_gradient_threshold value
        self.gradient_threshold = None

        # Note: in caffe admm-nn they used batch_size to normalize model
        self.batch_size_norm_coeff = None
        # Note: regularizatoin l1 or l2
        self.regularization_l2_norm_enabled = None
        self.regularization_l_norm_decay = None

        # layers to be pruned
        self.list_W = []
        # Dual Variable layers
        self.list_U = []
        # Auxiliary Variable layers
        self.list_Z = []

        # layer masks
        self.list_masks = []

        # main iterations
        self.main_iterations = None

        # admm itertations for updates of u and z
        self.admm_iterations = None

    def setADMMConfig(self, kwargs):
        logger.info("ADMM Config was loaded into ADMMTrainer")
        self.admmConfig = kwargs
        # logger.critical(kwargs['trainer'])
        if kwargs.get("admm_trainer").get("main_iterations") is not None:
            self.main_iterations = kwargs.get("admm_trainer").get("main_iterations")
        if kwargs.get("admm_trainer").get("admm_iterations") is not None:
            self.admm_iterations = kwargs.get("admm_trainer").get("admm_iterations")
        if kwargs.get("admm_trainer").get("rho") is not None:
            self.rho = kwargs.get("admm_trainer").get("admm_iterations")
        if kwargs.get("admm_trainer").get("gradient_threshold") is not None:
            self.gradient_threshold = kwargs.get("admm_trainer").get("gradient_threshold")

        # Note: custom batch_size for normalizing
        if kwargs.get("admm_trainer").get("batch_size") is not None:
            self.batch_size_norm_coeff = 1/kwargs.get("admm_trainer").get("batch_size")
        # Note: regularization with l1 or l2 norm
        if kwargs.get("admm_trainer").get("regularization_l2_norm_enabled") is not None:
            self.regularization_l2_norm_enabled = kwargs.get("admm_trainer").get("regularization_l2_norm_enabled")
        if kwargs.get("admm_trainer").get("regularization_l_norm_decay") is not None:
            self.regularization_l_norm_decay = kwargs.get("admm_trainer").get("regularization_l_norm_decay")

    def setADMMArchitectureConfig(self, kwargs):
        logger.info("ADMM Architecture Config was loaded into ADMMTrainer")
        self.admmArchitectureConfig = kwargs
        for val in self.admmArchitectureConfig:
            if val['sparsity'] != None:
                for name_module, module in self.model.named_modules():
                    if name_module == val['op_names']:
                        for name_param, param in self.model.named_parameters():
                            if name_param == name_module + ".weight":
                                self.list_W.append(LayerInfo(name_module, module, param, val['sparsity']))
                                break
                        break

            else:
                continue
        #logger.critical(self.list_W)
        #logger.critical(self.list_W[0].module.weight.data)

    # TODO: needs to be deleted at the end
    def testZCopy(self):
        for module_name, module in self.model.named_modules():
            if module_name == "model.conv1":
                weight_copy = copy.deepcopy(module.weight.data)
                #logger.critical(self.list_W[0].W)
                #self.list_W[0].module.weight.data += 0.1
                self.list_W[0].W += 0.1
                logger.critical(f"Model-Layer{torch.unique(module.weight.data - self.list_W[0].W)}")
                logger.critical(f"Deepcopy-Layer{torch.unique(weight_copy - self.list_W[0].W)}")
                logger.critical(f"Layer Shape: {self.list_W[0].W.shape}")
                logger.critical(f"Difference Model-Layer: {(module.weight.data - self.list_W[0].W).sum()}")
                logger.critical(f"Difference Deepcopy-Layer: {(weight_copy - self.list_W[0].W).sum()}")
        for module_name, param in self.model.named_parameters():
            if module_name == "model.conv1.weight":
                #logger.critical(self.list_W[0].W)
                grad_copy = copy.deepcopy(param.grad)
                #self.list_W[0].module.weight.data += 0.1
                self.list_W[0].dW += 0.1
                logger.critical(f"Model-Layer{torch.unique(param.grad - self.list_W[0].dW)}")
                logger.critical(f"Deepcopy-Layer{torch.unique(grad_copy - self.list_W[0].dW)}")
                logger.critical(f"Layer Shape: {self.list_W[0].dW.shape}")
                logger.critical(f"Difference Model-Layer: {(param.grad - self.list_W[0].dW).sum()}")
                logger.critical(f"Difference Deepcopy-Layer: {(grad_copy - self.list_W[0].dW).sum()}")


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

    def initialize_dualvar_auxvar(self):
        """
        Create two lists containing shallow copies of instances from the original list.
        :param original_list: List of instances with a make_copy method.
        :return: Two lists, each containing copies of the original instances.
        """
        # Create/Assign two lists using list comprehensions
        self.list_U = [copy.copy(instance) for instance in self.list_W]
        self.list_Z = [copy.copy(instance) for instance in self.list_W]

        # Change Settings of each layer in the lists
        [layer_W.set_admm_vars(ADMMVariable.W) for layer_W in self.list_W]
        [layer_U.set_admm_vars(ADMMVariable.U) for layer_U in self.list_U]
        [layer_Z.set_admm_vars(ADMMVariable.Z) for layer_Z in self.list_Z]
        logger.info(f"Layer lists were created")

    def initialize_pruning_mask_layer_list(self):
        '''
        This method should be general. You only have to change the implementation to get the right mask.
        Assings the class variable list_masks the pruning masks per layer on same index level.
        '''
        self.list_masks = [create_magnitude_pruned_mask(layerZ.Z, layerZ.sparsity) for layerZ in self.list_Z]
        logger.info(f"Pruning mask was created")

    def clip_gradients(self):
        if self.gradient_threshold is not None:
            '''
            with json file you can set a threshold value you can use it on whole model
            or change the method to just use it on the to prune -> needs modification
            atm it is used on whole model
            Note: you can use norm_type: float = 2.0 
            to have you own l-norm inside
            '''
            clip_grad_norm_(self.model.parameters(), self.gradient_threshold)
            #logger.info("Gradients will be clipped")

    # Note: this is a weird normalization function of admm-nn repo with batch_size
    def normalize_gradients(self):
        '''
        Normalizes model gradients layerwise with batch size
        :return:
        '''
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad *= self.batch_size_norm_coeff
        #logger.info(f"Gradients were Normalized with: {self.batch_size_norm_coeff}")

    def regularize_gradients(self):
        '''
        this regularizes the model gradients on l1 or l2 norm
        '''
        #logger.info(f"Gradients will be normalized with l2-norm={self.regularization_l2_norm_enabled}")
        if self.regularization_l2_norm_enabled:
            for param in self.model.parameters():
                if param.requires_grad:
                    param.grad += self.regularization_l_norm_decay * param.data
        else:
            for param in self.model.parameters():
                if param.requires_grad:
                    param.grad += self.regularization_l_norm_decay * torch.sign(param.data)

    def _add_layerW_layerU_tolayerZ(self):
        '''
        This method iterates over 3 Lists(layer classes) on same index level and adds the specific tensor vars
        Projection follows the formula Z^k = W^k + U^k
        :param target_list: List of layerobjects which receive the equation
        :param listA:  list of layerobjects which are term of summation
        :param listB:  list of layerobjects which are term of summation
        '''
        for layerZ, layerW, layerU in zip(self.list_Z, self.list_W, self.list_U):
            add_tensors_inplace(layerZ.Z, layerW.W, layerU.U)

    def _update_layerU_with_layerW_layerZ(self):
        '''
        This updates the U layer according to U^k = U^(k-1)+W^k-Z^k
        !!! Keep in mind not to mixup argument sequence, second argument in these functions will be copyed into target
        and then get added/subtracted by the third argument!!! (mixing up results in wrong values)
        :return:
        '''
        for layerZ, layerW, layerU in zip(self.list_Z, self.list_W, self.list_U):
            add_tensors_inplace(layerU.U, layerU.U, layerW.W)
            subtract_tensors_inplace(layerU.U, layerU.U, layerZ.Z)

    # TODO: vllt schon allgemeine Methode wird ja immer so ablaufen?? vllt ändern
    def _prune_layerZ_with_layerMask(self):
        '''
        Applies pre-existing pruning masks to the tensors in self.list_Z in-place.
        '''
        # Ensure that the list of masks matches the list of tensors in length
        if len(self.list_masks) != len(self.list_Z):
            raise ValueError("The length of pruned_masks_list must match the length of self.list_Z.")

        for layerZ, mask in zip(self.list_Z, self.list_masks):
            # Apply the mask in-place to prune the tensor
            layerZ.Z.mul_(mask)

    # TODO: vllt schon allgemeine Methode wird ja immer so ablaufen?? vllt ändern
    def _prune_layerW_with_layerMask(self):
        '''
        Applies pre-existing pruning masks to the weights/gradients in self.list_W in-place.
        '''
        # Ensure that the list of masks matches the list of tensors in length
        if len(self.list_masks) != len(self.list_Z):
            raise ValueError("The length of pruned_masks_list must match the length of self.list_W.")

        for layerW, mask in zip(self.list_W, self.list_masks):
            # Apply the mask in-place to prune the tensor
            layerW.W.mul_(mask)
            layerW.dW.mul_(mask)

    def _update_layerdW(self):
        '''
        This updates the dW layer according to dW^(k+1) = dW^k + rho*W^k + rho*U^k - rho*Z^k
        !!! Keep in mind not to mixup argument sequence, second argument in these functions will be copyed into target
        and then get added/subtracted by the third argument!!! (mixing up results in wrong values)
        :return:
        '''
        for layerZ, layerW, layerU in zip(self.list_Z, self.list_W, self.list_U):
            scale_and_add_tensors_inplace(layerW.dW, self.rho, layerW.W, layerW.dW) # dW^(k+1/3) = dW^k + rho*W^k
            scale_and_add_tensors_inplace(layerW.dW, -self.rho, layerZ.Z, layerW.dW) # dW^(k+2/3) = dW^(k+1/3) -rho*Z^k
            scale_and_add_tensors_inplace(layerW.dW, self.rho, layerU.U, layerW.dW) # dW^(k+1) = dW^(k+2/3) + rho*U

    def project_aux_layers(self):
        '''
        This method should be general. Changes in projecting of auxiliary layers (Z) should be inserted here
        '''
        self._add_layerW_layerU_tolayerZ()
        #logger.info(f"Auxiliary Layers were projected by ADMM")

    # TODO: vllt schon allgemeine Methode wird ja immer so ablaufen?? vllt ändern
    def prune_aux_layers(self):
        """
        This method should be general. It prunes the layerZ with according to the layerMask through simple
        multiplication
        """
        self._prune_layerZ_with_layerMask()
        #logger.info(f"Auxiliary layers were pruned by ADMM")

    # TODO: vllt schon allgemeine Methode wird ja immer so ablaufen?? vllt ändern
    def prune_weight_layer(self):
        """
        This method should be general. It prunes the layerW with according to the layerMask through simple
        multiplication
        """
        self._prune_layerW_with_layerMask()
        #logger.info("Weight layer was pruned by ADMM")

    def update_dual_layers(self):
        '''
        This method should be general. Changes in updating the Dual Layers (U) should ge inserted here
        '''
        self._update_layerU_with_layerW_layerZ()
        #logger.info("Dual Layers were updated by ADMM")

    def solve_admm(self):
        '''
        This method should be general. Changes in updating the Weight Layers (W) should ge inserted here
        '''
        self._update_layerdW()
        #logger.info("Gradients were updated by ADMM")





    # TODO: weiß nichtmehr genau aber einfach im Kopf behalten
    def admm(self, curr_iteration):
        #logging.disable(logging.WARNING)
        self.clip_gradients()
        self.normalize_gradients()
        self.regularize_gradients()
        if curr_iteration % self.admm_iterations == 0:
            logger.critical(curr_iteration)
            if curr_iteration > 8100:
                exit()
            self.initialize_pruning_mask_layer_list()
            self.project_aux_layers()
            self.prune_aux_layers()
            if curr_iteration != 0:
                self.update_dual_layers()
        self.solve_admm()
        pass

    def train(self, test=False):
        self.model.train()
        self.preTrainingChecks()

        self.initialize_dualvar_auxvar()

        dataloader = self.createDataLoader(self.dataset)

        counter = 0
        if test is True:
            self.prepareDataset(testset=True)
            test_loader = self.createDataLoader(self.testset)
            test_loader.shuffle = False
        for epo in range(self.epoch):
            for batch_idx, (data, target) in enumerate(dataloader):
                if counter > self.main_iterations:
                    break
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                self.admm(counter)
                self.optimizer.step()
                counter +=1
                if batch_idx % 100 == 0:
                    print(f'Train Epoch: {epo} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

            if test is True:
                self.test(test_loader, snapshot_enabled=self.snapshot_enabled, current_epoch=epo)


# TODO: Normalization like this with size of batchsize
# iteration_size = len(data_loader) # or any specific iteration size you have in mind
#
# for inputs, targets in data_loader:
#     optimizer.zero_grad() # Zero the gradients at the start of the batch
#     outputs = model(inputs) # Forward pass
#     loss = loss_function(outputs, targets) # Compute the loss
#     loss.backward() # Backward pass to calculate gradients
#
#     # Normalize gradients
#     for param in model.parameters():
#         if param.grad is not None:
#             param.grad /= iteration_size















