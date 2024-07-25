import copy
import csv
import os

from .utils import tqdm_progressbar

from .defaultTrainer import DefaultTrainer
from .admm_utils.utils import (create_magnitude_pruned_mask, add_tensors_inplace, subtract_tensors_inplace,
                               scale_and_add_tensors_inplace)
from .admm_utils.maskLoader import load_pruning_mask_csv
from .admm_utils.layerInfo import ADMMVariable
from .mapper.admm_mapper import ADMMConfigMapper, ADMMArchitectureConfigMapper
from .admm_utils.multiProcessHandler import MultiProcessHandler
from .admm_utils.tensorBuffer import TensorBuffer
from .admm_utils.pattern_pruning.patternManager import PatternManager

import torch
from torch.nn import Conv2d, Linear
from torch.nn.utils import clip_grad_norm_
from multiprocessing import Process, Queue
import logging
from multiprocessing import Event
from torch.utils.tensorboard import SummaryWriter

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

        # phases of iterating
        self.phase_list = []

        # flag for saving model afterwards
        self.save = False
        self.save_path = None

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

        # mask creation mode, dynamic is True, static is False
        self.dynamic_masking = True

        # admm itertations for updates of u and z
        self.admm_iterations = None

        # TODO: safe epsilone history for post evaluating
        self.epsilon_W = None
        self.epsilon_Z = None
        self.threshold_warmup = 0.01                         # start calc. termination criterion with delay
        # stores old layerlist Z for termination criterion
        self.old_layer_list_z = None
        self.history_epsilon_W = []
        self.history_epsilon_Z = []
        self.early_termination_flag = False

        #self.patternManager = PatternManager()
        self.patternManager = None
        self.unstructured_magnitude_pruning_enabled = False
        self.pattern_pruning_all_patterns_enabled = False
        self.pattern_pruning_elog_patterns_enabled = False
        self.connectivity_pruning_enabled = False
        #self.patternManager.setConnectivityPruning(True)

        #loading existing pruning mask
        self.pruning_mask_loading_enabled = False
        self.pruning_mask_loading_path = None

        self.tensor_buffering_enabled = False
        self.onnx_enabled = False

        # TODO: not set in configs
        self.adv_attacker = None
        self.adv_enabled = False
        self.adv_fraction = None
        self.adv_test_enabled = None
        self.tensorboard_writer = None

        logger.debug(f"ADMM Trainer was initialized: \n {self.__dict__}")


    def checkPruningTypes(self):
        if self.pattern_pruning_elog_patterns_enabled and self.pattern_pruning_all_patterns_enabled:
            logger.critical(f"two pattern pruning types are selected, please deactivate one in ADMMConfig.json")
            return

        if self.pattern_pruning_elog_patterns_enabled != self.pattern_pruning_all_patterns_enabled:
            logger.debug("Pattern Manager was initialized. Pattern Pruning was enabled in ADMMConfig.json")
            if self.pattern_pruning_elog_patterns_enabled is True:
                self.patternManager = PatternManager(pattern_library='elog')
            else:
                self.patternManager = PatternManager(pattern_library='all')

            if self.connectivity_pruning_enabled is True:
                logger.debug("Connectivity Pruning was enabled")
                self.patternManager.setConnectivityPruning(True)


    def setADMMConfig(self, kwargs):
        ADMMConfigMapper(self, kwargs)
        self.checkPruningTypes()

    def setADMMArchitectureConfig(self, kwargs):
        ADMMArchitectureConfigMapper(self, kwargs)

    def getHistoryEpsilonW(self):
        return self.history_epsilon_W

    def getEpsilonResults(self):
        # Check if attributes exist, otherwise return None
        epsilon_W = getattr(self, 'epsilon_W', None)
        epsilon_Z = getattr(self, 'epsilon_Z', None)

        # Check if history lists are empty
        history_epsilon_W = self.getHistoryEpsilonW() if self.getHistoryEpsilonW() else None
        history_epsilon_Z = self.getHistoryEpsilonZ() if self.getHistoryEpsilonZ() else None

        return history_epsilon_W, history_epsilon_Z, epsilon_W, epsilon_Z

    def getHistoryEpsilonZ(self):
        return self.history_epsilon_Z

    # def getEpsilonResults(self):
    #     return self.getHistoryEpsilonW(), self.getHistoryEpsilonZ(), self.epsilon_W, self.epsilon_Z

    def setAdversarialTraining(self, adv_attacker, adv_fraction, adv_test_enabled = False):
        self.adv_attacker = adv_attacker
        self.adv_fraction = adv_fraction
        self.adv_enabled = True
        self.adv_test_enabled = adv_test_enabled

    # def getTestLoader(self):
    #     return super(ADMMTrainer, self).getTestLoader()
    #
    # def getLossFunction(self):
    #     return super(ADMMTrainer, self).getLossFunction()

    # TODO: Mapper erstellen
    def setTensorBufferConfig(self, kwargs):
        self.handler = MultiProcessHandler()
        self.handler.setTensorBufferConfig(kwargs)
        self.handler.setFilePath(self.save_path)

    def changePaths(self, path):
        if self.tensor_buffering_enabled is True:
            logger.warning(f"Changing tensorbuffering path from {self.handler.getFilePath()} to {path}")
            self.handler.setFilePath(path)
        else:
            logger.warning("TensorBuffering is not activated")

        # Changes all paths the Trainer is interacting with
        self.save_path = path

    def setModelName(self, name):
        self.model_name = name


    def load_pruning_mask(self):
        if os.path.exists(self.pruning_mask_loading_path):
            self.list_masks = load_pruning_mask_csv(self.pruning_mask_loading_path)
            if self.cuda_enabled is True:
                for i in range(len(self.list_masks)):
                    self.list_masks[i].to('cuda')

    def show_pruning_layer_job(self):
        '''
        method shows the list of layer which will be pruned with some info
        '''
        logger.info(f"======================= SELECTED PRUNING LAYERS =======================")
        for layer in self.list_W:
            logger.info(f"Layer {layer.name} with shape {layer.W.shape} sparsity-ratio: {layer.sparsity}")
        logger.info(f"=======================================================================")


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
        logger.info(f"Layer lists (ADMM Variables) were initialized")

    # TODO: Methode erstellen welche Pattern pruning mit magnitude Pruning von fc layern zusammenfügt
    # TODO: PruningRate bei connectivity Pruning soll über Architecture config übernommen werden nicht als
    # TODO: hardcoded list values
    # TODO: pruning Ratios sollen dauerhaft gespeichert werden
    # TODO: die logik muss in den Pattern Manager
    def initialize_pruning_mask_layer_list(self, firstTime=True):
        '''
        This method should be general. You only have to change the implementation to get the right mask.
        Assings the class variable list_masks the pruning masks per layer on same index level.
        '''
        # isinstance(module, Conv2d)
        # isinstance(module, Linear)
        # [pattern_library[i] if i is not None else torch.zeros(3, 3) for i in group_indices]
        self.list_masks = []
        if self.pattern_pruning_all_patterns_enabled or self.pattern_pruning_elog_patterns_enabled:
            conv_list = [layerZ.Z for layerZ in self.list_Z if isinstance(layerZ.module, Conv2d)]
            conv_layer_pruning_ratio_list = [layerZ.sparsity for layerZ in self.list_Z if isinstance(layerZ.module, Conv2d)]
            fc_list = [layerZ for layerZ in self.list_Z if isinstance(layerZ.module, Linear)]
            #self.list_masks = []
            if firstTime is True:
                self.patternManager.assign_patterns_to_tensors(conv_list, conv_layer_pruning_ratio_list)
                self.list_masks = self.patternManager.get_pattern_masks()
                self.list_masks.extend([create_magnitude_pruned_mask(layerZ.Z, layerZ.sparsity) for layerZ in fc_list])
            else:
                #self.patternManager.reduce_available_patterns(2)
                self.patternManager.update_pattern_assignments(conv_list, min_amount_indices=4,
                                                               pruning_ratio_list=conv_layer_pruning_ratio_list,
                                                               admm_iter=self.main_iterations//self.admm_iterations)
                self.list_masks = self.patternManager.get_pattern_masks()
                self.list_masks.extend([create_magnitude_pruned_mask(layerZ.Z, layerZ.sparsity) for layerZ in fc_list])
            # self.list_masks = [create_magnitude_pruned_mask(layerZ.Z, layerZ.sparsity) for layerZ in self.list_Z]
            #logger.info(f"Pruning mask was created")
        else:
            self.list_masks.extend([create_magnitude_pruned_mask(layerZ.Z, layerZ.sparsity) for layerZ in self.list_Z])

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
                    param.grad += 2 * self.regularization_l_norm_decay * param.data
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
    # TODO: NOTE: Hier war debug auf if abfrage weiß nicht wieso genauer hinschauen
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
        #self._prune_layerW_with_layerMask()
        #logger.info("Gradients were updated by ADMM")

    def update_termination_criterion(self):
        frob_criterion_W = self._termination_criterion_layer_W()
        frob_criterion_Z = self._termination_criterion_layer_Z()
        self.history_epsilon_W.append(frob_criterion_W)
        self.history_epsilon_Z.append(frob_criterion_Z)
        if frob_criterion_W < self.epsilon_W and frob_criterion_Z < self.epsilon_Z:
            logger.info("Early termination flag was set.")
            self.early_termination_flag = True

    def store_old_AuxVariable(self):
        self.old_layer_list_z = copy.deepcopy(self.list_Z)

    # TODO: terminate ADMM in a good manner
    def _termination_criterion_layer_W(self):
        frobenius_distance = 0

        for layerZ, layerW in zip(self.list_Z, self.list_W):
            differenceW = layerW.W - layerZ.Z
            frobenius_distance += torch.norm(differenceW, p='fro')

        logger.debug(f"Condition for termination criterion W^(k+1)-Z^(k+1)={frobenius_distance}/{self.epsilon_W}")
        return float(frobenius_distance)

    # TODO: terminate ADMM in a good manner
    def _termination_criterion_layer_Z(self):
        frobenius_distance = 0

        for layerZ, old_layerZ in zip(self.list_Z, self.old_layer_list_z):
            differenceW = layerZ.Z - old_layerZ.Z
            frobenius_distance += torch.norm(differenceW, p='fro')

        logger.debug(f"Condition for termination criterion Z^(k+1)-Z^k= {frobenius_distance}/{self.epsilon_Z}")
        return float(frobenius_distance)

    @staticmethod
    def _kernel_tensor_extraction(tensors, batch_index, channel_index):
        """
        Returns a specific (3,3) tensor based on provided indices.

        Parameters:
        - tensors: tensors with shapes like (1,6,3,3) or (16,6,3,3).
        - list_index: Index in the list to select the tensor.
        - batch_index: Index for the batch dimension within the selected tensor.
        - channel_index: Index for the channel dimension within the selected tensor.

        Returns:
        - The specified (3,3) tensor.
        """
        # Validate input indices
        if batch_index < 0 or channel_index < 0:
            raise IndexError(f"batch_index or channel_index is out of bounds {tensors.shape}.")

        selected_tensor = tensors[batch_index, channel_index].clone().detach().numpy()

        return selected_tensor

    def w_z_kernel_weight_extraction(self, list_index, batch_index, channel_index):
        if list_index < 0 or list_index >= len(self.list_W):
            raise IndexError("list_index is out of bounds.")

        w_weight = self._kernel_tensor_extraction(self.list_W[list_index].W, batch_index, channel_index)
        z_weight = self._kernel_tensor_extraction(self.list_Z[list_index].Z, batch_index, channel_index)

        return [w_weight, z_weight]

    def tensorBuffer_saving(self):
        '''
        Just for extracting config and select kernel
        :return:
        '''
        for index, process_id, oi_channel in zip(self.handler.layer_index, self.handler.process_ids,
                                           self.handler.output_input_channel):
            w_z_weight = self.w_z_kernel_weight_extraction(index,oi_channel[1],oi_channel[0])
            self.handler.put_item_in_queue(process_id, w_z_weight)


    # TODO: weiß nichtmehr genau aber einfach im Kopf behalten
    def admm(self, curr_iteration):
        #logging.disable(logging.WARNING)
        self.clip_gradients()
        self.normalize_gradients()
        self.regularize_gradients()
        if curr_iteration % self.admm_iterations == 0:

            if self.tensorboard_writer is not None:
            # Logge die Histogramme der Modellgewichte
                for name, param in self.model.named_parameters():
                    self.tensorboard_writer.add_histogram(name, param, curr_iteration)

            # TODO: abbruch testen
            self.store_old_AuxVariable()
            self.project_aux_layers()
            # if statement für dynamische oder statische maske
            if self.dynamic_masking is True and curr_iteration != 0:
                self.initialize_pruning_mask_layer_list(firstTime=False)
            elif curr_iteration == 0:
                self.initialize_pruning_mask_layer_list(firstTime=True)
            self.prune_aux_layers()
            if curr_iteration != 0:
                if curr_iteration/self.main_iterations > self.threshold_warmup:
                    self.update_termination_criterion()

                self.update_dual_layers()

            # feature progress bar
            # self.pbar_iteration.update(self.admm_iterations)

        self.solve_admm()
        pass

    def retrain(self, curr_iteration):
        self.clip_gradients()
        self.normalize_gradients()
        self.regularize_gradients()
        if curr_iteration == 0 and "admm" not in self.phase_list:
            if self.pruning_mask_loading_enabled is True:
                self.load_pruning_mask()
            else:
                self.initialize_pruning_mask_layer_list(True)
            #self.list_masks = self.patternManager.get_pattern_masks()
        elif curr_iteration % self.admm_iterations == 0:
            if self.tensorboard_writer is not None:
                for name, param in self.model.named_parameters():
                    self.tensorboard_writer.add_histogram(name, param, self.main_iterations + curr_iteration)
        self.prune_weight_layer()


    # TODO: methode umschreiben so dass epoch nichtmehr gebraucht wird für admm
    # TODO: symbiose von ADMM und retraining
    def train(self, test=False, tensorboard=False):

        logger.info(f"Current Trainer Tasks: {self.phase_list}")

        self.model.train()
        self.preTrainingChecks()

        dataloader = self.createDataLoader(self.dataset)


        test_loader = None
        if test is True or "retrain" in self.phase_list:
            self.prepareDataset(testset=True)
            test_loader = self.createDataLoader(self.testset)
            test_loader.shuffle = False

        if self.save is True:
            if self.save_path is not None:
                self.export_model(os.path.join(self.save_path, self.model_name), onnx=self.onnx_enabled)
            else:
                logger.error(f"Variable save_path is not configured in ADMMConfig or some other errror")
                self.export_model(self.model_name, onnx=self.onnx_enabled)

        for phase in self.phase_list:
            model_filename = self.model_name + "_admm_" + phase
            if phase == "train":
                logger.info(f"Training phase -> {phase} is starting")

                for epo in range(self.epoch):
                    for batch_idx, (data, target) in enumerate(dataloader):

                        self.optimizer.zero_grad()
                        output = self.model(data)
                        loss = self.loss(output, target)
                        loss.backward()

                        self.optimizer.step()

                        if batch_idx % 100 == 0:
                            logger.info(f'Train Epoch: {epo} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

                    if test is True:
                        self.test(test_loader, snapshot_enabled=self.snapshot_enabled, current_epoch=epo)
                    self.scheduler_step()

                self.reset_scheduler()

                #self.export_model(model_path=save_path)
                if self.save is True:
                    if self.save_path is not None:
                        full_path = os.path.join(self.save_path, model_filename)
                        self.export_model(model_path=full_path, onnx=self.onnx_enabled)
                    else:
                        self.export_model(model_path=model_filename, onnx=self.onnx_enabled)
                    #torch.save(self.model.state_dict(), self.model_name + "_admm_" + phase + ".pth")

            else:

                if tensorboard is True and self.tensorboard_writer is None:
                    self.tensorboard_writer = SummaryWriter(os.path.join(self.save_path,"tensorboard", phase))

                self.initialize_dualvar_auxvar()
                epo = 0
                counter = 0

                # self.pbar_iteration = tqdm(total=self.main_iterations)
                pbar = tqdm_progressbar(self.epoch, len(dataloader), phase, self.main_iterations)

                if phase == 'admm':
                    if self.tensor_buffering_enabled is True:
                        self.handler.init_processes()
                elif phase == 'retrain':
                    if self.tensor_buffering_enabled is True:
                        self.handler.terminate_all_processes()

                self.show_pruning_layer_job()
                logger.info(f"ADMM phase -> {phase} is starting")


                while self.main_iterations > counter and epo < self.epoch:
                    for batch_idx, (data, target) in enumerate(dataloader):

                        if self.adv_enabled is True: #and phase == "retrain":
                            num_adversarial_samples = int(self.adv_fraction * data.size(0))
                            adv_data = self.adv_attacker(data[:num_adversarial_samples], target[:num_adversarial_samples])
                            data = torch.cat([adv_data, data[num_adversarial_samples:]], dim=0)
                            pass

                        self.optimizer.zero_grad()
                        output = self.model(data)
                        loss = self.loss(output, target)
                        loss.backward()

                        if phase == "admm":
                            self.admm(counter)

                            # feature progress bar
                            if len(self.history_epsilon_W):
                                pbar.set_postfix({"Wk-Zk": self.history_epsilon_W[-1],
                                                  "Zk1-Zk": self.history_epsilon_Z[-1],
                                                  "loss": loss.item()}, refresh=True)
                            else:
                                pbar.set_postfix({"loss": loss.item()}, refresh=True)

                            # TODO: here we need to implement tensor weight buffering
                            if self.tensor_buffering_enabled is True and counter % (self.admm_iterations) == 0:
                                self.tensorBuffer_saving()
                            if self.early_termination_flag is True:
                                logger.info(f"Early Termination Flag was set, ADMM reached epsilon threshold")
                                counter += self.main_iterations
                                break

                        if phase == "retrain":
                            self.retrain(counter)
                            # feature progress bar
                            pbar.set_postfix({"loss": loss.item()})

                        self.optimizer.step()

                        counter += 1
                        pbar.update(1)


                        if self.main_iterations == counter:
                            break

                    self.scheduler_step()

                    if phase == "retrain":
                        #logger.debug(f'Retrain Epoch: {epo} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
                        if test is True:
                            self.test(test_loader, snapshot_enabled=self.snapshot_enabled)
                        epo +=1
                    elif test is True:
                        self.test(test_loader, snapshot_enabled=self.snapshot_enabled)
                    else:
                        logger.debug(f'Iteration Number: {counter} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

                # saving model
                #torch.save(self.model.state_dict(),self.model_name + "_admm_" + phase + ".pth")

                # Note: last time prune layers because optimizer tunes masked out values because of momentum etc.
                if phase == 'retrain':
                    self.prune_weight_layer()
                    self.test(test_loader, snapshot_enabled=self.snapshot_enabled)

                pbar.close()
                self.reset_scheduler()

                if self.save is True:
                    if self.save_path is not None:
                        full_path = os.path.join(self.save_path, model_filename)
                        self.export_model(model_path=full_path, onnx=self.onnx_enabled)
                        csv_path = os.path.join(self.save_path, f'{phase}_mask_tensor.csv')
                        self.export_tensor_list_csv(csv_path, self.list_masks)
                    else:
                        self.export_model(model_path=model_filename, onnx=self.onnx_enabled)
                        self.export_tensor_list_csv(f'{phase}_mask_tensor.csv', self.list_masks)

        if tensorboard is True and self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            del self.tensorboard_writer

        if self.getCudaState() is True:
            del self.list_Z, self.list_U, self.list_W, self.old_layer_list_z, self.list_masks, self.patternManager
            torch.cuda.empty_cache()

        # if self.tensor_buffering_enabled is True:
        #     self.handler.terminate_all_processes()



















