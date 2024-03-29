import copy

from .defaultTrainer import DefaultTrainer
from .admm_utils.utils import (create_magnitude_pruned_mask, add_tensors_inplace, subtract_tensors_inplace,
                               scale_and_add_tensors_inplace)
from .admm_utils.layerInfo import ADMMVariable
from .mapper.admm_mapper import ADMMConfigMapper, ADMMArchitectureConfigMapper
from .admm_utils.multiProcessHandler import MultiProcessHandler
from .admm_utils.tensorBuffer import TensorBuffer
from .admm_utils.pattern_pruning.patternManager import PatternManager

import torch
from torch.nn.utils import clip_grad_norm_
from multiprocessing import Process, Queue
import logging
from multiprocessing import Event

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
        # stores old layerlist Z for termination criterion
        self.old_layer_list_z = None
        self.history_epsilon_W = []
        self.history_epsilon_Z = []
        self.early_termination_flag = False

        self.patternManager = PatternManager()
        self.patternManager.setConnectivityPruning(True)



    def setADMMConfig(self, kwargs):
        ADMMConfigMapper(self, kwargs)

    def setADMMArchitectureConfig(self, kwargs):
        ADMMArchitectureConfigMapper(self, kwargs)

    def getHistoryEpsilonW(self):
        return self.history_epsilon_W

    def getHistoryEpsilonZ(self):
        return self.history_epsilon_Z


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

    def initialize_pruning_mask_layer_list(self, firstTime=True):
        '''
        This method should be general. You only have to change the implementation to get the right mask.
        Assings the class variable list_masks the pruning masks per layer on same index level.
        '''
        conv_list = [layerZ.Z for layerZ in self.list_Z]
        if firstTime is True:
            self.patternManager.assign_patterns_to_tensors(conv_list)
            self.list_masks = self.patternManager.get_pattern_masks()
        else:
            #self.patternManager.reduce_available_patterns(2)
            self.patternManager.update_pattern_assignments(conv_list, min_amount_indices=12)
            self.list_masks = self.patternManager.get_pattern_masks()
        # self.list_masks = [create_magnitude_pruned_mask(layerZ.Z, layerZ.sparsity) for layerZ in self.list_Z]
        #logger.info(f"Pruning mask was created")

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

        logger.info(f"Condition for termination criterion W^(k+1)-Z^(k+1)={frobenius_distance}/{self.epsilon_W}")
        return float(frobenius_distance)

    # TODO: terminate ADMM in a good manner
    def _termination_criterion_layer_Z(self):
        frobenius_distance = 0

        for layerZ, old_layerZ in zip(self.list_Z, self.old_layer_list_z):
            differenceW = layerZ.Z - old_layerZ.Z
            frobenius_distance += torch.norm(differenceW, p='fro')

        logger.info(f"Condition for termination criterion Z^(k+1)-Z^k= {frobenius_distance}/{self.epsilon_Z}")
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




    # TODO: weiß nichtmehr genau aber einfach im Kopf behalten
    def admm(self, curr_iteration):
        #logging.disable(logging.WARNING)
        self.clip_gradients()
        self.normalize_gradients()
        self.regularize_gradients()
        if curr_iteration % self.admm_iterations == 0:
            # TODO: abbruch testen
            self.store_old_AuxVariable()
            self.project_aux_layers()
            # if statement für dynamische oder statische maske
            if self.dynamic_masking is True and curr_iteration != 0:
                self.initialize_pruning_mask_layer_list(False)
            elif curr_iteration == 0:
                self.initialize_pruning_mask_layer_list(True)
            self.prune_aux_layers()
            if curr_iteration != 0:
                self.update_termination_criterion()
                self.update_dual_layers()
        self.solve_admm()
        pass

    def retrain(self, curr_iteration):
        self.clip_gradients()
        self.normalize_gradients()
        self.regularize_gradients()
        if curr_iteration == 0:
            #self.initialize_pruning_mask_layer_list(True)
            self.list_masks = self.patternManager.get_pattern_masks()
        self.prune_weight_layer()



    # TODO: methode umschreiben so dass epoch nichtmehr gebraucht wird für admm
    # TODO: symbiose von ADMM und retraining
    def train(self, test=False, onnx_enabled=False, tensor_buffering = False):
        self.model.train()
        self.preTrainingChecks()
        handler = MultiProcessHandler()

        # self.initialize_dualvar_auxvar()

        dataloader = self.createDataLoader(self.dataset)
        test_loader = None
        if test is True or "retrain" in self.phase_list:
            self.prepareDataset(testset=True)
            test_loader = self.createDataLoader(self.testset)
            test_loader.shuffle = False
        for phase in self.phase_list:
            save_path = self.model_name + "_admm_" + phase
            if phase == "train":
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

                #self.export_model(model_path=save_path)
                if self.save is True:
                    self.export_model(model_path=save_path, onnx=onnx_enabled)
                    #torch.save(self.model.state_dict(), self.model_name + "_admm_" + phase + ".pth")

            else:
                self.initialize_dualvar_auxvar()
                epo = 0
                counter = 0
                #handler = MultiProcessHandler()

                # TODO: here we need to implement tensor buffering initialization of multiprocessing

                if phase == 'admm' and tensor_buffering is True:
                    # tensor_queue = Queue()
                    # process = Process(target=tensor_saving_process, args=(tensor_queue, True,))
                    # process.start()
                    complete = Event()
                    process_id = 1
                    handler.start_process(
                        process_id=process_id,
                        class_to_instantiate=TensorBuffer,
                        init_args=[5],  # Assuming the first argument is 'capacity'
                        init_kwargs={
                            'file_path': 'experiment/data/frames_w',
                            'clear_file': True,
                            'convert_to_png': True,
                            'file_path_zero_matrices': 'experiment/data/frames_z'
                            # Add other constructor arguments here
                        },
                        process_args=[],  # Additional args for the method you're calling in the loop
                        process_kwargs={'event': complete}  # Additional kwargs for the method
                    )
                    logger.critical("right before starting")
                    complete.wait()
                    logger.critical("After complete Event")

                    complete = Event()
                    process_id2 = 2
                    handler.start_process(
                        process_id=process_id2,
                        class_to_instantiate=TensorBuffer,
                        init_args=[5],  # Assuming the first argument is 'capacity'
                        init_kwargs={
                            'file_path': 'experiment/data/frames_w2',
                            'clear_file': True,
                            'convert_to_png': True,
                            'file_path_zero_matrices': 'experiment/data/frames_z2'
                            # Add other constructor arguments here
                        },
                        process_args=[],  # Additional args for the method you're calling in the loop
                        process_kwargs={'event': complete}  # Additional kwargs for the method
                    )
                    logger.critical("right before starting")
                    complete.wait()
                    logger.critical("After complete Event")

                    complete = Event()
                    process_id3 = 3
                    handler.start_process(
                        process_id=process_id3,
                        class_to_instantiate=TensorBuffer,
                        init_args=[5],  # Assuming the first argument is 'capacity'
                        init_kwargs={
                            'file_path': 'experiment/data/frames_w3',
                            'clear_file': True,
                            'convert_to_png': True,
                            'file_path_zero_matrices': 'experiment/data/frames_z3'
                            # Add other constructor arguments here
                        },
                        process_args=[],  # Additional args for the method you're calling in the loop
                        process_kwargs={'event': complete}  # Additional kwargs for the method
                    )
                    logger.critical("right before starting")
                    complete.wait()
                    logger.critical("After complete Event")

                    complete = Event()
                    process_id4 = 4
                    handler.start_process(
                        process_id=process_id4,
                        class_to_instantiate=TensorBuffer,
                        init_args=[5],  # Assuming the first argument is 'capacity'
                        init_kwargs={
                            'file_path': 'experiment/data/frames_w4',
                            'clear_file': True,
                            'convert_to_png': True,
                            'file_path_zero_matrices': 'experiment/data/frames_z4'
                            # Add other constructor arguments here
                        },
                        process_args=[],  # Additional args for the method you're calling in the loop
                        process_kwargs={'event': complete}  # Additional kwargs for the method
                    )
                    logger.critical("right before starting")
                    complete.wait()
                    logger.critical("After complete Event")

                    complete = Event()
                    process_id5 = 5
                    handler.start_process(
                        process_id=process_id5,
                        class_to_instantiate=TensorBuffer,
                        init_args=[5],  # Assuming the first argument is 'capacity'
                        init_kwargs={
                            'file_path': 'experiment/data/frames_w5',
                            'clear_file': True,
                            'convert_to_png': True,
                            'file_path_zero_matrices': 'experiment/data/frames_z5'
                            # Add other constructor arguments here
                        },
                        process_args=[],  # Additional args for the method you're calling in the loop
                        process_kwargs={'event': complete}  # Additional kwargs for the method
                    )
                    logger.critical("right before starting")
                    complete.wait()
                    logger.critical("After complete Event")

                    complete = Event()
                    process_id6 = 6
                    handler.start_process(
                        process_id=process_id6,
                        class_to_instantiate=TensorBuffer,
                        init_args=[5],  # Assuming the first argument is 'capacity'
                        init_kwargs={
                            'file_path': 'experiment/data/frames_w6',
                            'clear_file': True,
                            'convert_to_png': True,
                            'file_path_zero_matrices': 'experiment/data/frames_z6'
                            # Add other constructor arguments here
                        },
                        process_args=[],  # Additional args for the method you're calling in the loop
                        process_kwargs={'event': complete}  # Additional kwargs for the method
                    )
                    logger.critical("right before starting")
                    complete.wait()
                    logger.critical("After complete Event")


                while self.main_iterations > counter and epo < self.epoch:
                    logger.critical("Loop Started")
                    for batch_idx, (data, target) in enumerate(dataloader):

                        self.optimizer.zero_grad()
                        output = self.model(data)
                        loss = self.loss(output, target)
                        loss.backward()

                        if phase == "admm":
                            self.admm(counter)
                            # TODO: here we need to implement tensor weight buffering
                            if tensor_buffering is True and counter % (self.admm_iterations) == 0:
                                w_z_weight = self.w_z_kernel_weight_extraction(0,0,0)
                                handler.put_item_in_queue(process_id, w_z_weight)
                                w_z_weight = self.w_z_kernel_weight_extraction(0,1,0)
                                handler.put_item_in_queue(process_id2, w_z_weight)
                                w_z_weight = self.w_z_kernel_weight_extraction(0,2,0)
                                handler.put_item_in_queue(process_id3, w_z_weight)
                                w_z_weight = self.w_z_kernel_weight_extraction(0,3,0)
                                handler.put_item_in_queue(process_id4, w_z_weight)
                                w_z_weight = self.w_z_kernel_weight_extraction(0,4,0)
                                handler.put_item_in_queue(process_id5, w_z_weight)
                                w_z_weight = self.w_z_kernel_weight_extraction(0,5,0)
                                handler.put_item_in_queue(process_id6, w_z_weight)
                            if self.early_termination_flag is True:
                                logger.info(f"Early Termination Flag was set, ADMM reached epsilon threshold")
                                counter += self.main_iterations
                                break

                        if phase == "retrain":
                            self.retrain(counter)

                        self.optimizer.step()

                        counter += 1
                        if self.main_iterations == counter:
                            break

                    if phase == "retrain":
                        logger.info(f'Retrain Epoch: {epo} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
                        self.test(test_loader, snapshot_enabled=False)
                        epo +=1
                    else:
                        logger.info(f'Iteration Number: {counter} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

                    if test is True:
                        self.test(test_loader, snapshot_enabled=False)
                # saving model
                #torch.save(self.model.state_dict(),self.model_name + "_admm_" + phase + ".pth")
                if self.save is True:
                    self.export_model(model_path=save_path, onnx=onnx_enabled)

        if tensor_buffering is True:
            #handler.terminate_process(1)
            handler.terminate_all_processes()
            # tensor_queue.put(None)
            # process.join()

















