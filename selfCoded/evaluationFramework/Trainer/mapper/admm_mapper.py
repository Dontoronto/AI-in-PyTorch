import logging

logger = logging.getLogger(__name__)

from ..admm_utils.layerInfo import LayerInfo

'''
These classes are for mapping json values to the proper class variables of 
the trainer class
'''


class ADMMConfigMapper:

    def __init__(self, trainer, config):
        self.trainer = trainer
        self.config = config
        self.map_config_to_trainer()

    def map_config_to_trainer(self):
        logger.info("ADMM Config was loaded into ADMMTrainer")

        # Assuming the configuration includes an 'admm_trainer' key with a nested dictionary
        admm_config = self.config.get('admm_trainer', {})

        # Map the configurations to the trainer instance variables
        self.trainer.main_iterations = admm_config.get('main_iterations', self.trainer.main_iterations)
        logger.info(f"main_iterations set to {self.trainer.main_iterations} of type {type(self.trainer.main_iterations)}")

        self.trainer.admm_iterations = admm_config.get('admm_iterations', self.trainer.admm_iterations)
        logger.info(f"admm_iterations set to {self.trainer.admm_iterations} of type {type(self.trainer.admm_iterations)}")

        # For mask creation mode in admm static mask for one time creation, dynamic for every admm iteration
        self.trainer.dynamic_masking = admm_config.get('dynamic_masking', self.trainer.dynamic_masking)
        logger.info(f"dynamic masking set to {self.trainer.dynamic_masking} of type {type(self.trainer.dynamic_masking)}")

        self.trainer.rho = admm_config.get('rho', self.trainer.rho)
        logger.info(f"rho set to {self.trainer.rho} of type {type(self.trainer.rho)}")

        self.trainer.gradient_threshold = admm_config.get('gradient_threshold', self.trainer.gradient_threshold)
        logger.info(f"gradient_threshold set to {self.trainer.gradient_threshold} of type {type(self.trainer.gradient_threshold)}")

        # Additional configurations
        if 'batch_size' in admm_config:
            self.trainer.batch_size_norm_coeff = 1 / admm_config.get('batch_size') if admm_config.get('batch_size') else self.trainer.batch_size_norm_coeff
            logger.info(f"batch_size_norm_coeff set to {self.trainer.batch_size_norm_coeff} of type {type(self.trainer.batch_size_norm_coeff)}")

        self.trainer.regularization_l2_norm_enabled = admm_config.get('regularization_l2_norm_enabled', self.trainer.regularization_l2_norm_enabled)
        logger.info(f"regularization_l2_norm_enabled set to {self.trainer.regularization_l2_norm_enabled} of type {type(self.trainer.regularization_l2_norm_enabled)}")

        self.trainer.regularization_l_norm_decay = admm_config.get('regularization_l_norm_decay', self.trainer.regularization_l_norm_decay)
        logger.info(f"regularization_l_norm_decay set to {self.trainer.regularization_l_norm_decay} of type {type(self.trainer.regularization_l_norm_decay)}")

        # Handling phase_list with logging for type validation
        self.trainer.phase_list = admm_config.get('phase_list', self.trainer.phase_list)
        if isinstance(self.trainer.phase_list, list):
            logger.info(f"Phase list set to {self.trainer.phase_list} of type {type(self.trainer.phase_list)}")
        else:
            logger.warning("Phase list from ADMMConfig.json is not of type list. Example: ['admm', 'retrain']")

        # For model saving configurations
        self.trainer.save = admm_config.get('save', self.trainer.save)
        logger.info(f"save set to {self.trainer.save} of type {type(self.trainer.save)}")


class ADMMArchitectureConfigMapper:

    def __init__(self, trainer, config):
        self.trainer = trainer
        self.config = config
        self.map_architecture_config_to_trainer()

    def map_architecture_config_to_trainer(self):
        logger.info("ADMM Architecture Config was loaded into ADMMTrainer")
        admmArchitectureConfig = self.config
        for val in admmArchitectureConfig:
            if val['sparsity'] is not None:
                for name_module, module in self.trainer.model.named_modules():
                    if name_module == val['op_names']:
                        for name_param, param in self.trainer.model.named_parameters():
                            if name_param == name_module + ".weight":
                                self.trainer.list_W.append(LayerInfo(name_module, module, param, val['sparsity']))
                                break
                        break
            else:
                continue