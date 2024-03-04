# trainerFactory.py
import logging

logger = logging.getLogger(__name__)

from .trainer import Trainer
from .defaultTrainer import DefaultTrainer
from .admmTrainer import ADMMTrainer
import torch
import copy



# TODO: Architecture vergleichen und sachen schön programmieren
class TrainerFactory:
    """
    This class if for creating a proper Trainer Object
    """

    @staticmethod
    def filterOptimizerConfigs(kwargs):
        """
        creates new dictionary with necessary values for optimizer function
        """
        tempConfig = copy.deepcopy(kwargs)
        del tempConfig['epoch']
        del tempConfig['loss']
        del tempConfig['optimizer']
        del tempConfig['pre_optimization_tuning_path']
        return tempConfig


    @staticmethod
    def createTrainer(model, dataHandler, kwargs):
        optimizer = None
        loss = None
        epoch = 1

        if kwargs.get('epoch') is not None:
            epoch = kwargs.get('epoch')
        # TODO: eigene funktion für auswahl des loss function
        if kwargs.get('loss') == "BCEWithLogitsLoss":
            loss = torch.nn.BCEWithLogitsLoss()
        elif kwargs.get('loss') == "CrossEntropyLoss":
            loss = torch.nn.CrossEntropyLoss()

        # TODO: eigene funktion für asuwahl des Optimizers
        if kwargs.get('optimizer') == "Adam":
            tempConfig = TrainerFactory.filterOptimizerConfigs(kwargs)
            logger.info("Filtered OptimizerConfig: " + str(tempConfig))
            optimizer = torch.optim.Adam(model.parameters(), **tempConfig)
        elif kwargs.get('optimizer') == "SGD":
            tempConfig = TrainerFactory.filterOptimizerConfigs(kwargs)
            logger.info("Filtered OptimizerConfig: " + str(tempConfig))
            optimizer = torch.optim.SGD(model.parameters(), **tempConfig)

        if kwargs.get('pre_optimization_tuning_path') == True:
            logger.info("Creating ADMMTrainer.")
            return ADMMTrainer(model, dataHandler, loss=loss, optimizer=optimizer, epoch=epoch)
        elif kwargs.get('pre_optimization_tuning_path') == False:
            logger.info("Creating DefaultTrainer.")
            return DefaultTrainer(model, dataHandler, loss=loss, optimizer=optimizer, epoch=epoch)
