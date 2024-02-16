# trainerFactory.py
import logging

logger = logging.getLogger(__name__)

from .trainer import Trainer
from .defaultTrainer import DefaultTrainer
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
        return tempConfig


    @staticmethod
    def createTrainer(model, dataHandler, kwargs):
        optimizer = None
        loss = None
        epoch = 1

        if kwargs.get('epoch') is not None:
            epoch = kwargs.get('epoch')
        # TODO: eigene funktion für auswahl des loss function
        if kwargs.get('loss') == "BinaryCrossEntropyLoss":
            loss = torch.nn.BCEWithLogitsLoss()

        # TODO: eigene funktion für asuwahl des Optimizers
        if kwargs.get('optimizer') == "Adam":
            tempConfig = TrainerFactory.filterOptimizerConfigs(kwargs)
            logger.info("Filtered OptimizerConfig: " + str(tempConfig))
            optimizer = torch.optim.Adam(model.parameters(), **tempConfig)

        return DefaultTrainer(model, dataHandler, loss=loss, optimizer=optimizer, epoch=epoch)
