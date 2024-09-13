import configParser
import torchvision.transforms as T
import torch
from SharedServices.utils import singleton
from IOComponent.datasetFactory import DatasetFactory
from IOComponent.transformators import transformators


import logging
logger = logging.getLogger(__name__)

@singleton
class Configurator:
    def __init__(self):

        self.ConfigParser = configParser.ConfigParser()
        self.configHandlerData = None
        self.configTransformer = None
        self.configDatahandlerSettings = None
        self.configDataset = None
        self.configTrainer = None
        self.configDataloader = None
        self.cuda_enabled = False

    def loadDataHandlerSettigns(self):

        self.configDatahandlerSettings = self._loadDataHandlerSettingsConfig()

        if self.configDatahandlerSettings.get('cuda') is True:
            self.cuda_enabled = True
            return True
        else:
            self.cuda_enabled = False
            return False


    def loadTransformerNEW(self):

        self.configTransformer = self._loadTransformerConfig()

        if self.configTransformer.get('dataset') is not None:
            if self.configTransformer.get('dataset') == "imagenet":
                return transformators.imagenet_transformer(image_flag=True)
            if self.configTransformer.get('dataset') == "adv_imagenet":
                return transformators.adv_imagenet_transformer()
            elif self.configTransformer.get('dataset') == "mnist":
                return transformators.mnist_transformer()

    def loadDataset(self, train=True):
        '''
        returns instance of a Dataset
        :param train: if False we download the testset if true we download the testset, if no testset is available
        testset and training dataset are the same
        :return:
        '''
        self.configDataset = self._loadDatasetConfig()
        self.configDataset["train"] = train
        return DatasetFactory.createDataset(self.configDataset)

    def _loadDataHandlerSettingsConfig(self):
        self.configDatahandlerSettings = self.ConfigParser.getDataHandlerSettingsConfig()
        logger.info("Settings from DataHandlerConfig was loaded via ConfigParser")
        return self.configDatahandlerSettings

    def _loadTransformerConfig(self):
        self.configTransformer = self.ConfigParser.getTransformerConfig()
        logger.info("TransformerConfig from DataHandlerConfig was loaded via ConfigParser")
        return self.configTransformer


    def _loadDatasetConfig(self):
        self.configDataset = self.ConfigParser.getDatasetConfig()
        logger.info("DatasetConfig was loaded via ConfigParser")
        return self.configDataset

    def loadTrainingConfig(self):
        self.configTrainer = self.ConfigParser.getTrainerConfig()
        logger.info("TrainerConfig settings was loaded via ConfigParser")
        return self.configTrainer

    def loadDataloaderConfig(self):
        self.configDataloader = self.ConfigParser.getDataLoaderConfig()
        logger.info("Dataloader settings was loaded via ConfigParser")
        return self.configDataloader

    def loadSnapshotConfig(self):
        self.configSnapshot = self.ConfigParser.getSnapshotConfig()
        logger.info("Snapshot settings was loaded via ConfigParser")
        return self.configSnapshot

    def loadConfigFromRegistry(self, configName):
        temp = self.ConfigParser.getConfigFromRegistry(configName)
        logger.info(f"Loaded {configName} from Registry")
        return temp


