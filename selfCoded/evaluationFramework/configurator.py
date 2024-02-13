import configParser
from SharedServices.utils import singleton

import logging
logger = logging.getLogger(__name__)


# TODO: hier hier für jeden Abschnitt eigenen loader erstellen
@singleton
class Configurator:
    def __init__(self):

        self.ConfigParser = configParser.ConfigParser()

    def loadDataHandlerConfig(self):
        self.configHandlerData = self.ConfigParser.getDataHandlerConfig()
        logger.info("DataHandlerConfig was loaded via ConfigParser")
        return self.configHandlerData

    def loadDatasetConfig(self):
        self.configDataset = self.ConfigParser.getDatasetConfig()
        logger.info("DatasetConfig was loaded via ConfigParser")
        return self.configDataset