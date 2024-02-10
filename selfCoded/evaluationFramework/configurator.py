import configParser

import logging
logger = logging.getLogger(__name__)


# TODO: hier hier für jeden Abschnitt eigenen loader erstellen

class Configurator:
    def __init__(self):

        self.ConfigParser = configParser.ConfigParser()

    def loadDataHandlerConfig(self):
        self.configHandlerData = self.ConfigParser.getDataHandlerConfig()
        logger.info("DataHandlerConfig was loaded via ConfigParser")
        return self.configHandlerData