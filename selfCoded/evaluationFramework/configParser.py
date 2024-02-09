import json

import logging
logger = logging.getLogger(__name__)

class ConfigParser:
    def __init__(self):
        self.dataHandlerConfig = None

    # TODO: need to be more general
    def getDataHandlerConfig(self):
        if self.dataHandlerConfig == None:
            with open('configs/DataHandlerConfig.json', 'r') as json_file:
                self.dataHandlerConfig = json.load(json_file)['transform']['transformParam']
                logger.info("Transformation Parameters from DataHandlerConfig.json:")
                logger.info(self.dataHandlerConfig)

        return self.dataHandlerConfig



