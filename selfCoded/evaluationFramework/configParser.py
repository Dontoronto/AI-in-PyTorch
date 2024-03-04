import json

import logging
logger = logging.getLogger(__name__)

class ConfigParser:
    def __init__(self):
        self.dataHandlerConfig = None
        self.datasetConfig = None
        self.trainerConfig = None
        self.dataloaderConfig = None

    # TODO: need to be more general
    # TODO: if get method is called but no file is there error message that loading didn't work
    def getDataHandlerConfig(self):
        if self.dataHandlerConfig == None:
            with open('configs/DataHandlerConfig.json', 'r') as json_file:
                self.dataHandlerConfig = json.load(json_file)['transform']['transformParam']
                logger.info("Transformation Parameters from DataHandlerConfig.json:")
                logger.info(self.dataHandlerConfig)

        return self.dataHandlerConfig

    def getDatasetConfig(self):
        if self.datasetConfig == None:
            with open('configs/DataHandlerConfig.json', 'r') as json_file:
                self.datasetConfig = json.load(json_file)['dataset']
                logger.info("Dataset Configs from DataHandlerConfig.json:")
                logger.info(self.datasetConfig)

        return self.datasetConfig

    def getTrainerConfig(self):
        if self.trainerConfig == None:
            with open('configs/TrainerConfig.json', 'r') as json_file:
                self.trainerConfig = json.load(json_file)['trainer']
                logger.info("Trainer Configs from TrainerConfig.json:")
                logger.info(self.trainerConfig)

        return self.trainerConfig

    def getDataLoaderConfig(self):
        if self.dataloaderConfig == None:
            with open('configs/TrainerConfig.json', 'r') as json_file:
                self.dataloaderConfig = json.load(json_file)['dataloader']
                logger.info("Dataloader Configs from TrainerConfig.json:")
                logger.info(self.dataloaderConfig)

        return self.dataloaderConfig

    def getConfigFromRegistry(self, configName):
        with open('configs/ConfigRegistry.json', 'r') as json_file:
            path = json.load(json_file)[configName]
            logger.info(f"Loaded {configName} from ConfigRegistry.json")
            logger.info(f"Loaded {configName} has path: {path}")
            with open(path, 'r') as config:
                logger.info(f"Loaded {configName}")
                configFile = json.load(config)
                return configFile




