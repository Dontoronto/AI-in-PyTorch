import configParser
import torchvision.transforms as T
import torch
from SharedServices.utils import singleton
from IOComponent.datasetFactory import DatasetFactory
from IOComponent.transformators import transformators


import logging
logger = logging.getLogger(__name__)

# TODO: hier hier für jeden Abschnitt eigenen loader erstellen
# TODO: Interpretationsteil hierher verschieben z.B. für Trainer. Er soll gleich die richtige loss func übergeben.(invP)
# TODO: Inversion of Injection. Ansonsten ist das nichts anderes als der Parser -> nochmal schauen????
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

    # TODO: depreciated will be deleted
    def loadTransformer(self):

        self.configHandlerData = self._loadDataHandlerConfig()

        transformerList = list()

        # TODO: Hier schauen mit Exceptions dass alle Werte immer irgendwie gesetzt werden oder falls falsch gesetzt wurde
        if self.configHandlerData.get('resize_size') is not None:

            # Interpolation is used when input image is smaller the resize size. Pixels will be filled with values
            # corresponding to other pixels dependent on interpolation method
            if self.configHandlerData.get('interpolation') is not None:
                logger.info("Preprocessing Resize interpolation is set to: " +
                            self.configHandlerData.get('interpolation'))
                interpolation = T.InterpolationMode(self.configHandlerData.get('interpolation'))
            else:
                logger.info("Preprocessing Resize interpolation is set to: DEFAULT")
                logger.info("Possible Values would be: nearest, nearest-exact, bilinear, bicubic, box, hamming, laczos")
                interpolation = T.InterpolationMode("bilinear")

            if self.configHandlerData.get('antialias') is False:
                antialias = False
                logger.info("Preprocessing Resize Antialias deactivated")
            else:
                antialias = True
                logger.info("Preprocessing Resize Antialias activated")
            transformerList.append(T.Resize(self.configHandlerData['resize_size'],
                                            interpolation=interpolation,
                                            antialias=antialias))

        if self.configHandlerData.get('crop_size') is not None:
            logger.info("Preprocessing crop_size is set so: " + str(self.configHandlerData.get('crop_size')))
            transformerList.append(T.CenterCrop(self.configHandlerData.get('crop_size')))

        transformerList.append(T.ToTensor())

        if self.configHandlerData.get('dtype') is not None:
            if self.configHandlerData.get('dtype') == "float":
                logger.info("Preprocessing dtype is set so: " + str(self.configHandlerData.get('dtype')))
                transformerList.append((T.ConvertImageDtype(dtype=torch.float)))

        if self.configHandlerData.get('normalize') is not None:
            logger.info("Preprocessing Normalization is activated")

            # TODO: ausnahmefälle heraussuchen und implementieren
            # TODO: prüfen ob datenformat richtig ist und typ
            # Falls keine Normalisierung stattfindet muss hier ein anderer Wert hin. evtl mean[0,0,0] und std[1,1,1]
            # siehe OneNote und Formel für inverse Preprocessing
            self.std = self.configHandlerData['normalize'].get('std')
            self.mean = self.configHandlerData['normalize'].get('mean')
            logger.info("Preprocessing Normalization std is set to: " + str(self.std))
            logger.info("Preprocessing Normalization mean is set to: " + str(self.mean))
            transformerList.append((T.Normalize(mean=self.mean
                                                , std=self.std)))

        self.transform = T.Compose(transformerList)
        logger.info(self.transform)
        return self.transform

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

    # TODO: depreciated will be deleted -> use _loadTransformerConfig instead
    def _loadDataHandlerConfig(self):
        self.configHandlerData = self.ConfigParser.getDataHandlerConfig()
        logger.info("DataHandlerConfig was loaded via ConfigParser")
        return self.configHandlerData

    def _loadDatasetConfig(self):
        self.configDataset = self.ConfigParser.getDatasetConfig()
        logger.info("DatasetConfig was loaded via ConfigParser")
        return self.configDataset

    # TODO: will be changing in the future. Initialization should be here
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


