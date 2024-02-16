import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
from PIL import Image

from SharedServices.utils import singleton


import logging
logger = logging.getLogger(__name__)

@singleton
class DataHandler:
    def __init__(self, Configurator=None):
        logger.info("Datahandler initialized")
        self.transform = None
        self.std = None
        self.mean = None
        self.Configurator = Configurator
        self.dataset = None

        if self.Configurator:
            self.setTransformer(Configurator.loadTransformer())

    def setConfigurator(self, Configurator):
        logger.info("Configurator was manually set to: " + str(self.Configurator))
        self.Configurator = Configurator

    def setTransformer(self, transformer):
        self.transform = transformer
        if isinstance(transformer, T.Compose):
            for t in transformer.transforms:
                if isinstance(transformer, T.Normalize):
                    self.mean = t.mean
                    self.std = t.std
        else:
            self.std = transformer.std
            self.mean = transformer.mean
        logger.info("Preprocessing Steps will be:")
        logger.info(self.transform)

    def getTransformer(self):
        if self.transform is not None:
            logger.info("DataHandler returned Transformer")
            return self.transform
        logger.info("Transformer is not configured")

    def preprocessNonBatched(self, img):
        return self.transform(img)

    def preprocessBatched(self, img):
        return self.transform(img).unsqueeze(0)

    def preprocessBackwardsNonBatched(self, tensor):
        # TODO: evtl. müssen wir nicht image tensoren sondern auch batch tensoren zurück umwandeln. Hier
        # TODO: testen und evtl. anpassen damit automatisch erkannt wird was gefordert ist
        tensorBack = tensor.clone().detach()
        if self.mean is not None and self.std is not None:
            meanBack = torch.tensor(self.mean).view(-1, 1, 1)
            stdBack = torch.tensor(self.std).view(-1, 1, 1)
            tensorBack = tensorBack * stdBack + meanBack
        tensorBack = torch.clamp(tensorBack, 0, 1)
        to_pil = ToPILImage()
        image = to_pil(tensorBack)
        return image

    def preprocessBackwardsBatched(self, batch):
        # TODO: evtl. müssen wir nicht image tensoren sondern auch batch tensoren zurück umwandeln. Hier
        # TODO: testen und evtl. anpassen damit automatisch erkannt wird was gefordert ist
        tensors = batch.clone().detach()
        image_list = list()
        for tensor in tensors:
            if self.mean is not None and self.std is not None:
                meanBack = torch.tensor(self.mean).view(-1, 1, 1)
                stdBack = torch.tensor(self.std).view(-1, 1, 1)
                tensor = tensor * stdBack + meanBack
            tensorBack = torch.clamp(tensor, 0, 1)
            to_pil = ToPILImage()
            image = to_pil(tensorBack)
            image_list.append(image)
        return image_list

    # loading images in standardized format example jpeg -> (244,244,3)
    @staticmethod
    def loadImage(path):
        logger.info("Loading image from: " + path)
        return Image.open(path)

    def setDataset(self, dataset):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            logger.warning("Failed to set Dataset, Dataset is not of type torch.utils.data.Dataset")

    def loadDataset(self):
        if self.Configurator is None and self.dataset is None:
            logging.critical("No Configurator configured in DataHandler. "
                             "Please initialize with Configurator or use setConfigurator()")
            return
        self.dataset = self.Configurator.loadDataset()
        if self.transform is not None:
            self.updateDatasetTransformer()
            logger.info("Transformer loaded into Dataset")
        else:
            logger.warning("Tried to load Transformer into Dataset. Transformer not configured.")
        return self.dataset

    def updateDatasetTransformer(self):
        if isinstance(self.dataset, Dataset):
            logger.info("Transformer of Dataset was changed to:")
            logger.info(self.transform)
            self.dataset.transform = self.transform
            self.dataset.target_transform = ytrafo #lambda y: torch.zeros(1000, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)

    #TODO: if you want to create the Dataset via Code
    def createDataset(self):
        pass

def ytrafo(y):
    return torch.zeros(1000, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)