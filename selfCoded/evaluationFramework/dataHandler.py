import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset
from torchvision.transforms import ToPILImage
from torchvision.datasets.folder import ImageFolder
from PIL import Image


from SharedServices.utils import singleton, pil_loader

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
        self.testset = None

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

    def getPreprocessBatchedFunction(self):
        return self.preprocessBatched

    def getPreprocessBackwardsNonBatchedFunction(self):
        return self.preprocessBackwardsNonBatched

    def preprocessNonBatched(self, img):
        return self.transform(img)

    def preprocessBatched(self, img):
        return self.transform(img).unsqueeze(0)

    def preprocessBackwardsNonBatched(self, tensor, numpy_original_shape_flag=False):
        # TODO: evtl. müssen wir nicht image tensoren sondern auch batch tensoren zurück umwandeln. Hier
        # TODO: testen und evtl. anpassen damit automatisch erkannt wird was gefordert ist
        tensorBack = tensor.clone().detach()
        if self.mean is not None and self.std is not None:
            meanBack = torch.tensor(self.mean).view(-1, 1, 1)
            stdBack = torch.tensor(self.std).view(-1, 1, 1)
            tensorBack = tensorBack * stdBack + meanBack
        tensorBack = torch.clamp(tensorBack, 0, 1)
        if numpy_original_shape_flag is False:
            to_pil = ToPILImage()
            image = to_pil(tensorBack)
            return image
        else:
            numpy_array = tensorBack.squeeze(0).permute((1,2,0)).numpy()
            return numpy_array


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

    @staticmethod
    def loadImage(path):
        '''
        loading images in standardized format example jpeg -> (244,244,3)
        :param path: path to file
        :return: PIL Image Instance of image
        '''
        logger.info("Loading image from: " + path)
        return Image.open(path)

    def setDataset(self, dataset):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            logger.warning("Failed to set Dataset, Dataset is not of type torch.utils.data.Dataset")

    def setTestset(self, testset):
        if isinstance(testset, Dataset):
            self.testset = testset
        else:
            logger.warning("Failed to set Testset, Testset is not of type torch.utils.data.Dataset")

    def create_imageFolder_dataset(self, path):
        '''
        Creates a Dataset Object of type Dataset from structured custom Data seperated in directorys
        named according to the label
        :param path: path to the root directory where subfolders are located
        :return: Dataset object from torchvision ImageFolder
        '''
        return ImageFolder(path, loader=pil_loader, transform=self.getTransformer())


    #TODO: missmatch in train and test dataset when calling configurator first testset then argument train
    def loadDataset(self, testset=False):
        if testset is False:
            if self.Configurator is None and self.dataset is None:
                logging.critical("No Configurator  or Dataset configured in DataHandler. "
                                 "Please initialize with Configurator or use setConfigurator()")
                return
            self.dataset = self.Configurator.loadDataset()
            self.updateDatasetTransformer()
            if self.transform is not None:
                self.updateDatasetTransformer()
                logger.info("Transformer loaded into Dataset")
            else:
                logger.warning("Tried to load Transformer into Dataset. Transformer not configured.")
            return self.dataset

        elif testset is True:
            if self.Configurator is None and self.testset is None:
                logging.critical("No Configurator or Testset configured in DataHandler. "
                                 "Please initialize with Configurator or use setConfigurator()")
                return
            self.testset = self.Configurator.loadDataset(train=False)
            if self.transform is not None:
                self.updateDatasetTransformer()
                logger.info("Transformer loaded into Testset")
            else:
                logger.warning("Tried to load Transformer into Testset. Transformer not configured.")
            return self.testset


    def updateDatasetTransformer(self):
        if isinstance(self.dataset, Subset):
            logger.info("Transformer of Subset Dataset was changed to:")
            logger.info(self.transform)
            self.dataset.dataset.transform = self.transform
        elif isinstance(self.dataset, Dataset):
            logger.info("Transformer of Dataset was changed to:")
            logger.info(self.transform)
            self.dataset.transform = self.transform
        if isinstance(self.testset, Subset):
            logger.info("Transformer of Subset Dataset was changed to:")
            logger.info(self.transform)
            self.testset.dataset.transform = self.transform
        elif isinstance(self.testset, Dataset):
            logger.info("Transformer of Testset was changed to:")
            logger.info(self.transform)
            self.testset.transform = self.transform

