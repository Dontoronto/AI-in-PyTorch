import torch
import torchvision.transforms as T
from torchvision.transforms import ToPILImage
from PIL import Image

from SharedServices.utils import singleton

import logging
logger = logging.getLogger(__name__)

@singleton
class DataHandler:
    def __init__(self, Configurator=None):
        self.transform = None
        self.std = None
        self.mean = None
        self.Configurator = Configurator
        self.configHandlerData = None

    def setConfigurator(self, Configurator):
        logger.info("Configurator was manually set to: " + str(self.Configurator))
        self.Configurator = Configurator

    # TODO: evtl auf kwargs umstellen so dass variablen automatisch zugewiesen werden und nur reihenfolge fix sein soll
    # TODO: evtl. nur prüfen ob Format passt bzw. datentyp von den eigenen variablen
    def loadTransformer(self):

        if not self.configHandlerData:
            logging.critical("No Configurator configured. Please initialize with Configurator or use setConfigurator()")
        self.configHandlerData = self.Configurator.loadDataHandlerConfig()

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
        logger.info("Preprocessing Steps will be:")
        logger.info(self.transform)


    def setTransformer(self, transformer):
        self.transform = transformer
        self.std = transformer.std
        self.mean = transformer.mean
        logger.info("Preprocessing Steps will be:")
        logger.info(self.transform)

    def preprocess(self, img):
        return self.transform(img)

    def preprocessBackwards(self, tensor):
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

    # loading images in standardized format example jpeg -> (244,244,3)
    @staticmethod
    def loadImage(path):
        logger.info("Loading image from: " + path)
        return Image.open(path)
