# datasetFactory.py
import logging

logger = logging.getLogger(__name__)

from .h5pyImageDataset import H5PYImageDataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import torch
from .imageNetDataset import ImagenetDataset


@staticmethod
def kwargs_filter(kwargs, expected_keys):
    """
    Filters out the dictionary and removes unnecessary key,values
    without changing the original dictionary
    :param kwargs: -> dictionary
    :param expected_keys:  -> dictionary
    :return: -> filtered dictionary
    """
    return {k: v for k, v in kwargs.items() if k in expected_keys}



class DatasetFactory:
    """
    This class is to create the right Dataset considering the structure, etc.
    Preconfigured datasets can be loaded or customized ones
    """
    @staticmethod
    #datasetName: str, storageType: str,
    #storageType == "h5py"
    def createDataset(kwargs):
        #if datasetName == 'custom':
        if kwargs.get('datasetName') == "custom":
            #TODO: evtl noch für Daten in Ordnerstrukturen eigenes Dataset erstellen
            #TODO: https://github.com/pytorch/vision/blob/a52607ece94aedbe41107617ace22a8da91efc25/torchvision/datasets/folder.py#L107
            #TODO: ImageFolder klasse
            if kwargs.get('storageType') == "h5py":
                expected_keys = {"root", "query"}
                logger.info("Creating Custom Dataset of h5py-file")
                return H5PYImageDataset(**kwargs_filter(kwargs,expected_keys))
        elif kwargs.get('datasetName') == 'cifar10':
            logger.info("Creating preconfigured Dataset for CIFAR10")
            expected_keys = {"root","download","train"}
            return CIFAR10(**kwargs_filter(kwargs,expected_keys))

        elif kwargs.get('datasetName') == 'cifar100':
            logger.info("Creating preconfigured Dataset for CIFAR100")
            expected_keys = {"root","download","train"}
            return CIFAR100(**kwargs_filter(kwargs,expected_keys))

        elif kwargs.get('datasetName') == 'mnist':
            logger.info("Creating preconfigured Dataset for MNIST")
            expected_keys = {"root","download","train"}
            return MNIST(**kwargs_filter(kwargs,expected_keys))
        elif kwargs.get('datasetName') == 'imagenet':
            logger.info("Creating preconfigured Dataset for ImageNet")
            expected_keys = {"root","train"} #"split_ratio","seed","cuda_enabled"
            return ImagenetDataset.get_dataset(**kwargs_filter(kwargs,expected_keys))


