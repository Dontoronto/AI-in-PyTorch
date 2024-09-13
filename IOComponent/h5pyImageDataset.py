# datasetFactory.py
from torch.utils.data import Dataset
import h5py
import numpy
from functools import reduce
from PIL import Image

import logging
logger = logging.getLogger(__name__)

class H5PYImageDataset(Dataset):

    def __init__(self, root, query=None, transform=None):
        """

        :param root: path to file
        :param query: should be array of Strings
            Example:
                hdf5_file = '/Volumes/PortableSSD/Projekte/datasets/celeba_dataset/IMG/celeba_aligned_small.h5py'
                query = ['Volumes', 'PortableSSD', 'Projekte', 'datasets', 'celeba_dataset', 'IMG', 'img_align_celeba']
                dataset = h5pyImageDataset.H5PYImageDataset(path=hdf5_file, query=query)
        :__getitem__() -> just image tensor without label
        """
        try:
            self.file_object = h5py.File(root, 'r')
            logger.info("h5py-file was successfully loaded")
        except OSError as e:
            logger.exception("OSError")
        # Dynamically access the dataset based on the query path
        self.dataset = reduce(lambda obj, key: obj[key], query, self.file_object)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if (index >= len(self.dataset)):
            raise IndexError()
        if self.transform is not None:
            img = self.transform(numpy.array(self.dataset[str(index)+'.jpg']))
        else:
            #img = numpy.array(self.dataset[str(index)+'.jpg'])
            img = Image.fromarray(numpy.array(self.dataset[str(index)+'.jpg']))
        return img

    def show_image(self, index):
        Image.fromarray(numpy.array(self.dataset[str(index)+'.jpg'])).show()

