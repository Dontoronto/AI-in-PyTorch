# dataset class
from torch.utils.data import Dataset
import h5py
import numpy
import torch
import matplotlib.pyplot as plt
from functools import reduce

class H5PYImageDataset(Dataset):

    def __init__(self, path, query=None, transform=None):
        """

        :param path: path to file
        :param query: should be array of Strings
            Example:
                hdf5_file = '/Volumes/PortableSSD/Projekte/datasets/celeba_dataset/IMG/celeba_aligned_small.h5py'
                query = ['Volumes', 'PortableSSD', 'Projekte', 'datasets', 'celeba_dataset', 'IMG', 'img_align_celeba']
                dataset = h5pyImageDataset.H5PYImageDataset(path=hdf5_file, query=query)
        :__getitem__() -> just image tensor without label
        """
        self.file_object = h5py.File(path, 'r')
        # Dynamically access the dataset based on the query path
        self.dataset = reduce(lambda obj, key: obj[key], query, self.file_object)
        self.transform = transform


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if (index >= len(self.dataset)):
            raise IndexError()
        if self.transform is not None:
            imgTensor = self.transform(numpy.array(self.dataset[str(index)+'.jpg']))
        else:
            img = numpy.array(self.dataset[str(index)+'.jpg'])/255
            imgTensor = torch.tensor(img, dtype=torch.float32)
        return imgTensor

    def plot_image(self, index):
        plt.imshow(numpy.array(self.dataset[str(index)+'.jpg']), interpolation='nearest')


