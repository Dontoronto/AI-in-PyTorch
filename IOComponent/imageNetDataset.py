import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split
from torchvision import datasets, transforms

class ImagenetDataset:

    train_set = None
    val_set = None

    @classmethod
    def get_dataset(cls, root, train=True, split_ratio=[0.7, 0.3], seed=42, cuda_enabled=False):
        """
        Returns the train or validation dataset based on the specified arguments.

        Args:
        path (str): The path to the dataset.
        train (bool): If True, returns the training set. If False, returns the validation set.
        split_ratio (list): The ratio to split the dataset into training and validation sets.
        seed (int): The seed for random splitting to ensure reproducibility.

        Returns:
        torch.utils.data.Dataset: The requested dataset (train or val).
        """

        if train is True:
            return datasets.ImageNet(root=root, split='train',
                                               is_valid_file=lambda path: not os.path.basename(path).startswith("._"))
        elif train is False:
            return datasets.ImageNet(root=root, split='val',
                                              is_valid_file=lambda path: not os.path.basename(path).startswith("._"))

