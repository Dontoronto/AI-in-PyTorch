import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split

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

        # Check if datasets are already created
        if cls.train_set is not None and cls.val_set is not None:
            if train:
                return cls.train_set
            else:
                return cls.val_set

        # Create the ImageFolder dataset
        imagenet = ImageFolder(
            root=root,
            is_valid_file=lambda path: not os.path.basename(path).startswith("._")
        )

        # Create the generator with the specified seed
        if cuda_enabled is True:
            #generator1 = torch.Generator(device=torch.get_default_device()).manual_seed(seed)
            generator1 = torch.Generator(device=torch.device('cuda:0')).manual_seed(seed)
        else:
            generator1 = torch.Generator().manual_seed(seed)

        total_images = len(imagenet)
        sum_split_ratio = sum(split_ratio)

        if sum_split_ratio <= 1:
            if sum_split_ratio < 1:
                other = 1 - sum_split_ratio
                split_ratio.append(other)
                cls.train_set, cls.val_set, _ = random_split(imagenet, split_ratio, generator=generator1)
            else:
                # Split the dataset into train and validation sets
                cls.train_set, cls.val_set = random_split(imagenet, split_ratio, generator=generator1)
        elif sum_split_ratio <= total_images:
            if sum_split_ratio < total_images:
                other = total_images - sum_split_ratio
                split_ratio.append(other)
                cls.train_set, cls.val_set, _ = random_split(imagenet, split_ratio, generator=generator1)
            else:
                # Split the dataset into train and validation sets
                cls.train_set, cls.val_set = random_split(imagenet, split_ratio, generator=generator1)

        # Return the requested dataset
        if train:
            return cls.train_set
        else:
            return cls.val_set
# def get_dataset(root, train=True, split_ratio=[0.7, 0.3], seed=42):
#     """
#     Returns the train or validation dataset based on the specified arguments.
#
#     Args:
#     path (str): The path to the dataset.
#     train (bool): If True, returns the training set. If False, returns the validation set.
#     split_ratio (list): The ratio to split the dataset into training and validation sets.
#     seed (int): The seed for random splitting to ensure reproducibility.
#
#     Returns:
#     torch.utils.data.Dataset: The requested dataset (train or val).
#     """
#
#     # Create the ImageFolder dataset
#     imagenet = ImageFolder(
#         root=root,
#         is_valid_file=lambda path: not os.path.basename(path).startswith("._")
#     )
#
#     # Create the generator with the specified seed
#     generator1 = torch.Generator().manual_seed(seed)
#
#     total_images = len(imagenet)
#     sum_split_ratio = sum(split_ratio)
#
#     if sum_split_ratio <= 1:
#         if sum_split_ratio < 1:
#             other = 1 - sum_split_ratio
#             split_ratio.append(other)
#             train_set, val_set, _ = random_split(imagenet, split_ratio, generator=generator1)
#         else:
#             # Split the dataset into train and validation sets
#             train_set, val_set = random_split(imagenet, split_ratio, generator=generator1)
#     elif sum_split_ratio <= total_images:
#         if sum_split_ratio < total_images:
#             other = total_images - sum_split_ratio
#             split_ratio.append(other)
#             train_set, val_set, _ = random_split(imagenet, split_ratio, generator=generator1)
#         else:
#             # Split the dataset into train and validation sets
#             train_set, val_set = random_split(imagenet, split_ratio, generator=generator1)
#
#
#     # Return the requested dataset
#     if train:
#         return train_set
#     else:
#         return val_set
