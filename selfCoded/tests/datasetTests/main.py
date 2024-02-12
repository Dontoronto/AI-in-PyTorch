import sys, os

import torch

sys.path.append(os.getcwd())

import h5pyImageDataset
from datasetFactory import DatasetFactory
from configParser import ConfigParser
from torch.utils.data import DataLoader
import torchvision.transforms as T

import copy




def main():
    path = '/Volumes/PortableSSD/Projekte/datasets/celeba_dataset/IMG/celeba_aligned_small.h5py'
    query = ['Volumes','PortableSSD', 'Projekte', 'datasets', 'celeba_dataset', 'IMG', 'img_align_celeba']
    #dataset = h5pyImageDataset.H5PYImageDataset(path=hdf5_file,query=query)

    arguments = {'root': '/Volumes/PortableSSD/Projekte/datasets/celeba_dataset/IMG/celeba_aligned_small.h5py',
                    'query': ['Volumes','PortableSSD', 'Projekte', 'datasets', 'celeba_dataset', 'IMG', 'img_align_celeba'],
                 'placeholder': 1,
                 'storageType': "h5py",
                 'datasetName': "custom"}

    preprocess = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode("bilinear"), antialias=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.ConvertImageDtype(dtype=torch.float),
        T.Normalize(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406])
    ])

    loaded_args = ConfigParser().getDatasetConfig()

    #loaded_args['transform'] = preprocess
    print(loaded_args)

    dataset = DatasetFactory.createDataset(loaded_args)
    DatasetFactory.updateTransformer(dataset,preprocess)

    # dataset = DatasetFactory.createDataset(arguments)


    dataloader = DataLoader(dataset, batch_size=16)
    counter = 0
    for batch, labels in dataloader:
        print(batch.shape)
        for img in batch:
            print(img.shape)
        counter +=1
        if counter ==2:
            break

    pass

if __name__ == '__main__':
    main()


#%%
