import sys, os
sys.path.append(os.getcwd())

import h5pyImageDataset
from datasetFactory import DatasetFactory
from configParser import ConfigParser




def main():
    path = '/Volumes/PortableSSD/Projekte/datasets/celeba_dataset/IMG/celeba_aligned_small.h5py'
    query = ['Volumes','PortableSSD', 'Projekte', 'datasets', 'celeba_dataset', 'IMG', 'img_align_celeba']
    #dataset = h5pyImageDataset.H5PYImageDataset(path=hdf5_file,query=query)

    arguments = {'root': '/Volumes/PortableSSD/Projekte/datasets/celeba_dataset/IMG/celeba_aligned_small.h5py',
                    'query': ['Volumes','PortableSSD', 'Projekte', 'datasets', 'celeba_dataset', 'IMG', 'img_align_celeba'],
                 'placeholder': 1,
                 'storageType': "h5py",
                 'datasetName': "custom"}

    loaded_args = ConfigParser().getDatasetConfig()

    print(loaded_args)

    dataset = DatasetFactory.createDataset(loaded_args)
    counter = 0
    for i in dataset:
        print(i)
        #i[0].show()
        counter +=1
        if counter ==2:
            break
    # dataset = DatasetFactory.createDataset(arguments)

    #dataset.plot_image(45)
    pass

if __name__ == '__main__':
    main()


#%%
