import sys, os
sys.path.append(os.getcwd())

import h5pyImageDataset




def main():
    hdf5_file = '/Volumes/PortableSSD/Projekte/datasets/celeba_dataset/IMG/celeba_aligned_small.h5py'
    query = ['Volumes','PortableSSD', 'Projekte', 'datasets', 'celeba_dataset', 'IMG', 'img_align_celeba']
    dataset = h5pyImageDataset.H5PYImageDataset(path=hdf5_file,query=query)

    dataset.plot_image(45)
    pass

if __name__ == '__main__':
    main()


#%%
