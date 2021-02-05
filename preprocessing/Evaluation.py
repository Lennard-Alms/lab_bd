import h5py
import numpy as np

def createCoffeemakerEvaluation(image_paths, hashingStrategy):
    x = 1

def find_image_and_patch(patch_id, patch_size):

    with h5py.File('hashes.hdf5', 'r') as f:
        loc = f['loc' + str(patch_size)][:]
    ppi = loc.shape[0]
    image_id = patch_id // ppi
    location = patch_id - image_id * ppi

    return image_id, loc[location].astype(np.dtype('i'))
