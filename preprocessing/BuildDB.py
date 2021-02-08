from .HelperFunctions import get_patches_from_image
from .HelperFunctions import get_patch_locations
from .HelperFunctions import get_image
import tensorflow as tf
import math
import numpy as np
import sys
import time
import h5py
import gc
import cv2

# OUTPUT: creates h-file with hash singatures, returns list of hyperplane normals and list of patches per image
def buildDB(paths, patch_sizes=[(200,200),(400,400)], overlap=0.5, signature_size=4096, batch_size=10, mutationStrategy=None):

    # all patches are resized to 200x200
    vgg = tf.keras.applications.VGG16(include_top=False,
                                  weights='imagenet',
                                  input_shape=(200, 200, 3))

    # for each patch size we need to do a full run of predictions for each image
    for patch_size_idx, patch_size in enumerate(patch_sizes):

        # caluclate the maximum ammount of loops untill we have all images
        max_batches = math.ceil(len(paths) / batch_size)

        # record time for animation eta
        start_time = time.time()

        # make a test run to find output shape for each image this implies that all images have the same dimensions
        test_run = get_patches_from_image(get_image(paths[0]), patch_size, overlap)
        test_pred = tf.convert_to_tensor(test_run[:2], dtype=test_run.dtype)
        test_pred = vgg.predict(tf.keras.applications.vgg16.preprocess_input(test_pred), verbose=0)
        test_loc = get_patch_locations(get_image(paths[0]), patch_size, overlap)

        # initialize ETA
        eta = round((time.time() - start_time) * len(paths) * 2 / 60, 2)

        # calculate the hyperplane normals for hashing
        pred_dim = test_pred.shape[1] * test_pred.shape[2] * test_pred.shape[3]
        hyperplane_normals = np.random.normal(0,1,(pred_dim,signature_size))

        # write over the old file if we are in the first iteration - deletes the old h file
        open_strat = 'w' if patch_size_idx == 0 else 'a'

        with h5py.File('hashes.hdf5', open_strat) as f:
            # save hyperplanes for prediction
            set_name = 'hn' + str(patch_size)
            if set_name in f:
                hfile = f[set_name]
            else:
                hfile = f.create_dataset(set_name, hyperplane_normals.shape , dtype='f')
            hfile[:] = hyperplane_normals

            # save location of patches
            set_name = 'loc' + str(patch_size)
            if set_name in f:
                hfile = f[set_name]
            else:
                hfile = f.create_dataset(set_name, test_loc.shape , dtype='i')
            hfile[:] = test_loc.astype(np.dtype('i'))

        # free the variable since we have no further use for it
        del(test_run)
        del(test_loc)
        del(test_pred)
        gc.collect()

        # record time for animation eta
        start_time = time.time()

        # do the whole hashing for each batch
        for batch_idx in range(max_batches):

            # animation for observing during runtime
            print('Size ' + str(patch_size) + ' Batch ' + str(batch_idx+1) + '/' + str(max_batches) + ' ETA: ' + str(eta) + ' min')

            # calculate the range of images
            path_idx_start = batch_size * batch_idx
            path_idx_end = path_idx_start + batch_size

            # load the images and create the patches
            patches = [get_patches_from_image(get_image(path), patch_size, overlap) for path in paths[path_idx_start:path_idx_end]]

            patches = np.concatenate([cv2.resize(pat, (200,200)) for pat in patches])

            if mutationStrategy is not None:
                labels = np.zeros(patches.shape[0])
                for p_idx in range(patches.shape[0]):
                    patches[p_idx], labels[p_idx] = mutationStrategy.mutate(patches[p_idx])

                with h5py.File('hashes.hdf5', 'a') as f:
                    # open h file dataset or create a new one if this is the first iteration
                    set_name = 'lab' + str(patch_size)
                    if set_name in f:
                        hfile = f[set_name]
                    else:
                        # save as boolean file with '?' parameter
                        hfile = f.create_dataset(set_name, (0,) , dtype='i', maxshape=(None,))

                    # save the calculated hashes in the h file dataset
                    hfile_index = hfile.shape[0]
                    hfile.resize(hfile.shape[0] + labels.shape[0], axis = 0)
                    hfile[hfile_index:] = labels


            # use vgg to calculate the feature vectors
            patches = tf.convert_to_tensor(patches, dtype=patches.dtype)
            patches = vgg.predict(tf.keras.applications.vgg16.preprocess_input(patches), verbose=1)

            # flatten the feature vectors
            patches = patches.reshape((patches.shape[0],pred_dim))

            # calculate the hash signatures
            patches = np.dot(patches, hyperplane_normals) < 0

            # save in file with option 'a' => read write if exists esle create
            with h5py.File('hashes.hdf5', 'a') as f:

                # open h file dataset or create a new one if this is the first iteration
                set_name = 'db' + str(patch_size)
                if set_name in f:
                    hfile = f[set_name]
                else:
                    # save as boolean file with '?' parameter
                    hfile = f.create_dataset(set_name, (0, signature_size) , dtype='?', maxshape=(None, signature_size))

                # save the calculated hashes in the h file dataset
                hfile_index = hfile.shape[0]
                hfile.resize(hfile.shape[0] + patches.shape[0], axis = 0)
                hfile[hfile_index:] = patches

            # calculate eta for the animation
            eta = round((time.time() - start_time) * (max_batches-batch_idx-1) / 60, 2)
            start_time = time.time()

            # free variables
            del(patches)
            gc.collect()

        # free variables
        del(hyperplane_normals)
        gc.collect()

    # free variables
    del(vgg)
    gc.collect()























#
