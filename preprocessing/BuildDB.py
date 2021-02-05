from .HelperFunctions import get_patches_from_image
import tensorflow as tf
import math
import numpy as np
import sys
import cv2
import time
import h5py
import gc

# OUTPUT: creates h-file with hash singatures, returns list of hyperplane normals and list of patches per image
def buildDB(paths, patch_sizes=[(200,200),(400,400)], overlap=0.5, signature_size=4096, batch_size=10):

    hyperplane_normals_list = list()
    patches_per_image_list = list()

    # for each patch size we need to do a full run of predictions for each image
    for patch_size in patch_sizes:

        # vgg needs a specific input shape thats why we declare it inside the patch loop
        vgg = tf.keras.applications.VGG16(include_top=False,
                                      weights='imagenet',
                                      input_shape=(patch_size[0], patch_size[1], 3))

        # caluclate the maximum ammount of loops untill we have all images
        max_batches = math.ceil(len(paths) / batch_size)

        # record time for animation eta
        start_time = time.time()
        print('exit0')
        # make a test run to find output shape for each image this implies that all images have the same dimensions
        test_run = get_patches_from_image(cv2.imread(paths[0]), patch_size, overlap)
        test_pred = tf.convert_to_tensor(test_run[:2], dtype=test_run.dtype)
        test_pred = vgg.predict(tf.keras.applications.vgg16.preprocess_input(test_pred), verbose=0)
        print('exit1')

        # save the patches per image so we can do a back calculation later and see what patch came from what image
        patches_per_image = test_run.shape[0]
        patches_per_image_list.append(patches_per_image)
        print('exit2')

        # initialize ETA
        eta = round((time.time() - start_time) * len(paths) * 2 / 60, 2)
        print('exit3')

        # calculate the hyperplane normals for hashing
        pred_dim = test_pred.shape[1] * test_pred.shape[2] * test_pred.shape[3]
        hyperplane_normals = np.random.normal(0,1,(pred_dim,signature_size))
        print('exit4')

        # save hyperplane normals for prediction later
        hyperplane_normals_list.append(hyperplane_normals)
        print('exit5')

        # free the variable since we have no further use for it
        del(test_run)
        del(test_pred)
        gc.collect()
        print('exit6')

        # record time for animation eta
        start_time = time.time()

        # do the whole hashing for each batch
        for batch_idx in range(max_batches):

            # animation for observing during runtime
            print("Size " + str(patch_size) + " Batch " + str(batch_idx+1) + "/" + str(max_batches) + " ETA: " + str(eta) + " min")

            # calculate the range of images
            path_idx_start = batch_size * batch_idx
            path_idx_end = path_idx_start + batch_size
            print('exit7')

            # load the images and create the patches
            patches = np.concatenate([get_patches_from_image(cv2.imread(path), patch_size, overlap) for path in paths[path_idx_start:path_idx_end]])
            print('exit8')

            # use vgg to calculate the feature vectors
            patches = tf.convert_to_tensor(patches, dtype=patches.dtype)
            patches = vgg.predict(tf.keras.applications.vgg16.preprocess_input(patches), verbose=1)
            print('exit9')

            # flatten the feature vectors
            patches = patches.reshape((patches.shape[0],pred_dim))

            # calculate the hash signatures
            patches = np.dot(patches, hyperplane_normals) < 0

            # save in file with option "a" => read write if exists esle create
            with h5py.File("hashes.hdf5", "w") as f:

                # open h file dataset or create a new one if this is the first iteration
                if str(patch_size) in f:
                    hfile = f[str(patch_size)]
                else:
                    # save as boolean file with '?' parameter
                    hfile = f.create_dataset(str(patch_size), (0, signature_size) , dtype='?', maxshape=(None, signature_size))

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
        del(vgg)
        gc.collect()

    # save ppi and hash normals
    patches_per_image_list = np.array(patches_per_image_list)
    hyperplane_normals_list = np.array(hyperplane_normals_list)
    with h5py.File("hashes.hdf5", "w") as f:
        hfile = f.create_dataset('ppi', patches_per_image_list.shape , dtype='i')
        hfile[:] = patches_per_image_list
        hfile = f.create_dataset('hn', hyperplane_normals_list.shape , dtype='f')
        hfile[:] = patches_per_image_list

    # free variables
    del(patches_per_image_list)
    del(hyperplane_normals_list)
    gc.collect()























#
