import tensorflow as tf
import numpy as np
import cv2
import h5py
import gc

# OUTPUT: creates h-file with hash singatures of queries.
# this assumes that queries contain only relevant parts and no unnecessary background
# queries are squares
def buildQuery(paths, hyperplane_normals_list, patch_sizes=[(200,200),(400,400)]):

    # load images
    images = [cv2.imread(path) for path in paths]

    for normals_index, patch_size in enumerate(patch_sizes):
        print("Size " + str(patch_size))

        # select hyperplane normals for hashing
        hyperplane_normals = hyperplane_normals_list[normals_index]

        # vgg needs a specific input shape thats why we declare it inside the patch loop
        vgg = tf.keras.applications.VGG16(include_top=False,
                                      weights='imagenet',
                                      input_shape=(patch_size[0], patch_size[1], 3))

        # generate patches via rescaling
        patches = np.array([cv2.resize(img, patch_size) for img in images])

        # use vgg to calculate the feature vectors
        patches = tf.convert_to_tensor(patches, dtype=patches.dtype)
        patches = vgg.predict(tf.keras.applications.vgg16.preprocess_input(patches), verbose=1)

        # flatten the feature vectors
        patches = patches.reshape((patches.shape[0],-1))

        # calculate the hash signatures
        patches = np.dot(patches, hyperplane_normals) < 0

        # save in file with option "a" => read write if exists esle create
        with h5py.File("hashes.hdf5", "w") as f:
            hfile = f.create_dataset('Q' + str(patch_size), patches.shape , dtype='?')

            # save the calculated hashes in the h file dataset
            hfile[:] = patches

        # free variables
        del(patches)
        del(vgg)
        gc.collect()























#
