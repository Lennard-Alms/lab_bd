import tensorflow as tf
import numpy as np
import cv2
import h5py
import gc

# OUTPUT: creates h-file with hash singatures of queries.
# this assumes that queries contain only relevant parts and no unnecessary background
# queries are squares
def buildQuery(paths, patch_sizes=[(200,200),(400,400)]):

    # load images
    images = [cv2.imread(path) for path in paths]

    for normals_index, patch_size in enumerate(patch_sizes):
        print("Size " + str(patch_size))
        print('exit0')

        # select hyperplane normals for hashing
        with h5py.File("hashes.hdf5", "a") as f:
            set_name = 'hn' + str(patch_size)
            hyperplane_normals = f[set_name][:]
        print('exit1')
        print(hyperplane_normals.shape)

        # vgg needs a specific input shape thats why we declare it inside the patch loop
        vgg = tf.keras.applications.VGG16(include_top=False,
                                      weights='imagenet',
                                      input_shape=(patch_size[0], patch_size[1], 3))
        print('exit2')

        # generate patches via rescaling
        patches = np.array([cv2.resize(img, patch_size) for img in images])
        print('exit3')

        # use vgg to calculate the feature vectors
        patches = tf.convert_to_tensor(patches, dtype=patches.dtype)
        patches = vgg.predict(tf.keras.applications.vgg16.preprocess_input(patches), verbose=1)
        print('exit4')

        # flatten the feature vectors
        patches = patches.reshape((patches.shape[0],-1))
        print('exit5')

        # calculate the hash signatures
        patches = np.dot(patches, hyperplane_normals) < 0
        print('exit6')
        # save in file with option "a" => read write if exists esle create
        set_name = 'Q' + str(patch_size)
        with h5py.File("hashes.hdf5", "a") as f:
            if set_name in f:
                hfile = f[set_name]
            else:
                hfile = f.create_dataset(set_name, patches.shape , dtype='?')
            # save the calculated hashes in the h file dataset
            hfile[:] = patches
        print('exit7')

        # free variables
        del(patches)
        del(hyperplane_normals)
        del(vgg)
        gc.collect()























#
