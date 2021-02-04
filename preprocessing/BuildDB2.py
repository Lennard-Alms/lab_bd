import numpy as np
import h5py
import cv2
from .HelperFunctions import get_patches_from_image
import tensorflow as tf

def get_features(patches, vgg):
    x = tf.keras.applications.vgg19.preprocess_input(patches)
    return vgg.predict(x)

def buildDB2(image_names, image_batch_size = 4, signature_size = 4096, compress=False, window_size=(200,200), window_overlap=0.5):
    vgg = tf.keras.applications.VGG19(include_top=False,
                                        weights='imagenet',
                                        input_shape=(window_size[0], window_size[1], 3))
    feature_size = 6 * 6 * 512
    projection_normals = np.array([np.random.normal(0,1,feature_size) for _ in range(signature_size)])
    file = h5py.File('output.h5', 'w')
    batch_index = 0
    num_batches = len(image_names) // image_batch_size
    if(compress):
        save_size = signature_size // 8
    else:
        save_size = signature_size

    labels = []

    h5_hashes = file.create_dataset("hashes", (0, save_size), maxshape=(None,save_size))
    while len(image_names) > image_batch_size * batch_index:
        print("Batch ", str(batch_index + 1), " / " + str(num_batches))

        index_start = image_batch_size * batch_index
        index_end = index_start + image_batch_size
        batch = image_names[index_start:index_end]
        im_array = [cv2.imread(idx) for idx in batch]
        patches = []
        for i, img in enumerate(im_array):
            if(img is None):
                print("None File:", batch[i])
        else:
            res = get_patches_from_image(img, window_size, window_overlap)
            patches.append(np.array(res))
            _labels = [batch[i]] * len(res)
            labels = labels + _labels

        # labels = np.array(labels)
        patches = np.concatenate(patches)
        features = get_features(patches, vgg)
        features = features.reshape(features.shape[0], -1)
        if compress:
            hashes = np.packbits(np.dot(features, projection_normals.T) < 0, axis = 1)
        else:
            hashes = np.dot(features, projection_normals.T) < 0
        # print(np.packbits(hashes, axis=1).shape)
        f_index = h5_hashes.shape[0]
        # l_index = h5_labels.shape[0]
        h5_hashes.resize(h5_hashes.shape[0] + hashes.shape[0], axis = 0)
        h5_hashes[f_index:] = hashes

        # h5_labels.resize(h5_labels.shape[0] + len(labels), axis = 0)
        # h5_labels[l_index:] = np.array(labels)[:,np.newaxis]

        batch_index += 1
    f.close()
    with open('labels.txt', 'w') as f:
        for item in labels:
            f.write("%s\n" % item)
