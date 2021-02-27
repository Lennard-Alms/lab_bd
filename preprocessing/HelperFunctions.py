import math
import numpy as np
import cv2
import tensorflow as tf
import gc
from PIL import Image, ImageFile


def get_patches_from_image(image, window_size, window_overlap):
    number_of_tiles_x = int(math.ceil(image.shape[1] / (window_size[1] * (1 - window_overlap))))
    number_of_tiles_y = int(math.ceil(image.shape[0] / (window_size[0] * (1 - window_overlap))))
    tile_centers_x = np.linspace(
        window_size[1] // 2,
        image.shape[1] - window_size[1] // 2,
        num=number_of_tiles_x,
        endpoint=True
    )
    tile_centers_y = np.linspace(
        window_size[0] // 2,
        image.shape[0] - window_size[0] // 2,
        num=number_of_tiles_y,
        endpoint=True
    )
    patches = []
    for x_center in tile_centers_x:
        for y_center in tile_centers_y:
            y_start = int(np.maximum(y_center - window_size[0] // 2, 0))
            y_end = int(np.minimum(y_center + window_size[0] // 2, image.shape[0]))
            x_start = int(np.maximum(x_center - window_size[1] // 2, 0))
            x_end = int(np.minimum(x_center + window_size[1] // 2, image.shape[1]))
            patch = image[y_start:y_end, x_start:x_end]
            patch = cv2.resize(patch, (200,200))
            patches.append(patch)
    return np.array(patches)

def get_patch_locations(image, window_size, window_overlap):
    number_of_tiles_x = int(math.ceil(image.shape[1] / (window_size[1] * (1 - window_overlap))))
    number_of_tiles_y = int(math.ceil(image.shape[0] / (window_size[0] * (1 - window_overlap))))
    tile_centers_x = np.linspace(
        window_size[1] // 2,
        image.shape[1] - window_size[1] // 2,
        num=number_of_tiles_x,
        endpoint=True
    )
    tile_centers_y = np.linspace(
        window_size[0] // 2,
        image.shape[0] - window_size[0] // 2,
        num=number_of_tiles_y,
        endpoint=True
    )
    patches = []
    for x_center in tile_centers_x:
        for y_center in tile_centers_y:
            y_start = int(np.maximum(y_center - window_size[0] // 2, 0))
            y_end = int(np.minimum(y_center + window_size[0] // 2, image.shape[0]))
            x_start = int(np.maximum(x_center - window_size[1] // 2, 0))
            x_end = int(np.minimum(x_center + window_size[1] // 2, image.shape[1]))
            patches.append(np.array([y_start,y_end,x_start,x_end]))
    return np.array(patches)

def pil_loader(path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_image(path_or_image, pil=False):
    if isinstance(path_or_image, str):
        if pil:
            return np.array(pil_loader(path_or_image))
        else:
            return cv2.imread(path_or_image)
    else:
        return path_or_image

def predict_w_vgg(patches, patch_size):
    vgg = tf.keras.applications.VGG16(include_top=False,
                                  weights='imagenet',
                                  input_shape=(patch_size[0], patch_size[1], 3))
    patches = tf.convert_to_tensor(patches, dtype=patches.dtype)
    patches = vgg.predict(tf.keras.applications.vgg16.preprocess_input(patches), verbose=1)
    patches = patches.reshape((patches.shape[0],-1))
    del(vgg)
    gc.collect()
    return patches
