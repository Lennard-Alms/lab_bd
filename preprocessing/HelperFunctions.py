import math
import numpy as np
import cv2

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

def get_image(path_or_image):
    if isinstance(path_or_image, str):
        return cv2.imread(path_or_image)
    else:
        return path_or_image
