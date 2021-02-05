import numpy as np
import cv2


def add_patch_to_image(image, patch):
    if patch.shape[2] == 4:
        mask = (patch[:,:,3] != 0)
        patch[~mask] = 0
        patch = patch[:,:,0:3]
        r1,r2 = np.random.randint(0, image.shape[0] - patch.shape[0]), np.random.randint(0, image.shape[1] - patch.shape[1])
        im_patch = image[r1:r1 + patch.shape[0], r2:r2+patch.shape[1]]
        im_patch[mask] = 0
        im_patch += patch
        image[r1:r1 + patch.shape[0], r2:r2+patch.shape[1]] = im_patch
        return image
    else:
        rgbsum = patch.sum(axis=2)
        mask = (rgbsum[:,:] != 255*3)
        patch[~mask] = 0
        r1,r2 = np.random.randint(0, image.shape[0] - patch.shape[0]), np.random.randint(0, image.shape[1] - patch.shape[1])
        im_patch = image[r1:r1 + patch.shape[0], r2:r2+patch.shape[1]]
        im_patch[mask] = 0
        im_patch += patch
        image[r1:r1 + patch.shape[0], r2:r2+patch.shape[1]] = im_patch
        return image


def add_test_image(images, evaluation_patches, prob, sample_size):
    """Adds randomly a set of image patches to a subset of images"""
    positive = []
    for i, im in enumerate(images):
        if np.random.rand() < prob / sample_size:
            evaluation_image = None
            if isinstance(evaluation_patches, list):
                r = np.random.randint(0,len(evaluation_patches))
                evaluation_image = evaluation_patches[r]
                positive.append(i)
                im = add_patch_to_image(im, evaluation_image)
    return images, positive


class PatchMutation:
    def __init__(self, patches, mutation_probability=0.2):
        """Takes patches of arbitrary size and mutates incoming images"""
        self.patches = [cv2.imread(path) for path in patches]
        self.mutation_probability = mutation_probability

    def mutate(self, image):
        mutation_label = 0
        if np.random.rand() < self.mutation_probability:
            evaluation_image = None
            r = np.random.randint(0, len(self.patches))
            mutation_label = r + 1
            evaluation_image = self.patches[r]
            image = add_patch_to_image(image, evaluation_image)
        return image, mutation_label
