from falconn import LSHIndex, LSHConstructionParameters, get_default_parameters
import falconn
import numpy as np
import glob, os
import cv2
from keras import Model
from os.path import join
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.transform import rescale, resize, downscale_local_mean
import scipy.ndimage as ndi



data_dir = '/run/media/markus/Shared/datasets/a2d2-preview/camera_lidar/20190401_121727/camera/cam_front_center/'

# os.chdir(data_dir)
numberOfBins = 32
patchSize = 90
images = np.empty((0,patchSize,patchSize,3), np.uint8)
red = np.empty((0,numberOfBins), np.int)
green = np.empty((0,numberOfBins), np.int)
blue = np.empty((0,numberOfBins), np.int)
bins = np.linspace(0,255,numberOfBins+1)


test_image = cv2.imread("/home/markus/Desktop/patch2.png")
test_red = np.histogram(test_image[:,:,0], bins)[0]
test_green = np.histogram(test_image[:,:,1], bins)[0]
test_blue = np.histogram(test_image[:,:,2], bins)[0]
test_feature = np.stack((test_red,test_green,test_blue), axis=1).reshape(numberOfBins*3).astype(np.float64)
for file in glob.glob(data_dir + "*.png"):
    im = cv2.imread(join(data_dir, file))
    patches = extract_patches_2d(im, (patchSize,patchSize), max_patches=100)
    _red = np.empty((patches.shape[0], numberOfBins))
    _green = np.empty((patches.shape[0], numberOfBins))
    _blue = np.empty((patches.shape[0], numberOfBins))

    patch_id = 0
    for patch in patches:
        _red[patch_id] = np.histogram(patch[:,:,0], bins)[0]
        _green[patch_id] = np.histogram(patch[:,:,1], bins)[0]
        _blue[patch_id] = np.histogram(patch[:,:,2], bins)[0]
        patch_id += 1
    blue = np.append(blue, _blue, axis=0)
    green = np.append(green, _green, axis=0)
    red = np.append(red, _red, axis=0)
    images = np.append(images, patches, axis=0)

feature_vectors = np.stack((red,green,blue), axis=2)
feature_vectors = feature_vectors.reshape(feature_vectors.shape[0], numberOfBins * 3)
# feature_vectors = model.predict(images)
# feature_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1)[:, np.newaxis]
# print(feature_vectors)

lsh_index = LSHIndex(get_default_parameters(
    num_points = feature_vectors.shape[0],
    dimension=numberOfBins*3,
    distance=falconn.DistanceFunction.EuclideanSquared))


lsh_index.setup(feature_vectors)
query = lsh_index.construct_query_object()
res = query.find_k_nearest_neighbors(test_feature, 10)
print(len(res))
print(res)

for index in res:
    cv2.imshow("h", images[index])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
