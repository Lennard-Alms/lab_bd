import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import numpy as np
import gc
from .HelperFunctions import get_image, get_patches_from_image

class VGGFeatureExtractorMax:
    def __init__(self, window_size=(200,200), mutation_strategy = None):
        self.window_size = window_size
        self.mutation_strategy = mutation_strategy
        vgg = tf.keras.applications.VGG16(include_top=False,
                                        weights='imagenet',
                                        input_shape=(window_size[0], window_size[1], 3))
        vgg_max_pooling = layers.Lambda(lambda x: K.max(x, axis=(1,2)))(vgg.output)
        vgg_flattened = layers.Flatten()(vgg_max_pooling)
        model = tf.keras.Model([vgg.input], vgg_flattened)
        self.model = model

    def get_output_shapes(self):
        shapes = []
        shapes.append(self.model.output.shape[1:])
        if self.mutation_strategy is not None:
            shapes.append((1,))
        return shapes

    def get_no_of_outputs(self):
        no_of_outputs = 1
        if self.mutation_strategy is not None:
            no_of_outputs += 1
        return no_of_outputs

    def get_dataset_name_postfixes(self):
        postfixes = [""]
        if self.mutation_strategy is not None:
            postfixes.append("label")
        return postfixes

    def execute(self, items):
        labels = []
        patches = []
        for path in items:
            im = get_image(path)
            _p = get_patches_from_image(im, window_size=self.window_size, window_overlap=0.5)
            patches.append(_p)
        patches = np.concatenate(patches)

        if self.mutation_strategy is not None:
            for item in patches:
                item, label = self.mutation_strategy.mutate(item)
                labels.append(label)
        prep = tf.keras.applications.vgg16.preprocess_input(patches)
        labels = np.array(labels)[:,np.newaxis]
        gc.collect()
        return self.model.predict(prep), labels

    def set_mutation_strategy(self, strategy):
        self.mutation_strategy = strategy
