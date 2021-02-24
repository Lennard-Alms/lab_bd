import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

class VGGFeatureExtractorMax:
    def __init__(self, window_size=(200,200)):
        vgg = tf.keras.applications.VGG16(include_top=False,
                                        weights='imagenet',
                                        input_shape=(window_size[0], window_size[1], 3))
        vgg_max_pooling = layers.Lambda(lambda x: K.max(x, axis=(1,2)))(vgg.output)
        vgg_flattened = layers.Flatten()(vgg_max_pooling)
        model = tf.keras.Model([vgg.input], vgg_flattened)
        print(model.output)
        self.model = model

    def get_output_shape(self):
        return self.model.output.shape[1:]

    def execute(self, items):
        prep = tf.keras.applications.vgg16.preprocess_input(items)
        return self.model.predict(prep)
