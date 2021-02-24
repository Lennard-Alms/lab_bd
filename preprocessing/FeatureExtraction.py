import tf


class VGGFeatureExtractorMax:
    def __init__(self, window_size=(200,200)):
        self.vgg = tf.keras.applications.VGG16(include_top=False,
                                        weights='imagenet',
                                        input_shape=(window_size[0], window_size[1], 3))
    def execute(self, items):
        prep = tf.keras.applications.vgg16.preprocess_input(items)
        return self.vgg.predict(prep)
