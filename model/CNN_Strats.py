class GeM(Layer):
    def __init__(self, **kwargs):
        super(GeM, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return tf.reduce_mean(x ** 4, axis=1) ** (1/4)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])



def vgg_vgg_att_gem():
    vgg = tf.keras.applications.VGG19(include_top=False,
                                      weights='imagenet',
                                      input_shape=shape_ex.shape)

    vgg_out = layers.Reshape((-1, vgg.output_shape[3]))(vgg.get_layer('block5_conv4').output)
    direct = GeM()(vgg_out)

    atten = layers.Attention()([vgg_out,vgg_out])
    atten = GeM()(atten)

    combined = layers.Lambda(lambda x: x[0] + x[1])([direct, atten])

    nomalized = layers.Lambda(lambda x: tf.linalg.normalize(x, axis=1))(combined)

    model = keras.models.Model(vgg.input, nomalized)

    return model

def vgg_vgg_att_l2_gem():
    vgg = tf.keras.applications.VGG19(include_top=False,
                                      weights='imagenet',
                                      input_shape=shape_ex.shape)

    vgg_out = layers.Reshape((-1, vgg.output_shape[3]))(vgg.get_layer('block5_conv4').output)
    direct = GeM()(vgg_out)

    atten = layers.Attention()([vgg_out,vgg_out])
    atten = GeM()(atten)

    combined = layers.Lambda(lambda x: tf.linalg.normalize(x[0], axis=1)[0] + tf.linalg.normalize(x[1], axis=1)[0])([direct, atten])

    nomalized = layers.Lambda(lambda x: tf.linalg.normalize(x, axis=1))(combined)

    model = keras.models.Model(vgg.input, nomalized)

    return model

def vgg_att_gem():
    vgg = tf.keras.applications.VGG19(include_top=False,
                                      weights='imagenet',
                                      input_shape=shape_ex.shape)

    vgg_out = layers.Reshape((-1, vgg.output_shape[3]))(vgg.get_layer('block5_conv4').output)

    atten = layers.Attention()([vgg_out,vgg_out])
    atten = GeM()(atten)

    nomalized = layers.Lambda(lambda x: tf.linalg.normalize(x, axis=1))(atten)

    model = keras.models.Model(vgg.input, nomalized)

    return model
