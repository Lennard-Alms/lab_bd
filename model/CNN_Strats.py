import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import keras
from keras.engine.topology import Layer
import math

class GeM(Layer):
    def __init__(self, exp, **kwargs):
        super(GeM, self).__init__(**kwargs)
        self.exp = exp

    def call(self, x, mask=None):
        return tf.reduce_mean(x ** self.exp, axis=1) ** (1/self.exp)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])



def vgg_vgg_att_gem(in_shape, exp):
    vgg = tf.keras.applications.VGG19(include_top=False,
                                      weights='imagenet',
                                      input_shape=in_shape)

    vgg_out = layers.Reshape((-1, vgg.output_shape[3]))(vgg.get_layer('block5_conv4').output)
    direct = GeM(exp)(vgg_out)

    atten = layers.Attention()([vgg_out,vgg_out])
    atten = GeM(exp)(atten)

    combined = layers.Lambda(lambda x: x[0] + x[1])([direct, atten])

    nomalized = layers.Lambda(lambda x: tf.linalg.normalize(x, axis=1)[0])(combined)

    model = keras.models.Model(vgg.input, nomalized)

    return model

def vgg_vgg_att_l2_gem(in_shape, exp):
    vgg = tf.keras.applications.VGG19(include_top=False,
                                      weights='imagenet',
                                      input_shape=in_shape)

    vgg_out = layers.Reshape((-1, vgg.output_shape[3]))(vgg.get_layer('block5_conv4').output)
    direct = GeM(exp)(vgg_out)

    atten = layers.Attention()([vgg_out,vgg_out])
    atten = GeM(exp)(atten)

    combined = layers.Lambda(lambda x: tf.linalg.normalize(x[0], axis=1)[0] + tf.linalg.normalize(x[1], axis=1)[0])([direct, atten])

    nomalized = layers.Lambda(lambda x: tf.linalg.normalize(x, axis=1)[0])(combined)

    model = keras.models.Model(vgg.input, nomalized)

    return model

def vgg_att_gem(in_shape, exp):
    vgg = tf.keras.applications.VGG19(include_top=False,
                                      weights='imagenet',
                                      input_shape=in_shape)

    vgg_out = layers.Reshape((-1, vgg.output_shape[3]))(vgg.get_layer('block5_conv4').output)

    atten = layers.Attention()([vgg_out,vgg_out])
    atten = GeM(exp)(atten)

    nomalized = layers.Lambda(lambda x: tf.linalg.normalize(x, axis=1)[0])(atten)

    model = keras.models.Model(vgg.input, nomalized)

    return model

def vgg_gem(in_shape, exp):
    vgg = tf.keras.applications.VGG19(include_top=False,
                                      weights='imagenet',
                                      input_shape=in_shape)

    vgg_out = layers.Reshape((-1, vgg.output_shape[3]))(vgg.get_layer('block5_conv4').output)

    gem = GeM(exp)(vgg_out)

    nomalized = layers.Lambda(lambda x: tf.linalg.normalize(x, axis=1)[0])(gem)

    model = keras.models.Model(vgg.input, nomalized)

    return model

def vgg_r_mac(in_shape, exp, sum=True):
    vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet',
                                          input_shape=in_shape)

    exp_out = layers.Lambda(lambda x: x ** exp)(vgg.get_layer('block5_conv4').output)

    long_side = max(exp_out.shape[1], exp_out.shape[2])

    halfs = math.floor(long_side/2)
    quater = math.floor(long_side/4)
    eights = math.floor(long_side/8)
    sixtheens = math.floor(long_side/16)

    pool_halfs = layers.AveragePooling2D((halfs,halfs),(quater,quater), padding='VALID')(exp_out)
    pool_quater = layers.AveragePooling2D((quater,quater),(eights,eights), padding='VALID')(exp_out)
    pool_eights = layers.AveragePooling2D((eights,eights),(sixtheens,sixtheens), padding='VALID')(exp_out)

    pool_halfs = layers.Lambda(lambda x: x ** (1/exp))(pool_halfs)
    pool_quater = layers.Lambda(lambda x: x ** (1/exp))(pool_quater)
    pool_eights = layers.Lambda(lambda x: x ** (1/exp))(pool_eights)

    pool_halfs = layers.Lambda(lambda x: tf.linalg.normalize(x, axis=3)[0])(pool_halfs)
    pool_quater = layers.Lambda(lambda x: tf.linalg.normalize(x, axis=3)[0])(pool_quater)
    pool_eights = layers.Lambda(lambda x: tf.linalg.normalize(x, axis=3)[0])(pool_eights)

    if sum:

        pool_halfs = layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=(1,2)))(pool_halfs)
        pool_quater = layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=(1,2)))(pool_quater)
        pool_eights = layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=(1,2)))(pool_eights)

        r_mac = layers.Lambda(lambda x: x[0] + x[1] + x[2])([pool_halfs, pool_quater, pool_eights])

        nomalized = layers.Lambda(lambda x: tf.linalg.normalize(x, axis=1)[0])(r_mac)

        return keras.models.Model(vgg.input, nomalized)

    else:

        return keras.models.Model(vgg.input, [pool_halfs,pool_quater,pool_eights])

























    #sdf
