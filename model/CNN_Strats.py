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
        return tf.reduce_mean(x ** self.exp, axis=(1,2)) ** (1/self.exp)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

def normalize_model(tensor_in):
    return layers.Lambda(lambda x: tf.linalg.normalize(x, axis=3)[0])(tensor_in)

def combine_model(tensor_list_in):
    return layers.Lambda(lambda x: tf.add_n(x))(tensor_list_in)

def reduce_model(tensor_in):
    return layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=(1,2)))(tensor_in)

def self_attention_model(tensor_in):
    atten = layers.Reshape((-1, tensor_in.shape[3]))(tensor_in)
    atten = layers.Attention()([atten,atten])
    atten = layers.Reshape(tensor_in.shape)(atten)
    return atten

def mac_model(tensor_in, exp):
    gem = GeM(exp)(tensor_in)
    return gem

def rmac_model(tensor_in, exp):
    exp_out = layers.Lambda(lambda x: x ** exp)(tensor_in)

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

    pool_halfs = normalize_model(pool_halfs)
    pool_quater = normalize_model(pool_quater)
    pool_eights = normalize_model(pool_eights)

    return [pool_halfs, pool_quater, pool_eights]




def build_model(in_shape, exp, vgg_output=False, attention=False, mac=False, rmac=False, regions=False):
    vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet',
                                          input_shape=in_shape)

    vgg_out = vgg.get_layer('block5_conv4').output

    if attention:
        atten = self_attention_model(vgg_out)

    if vgg_output and attention and mac:
        vgg_out = mac_model(vgg_out, exp)
        vgg_out = normalize_model(vgg_out)
        atten = mac_model(atten, exp)
        atten = normalize_model(atten)
        combined = combine_model([vgg_out, atten])
        combined = normalize_model(combined)
        return keras.models.Model(vgg.input, combined)

    if vgg_output and mac:
        vgg_out = mac_model(vgg_out, exp)
        vgg_out = normalize_model(vgg_out)
        return keras.models.Model(vgg.input, vgg_out)

    if attention and mac:
        atten = mac_model(atten, exp)
        atten = normalize_model(atten)
        return keras.models.Model(vgg.input, atten)

    if vgg_output and attention and rmac:
        vgg_out = rmac_model(vgg_out)
        atten = rmac_model(atten)
        combined = combine_model(vgg_out + atten)
        combined = reduce_model(vgg_out)
        combined =  normalize_model(combined)
        return keras.models.Model(vgg.input, combined)

    if vgg_output and rmac:
        vgg_out = rmac_model(vgg_out)
        vgg_out = combine_model(vgg_out)
        vgg_out = reduce_model(vgg_out)
        vgg_out = normalize_model(vgg_out)
        return keras.models.Model(vgg.input, vgg_out)

    if attention and rmac:
        atten = rmac_model(atten)
        atten = combine_model(atten)
        atten = reduce_model(atten)
        atten = normalize_model(atten)
        return keras.models.Model(vgg.input, atten)

    if vgg_output and attention and rmac:
        vgg_out = rmac_model(vgg_out)
        atten = rmac_model(atten)
        c0 = combine_model([vgg_out[0], atten[0]])
        c1 = combine_model([vgg_out[1], atten[1]])
        c2 = combine_model([vgg_out[2], atten[2]])
        c0 = normalize_model(c0)
        c1 = normalize_model(c1)
        c2 = normalize_model(c2)
        return keras.models.Model(vgg.input, [c0,c1,c2])

    if vgg_output and rmac:
        vgg_out = rmac_model(vgg_out)
        return keras.models.Model(vgg.input, vgg_out)

    if attention and rmac:
        atten = rmac_model(atten)
        return keras.models.Model(vgg.input, atten)
























    #sdf
