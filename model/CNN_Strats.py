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
    last_axis = len(tensor_in.shape) - 1
    return layers.Lambda(lambda x: tf.linalg.normalize(x, axis=last_axis)[0])(tensor_in)

def combine_model(tensor_list_in):
    return layers.Lambda(lambda x: tf.add_n(x))(tensor_list_in)

def reduce_model(tensor_in):
    return layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=(1,2)))(tensor_in)

def self_attention_model(tensor_in):
    atten = layers.Reshape((-1, tensor_in.shape[3]))(tensor_in)
    atten = layers.Attention()([atten,atten])
    atten = layers.Reshape(tensor_in.shape[1:])(atten)
    return atten

def mac_model(tensor_in, exp):
    gem = GeM(exp)(tensor_in)
    return gem

def rmac_model(tensor_in, exp, depth):
    exp_out = layers.Lambda(lambda x: x ** exp)(tensor_in)
    short_side = min(exp_out.shape[1], exp_out.shape[2])

    maps = []

    for d in range(1,depth+1):
        size = math.floor(short_side/d)
        stride = math.floor(size * 0.2)

        pool = layers.AveragePooling2D((size,size),(stride,stride), padding='VALID')(exp_out)
        pool = layers.Lambda(lambda x: x ** (1/exp))(pool)
        pool = normalize_model(pool)
        maps.append(pool)

    return maps




def build_model(in_shape, exp, vgg_output=False, attention=False, mac=False, rmac=False, regions=False, depth=10):
    vgg = tf.keras.applications.VGG16(include_top=False,
                                          weights='imagenet',
                                          input_shape=in_shape)

    vgg_out = vgg.get_layer('block5_conv3').output

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
        vgg_out = rmac_model(vgg_out, exp, depth)
        atten = rmac_model(atten, exp, depth)
        comb = []
        for d in range(depth):
            c = combine_model([vgg_out[d], atten[d]])
            c = reduce_model(c)
            comb.append(c)
        combined = combine_model(comb)
        combined =  normalize_model(combined)
        return keras.models.Model(vgg.input, combined)

    if vgg_output and rmac:
        vgg_out = rmac_model(vgg_out, exp, depth)
        comb = []
        for d in range(depth):
            c = reduce_model(vgg_out[d])
            comb.append(c)
        vgg_out = combine_model(comb)
        vgg_out = normalize_model(vgg_out)
        return keras.models.Model(vgg.input, vgg_out)

    if attention and rmac:
        atten = rmac_model(atten, exp, depth)
        comb = []
        for d in range(depth):
            c = reduce_model(atten[d])
            comb.append(c)
        atten = combine_model(comb)
        atten = normalize_model(atten)
        return keras.models.Model(vgg.input, atten)

    if vgg_output and attention and regions:
        vgg_out = rmac_model(vgg_out, exp, depth)
        atten = rmac_model(atten, exp, depth)
        comb = []
        for d in range(depth):
            c = combine_model([vgg_out[d], atten[d]])
            c = normalize_model(c)
            comb.append(c)
        return keras.models.Model(vgg.input, comb)

    if vgg_output and regions:
        vgg_out = rmac_model(vgg_out, depth)
        return keras.models.Model(vgg.input, vgg_out)

    if attention and regions:
        atten = rmac_model(atten, depth)
        return keras.models.Model(vgg.input, atten)
























    #sdf
