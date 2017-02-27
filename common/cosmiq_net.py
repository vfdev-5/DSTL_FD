#
# Cosmiq V2 network inspired from https://gist.github.com/hagerty/5e1fb0eef76553f7d26dfb4d136b3443
#
import numpy as np
from keras.models import Model
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Convolution2D, UpSampling2D, Deconvolution2D
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization


def cosmiqnet_v2(input_a_shape, input_b_shape, output_n_classes, n_filters=64, size=3, beta=0.9):
    """
    
    :param input_a_shape: shape of larger size input, (h1, w1, nc1)
    :param input_b_shape: shape of smaller size input, (h2, w2, nc2) such that h2 = h1 / scale, w2 = w1 / scale
    :param output_shape:
    """
    assert input_a_shape[0] * 1.0 / input_b_shape[0] == input_a_shape[1] * 1.0 / input_b_shape[1], "Inputs image size should be proportional"
    inputs = [
        Input(shape=(input_a_shape[0],) + input_a_shape[:2]),
        Input(shape=(input_b_shape[0],) + input_b_shape[:2]),
    ]
    
    scale = int(input_a_shape[0] * 1.0 / input_b_shape[0])


    x_a = Convolution2D(n_filters, size, size, subsample=(scale, scale), border_mode="same")

    
    pass
