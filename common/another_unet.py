#
# Another u-net from [pix2pix](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix)
#

import numpy as np
from keras.models import Model
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Convolution2D, UpSampling2D, Deconvolution2D
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization



def unet(n_classes, n_channels, input_width, input_height, bn_mode=2, use_deconv=False):
    """
        Network U-Net
        Compile with Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    """
    nb_filters = 64

    bn_axis = 1
    min_s = min(input_width, input_height)

    inputs = Input(shape=(n_channels, input_height, input_width))

    # Prepare encoder filters
    nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # Encoder
    list_encoder = [Convolution2D(list_nb_filters[0], 3, 3,
                                  subsample=(2, 2), border_mode="same")(inputs)]
    for i, f in enumerate(list_nb_filters[1:]):
        conv = conv_block_unet(list_encoder[-1], f, bn_mode, bn_axis)
        list_encoder.append(conv)

    # Prepare decoder filters
    list_nb_filters = list_nb_filters[:-2][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)

    # Decoder
    _up_conv_block_unet = deconv_block_unet if use_deconv else up_conv_block_unet
    list_decoder = [_up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                                       list_nb_filters[0], bn_mode, bn_axis, dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        # Dropout only on first few layers        
        d = True if i < 2 else False
        conv = _up_conv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)

    x = Activation("relu")(list_decoder[-1])
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(n_classes, 3, 3, name="last_conv", border_mode="same")(x)
    outputs = Activation("sigmoid")(x)

    return Model(input=inputs, output=outputs, name="U-net")


def conv_block_unet(x, f, bn_mode, bn_axis, bn=True, subsample=(2,2)):

    x = LeakyReLU(0.2)(x)
    x = Convolution2D(f, 3, 3, subsample=subsample, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)

    return x


def up_conv_block_unet(x, x2, f, bn_mode, bn_axis, bn=True, dropout=False):

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(f, 3, 3, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = merge([x, x2], mode='concat', concat_axis=bn_axis)

    return x


def deconv_block_unet(x, x2, f, h, w, batch_size, bn_mode, bn_axis, bn=True, dropout=False):

    o_shape = (batch_size, h * 2, w * 2, f)
    x = Activation("relu")(x)
    x = Deconvolution2D(f, 3, 3, output_shape=o_shape, subsample=(2, 2), border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = merge([x, x2], mode='concat', concat_axis=bn_axis)

    return x
