import keras_conf

from keras.layers import merge, Convolution2D, MaxPooling2D, Input, UpSampling2D, Flatten, Reshape, Activation, Lambda
from keras.models import Model
from keras.regularizers import l2


def _conv1(n_filters, size=3, l=0.01):
    return Convolution2D(n_filters, size, size,
                         activation='relu',
                         W_regularizer=l2(l),
                         border_mode='same')


def unet_zero(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=32):

    inputs = Input((n_channels, input_height, input_width))
    conv1 = _conv1(n_filters_0)(inputs)
    if deep:
        conv1 = _conv1(n_filters_0)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # (n_channels, input_height/2, input_width/2)

    conv2 = _conv1(n_filters_0*2)(pool1)
    if deep:
        conv2 = _conv1(n_filters_0*2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # (n_channels, input_height/4, input_width/4)

    conv3 = _conv1(n_filters_0*4)(pool2)
    if deep:
        conv3 = _conv1(n_filters_0*4)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # (n_channels, input_height/8, input_width/8)

    conv4 = _conv1(n_filters_0*8)(pool3)
    if deep:
        conv4 = _conv1(n_filters_0*8)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # (n_channels, input_height/16, input_width/16)

    conv5 = _conv1(n_filters_0*16)(pool4)
    if deep:
        conv5 = _conv1(n_filters_0*16)(conv5)
    # (n_channels, input_height/16, input_width/16)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = _conv1(n_filters_0*8)(up6)
    if deep:
        conv6 = _conv1(n_filters_0*8)(conv6)
    # (n_channels, input_height/8, input_width/8)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = _conv1(n_filters_0*4)(up7)
    if deep:
        conv7 = _conv1(n_filters_0*4)(conv7)
    # (n_channels, input_height/4, input_width/4)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = _conv1(n_filters_0*2)(up8)
    if deep:
        conv8 = _conv1(n_filters_0*2)(conv8)
    # (n_channels, input_height/4, input_width/4)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = _conv1(n_filters_0)(up9)
    if deep:
        conv9 = _conv1(n_filters_0)(conv9)
    # (n_channels, input_height/2, input_width/2)

    conv10 = Convolution2D(n_classes, 1, 1)(conv9)
    output = Reshape((n_classes, input_height * input_width))(conv10)
    output = Activation('softmax')(output)
    output = Reshape((n_classes, input_height, input_width))(output)

    model = Model(input=inputs, output=output)
    return model

    
def channels_ratio(input_layer):
    pass

    
def channels_ratio_shape(input_shape):
    h, w, k, n = input_shape
    
    
def unet_one(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=32):

    inputs = Input((n_channels, input_height, input_width))
    
    aug_inputs = inputs
    
    conv1 = _conv1(n_filters_0)(aug_inputs)
    if deep:
        conv1 = _conv1(n_filters_0)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # (n_channels, input_height/2, input_width/2)

    conv2 = _conv1(n_filters_0*2)(pool1)
    if deep:
        conv2 = _conv1(n_filters_0*2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # (n_channels, input_height/4, input_width/4)

    conv3 = _conv1(n_filters_0*4)(pool2)
    if deep:
        conv3 = _conv1(n_filters_0*4)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # (n_channels, input_height/8, input_width/8)

    conv4 = _conv1(n_filters_0*8)(pool3)
    if deep:
        conv4 = _conv1(n_filters_0*8)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # (n_channels, input_height/16, input_width/16)

    conv5 = _conv1(n_filters_0*16)(pool4)
    if deep:
        conv5 = _conv1(n_filters_0*16)(conv5)
    # (n_channels, input_height/16, input_width/16)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = _conv1(n_filters_0*8)(up6)
    if deep:
        conv6 = _conv1(n_filters_0*8)(conv6)
    # (n_channels, input_height/8, input_width/8)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = _conv1(n_filters_0*4)(up7)
    if deep:
        conv7 = _conv1(n_filters_0*4)(conv7)
    # (n_channels, input_height/4, input_width/4)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = _conv1(n_filters_0*2)(up8)
    if deep:
        conv8 = _conv1(n_filters_0*2)(conv8)
    # (n_channels, input_height/4, input_width/4)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = _conv1(n_filters_0)(up9)
    if deep:
        conv9 = _conv1(n_filters_0)(conv9)
    # (n_channels, input_height/2, input_width/2)

    conv10 = Convolution2D(n_classes, 1, 1)(conv9)
    output = Reshape((n_classes, input_height * input_width))(conv10)
    output = Activation('softmax')(output)
    output = Reshape((n_classes, input_height, input_width))(output)

    model = Model(input=inputs, output=output)
    return model