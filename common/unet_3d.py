import keras_conf

from keras.layers import merge, Convolution3D, Convolution2D, MaxPooling3D, Input, Permute
from keras.layers import UpSampling3D, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer


def conv(input_layer, n_filters_0=16, deep=False, l=0.01, size=3):
    """
    """
    x = Convolution3D(n_filters_0, size, size, size,
                      activation='elu',
                      W_regularizer=l2(l),
                      border_mode='same')(input_layer)
    if deep:
        x = Convolution3D(2 * n_filters_0, size, size, size,
                          activation='elu',
                          W_regularizer=l2(l),
                          border_mode='same')(x)
    return x


def unet_downsample(input_layer, n_filters_0, deep):
    """
    x -> [conv(x), conv^2(x))/2, conv^3(x))/4, conv^4(x))/8]
    """
    x1 = conv(input_layer, n_filters_0, deep)
    # (n_channels, input_height, input_width)
    
    x = MaxPooling3D(pool_size=(2, 2, 2))(x1)
    x2 = conv(x, 2*n_filters_0, deep)
    # (n_channels/2, input_height/2, input_width/2)
    
    x = MaxPooling3D(pool_size=(2, 2, 2))(x2)
    x3 = conv(x, 4*n_filters_0, deep)
    # (n_channels/4, input_height/4, input_width/4)
    
    x = MaxPooling3D(pool_size=(2, 2, 2))(x3)
    x = conv(x, 8*n_filters_0, deep)
    # (n_channels/8 input_height/8, input_width/8)

    return x1, x2, x3, x


def unet_upsample(x1, x2, x3, x4, n_filters_0, deep):
    """
    [x1, x2, x3, x4, x5] -> conv([conv([conv([conv([x5*2, x4])*2, x3])*2, x2])*2, x1])
                
    """
    x = UpSampling3D(size=(2, 2, 2))(x4)
    x = merge([x, x3], mode='concat', concat_axis=1)
    x = conv(x, 4*n_filters_0, deep)
    # (input_height*2, input_width*2)

    x = UpSampling3D(size=(2, 2, 2))(x)
    x = merge([x, x2], mode='concat', concat_axis=1)
    x = conv(x, 2*n_filters_0, deep)
    # (input_height*4, input_width*4)

    x = UpSampling3D(size=(2, 2, 2))(x)
    x = merge([x, x1], mode='concat', concat_axis=1)
    x = conv(x, n_filters_0, deep)
    # (input_height*8, input_width*8)

    return x
    

def unet_base(input_layer, n_filters_0, deep):    
    x1, x2, x3, x = unet_downsample(input_layer, n_filters_0, deep)
    x = unet_upsample(x1, x2, x3, x, n_filters_0, deep)
    return x


def original_termination(input_layer, n_classes):
    """
    U-net original termination
    """
    return Convolution2D(n_classes, 1, 1, activation='sigmoid')(input_layer)


def unet_zero(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=32):

    inputs = Input((1, n_channels, input_height, input_width))
    x = unet_base(inputs, n_filters_0, deep)
    x = conv(x, 1)
    x = Reshape((n_channels, input_height, input_width))(x)
    outputs = original_termination(x, n_classes)
    model = Model(input=inputs, output=outputs)
    return model
