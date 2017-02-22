import keras_conf

from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Input, UpSampling2D, BatchNormalization, Reshape, Activation
from keras.models import Model


def create_encoding_layers(kernel=3, filter_size=64, pad=1, pool_size=2):
    return [
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size*2, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size*4, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size*8, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
    ]


def create_decoding_layers(kernel=3, filter_size=64, pad=1, pool_size=2):
    return[
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size*8, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size, pool_size)),
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size*4, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size, pool_size)),
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size*2, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size, pool_size)),
        ZeroPadding2D(padding=(pad, pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
    ]


def autoencoder_zero(n_classes, n_channels, input_width, input_height, n_filters_0=64):

    inputs = Input((n_channels, input_height, input_width))

    x = inputs
    encoding_layers = create_encoding_layers(filter_size=n_filters_0)
    decoding_layers = create_decoding_layers(filter_size=n_filters_0)
    for layer in encoding_layers:
        x = layer(x)
    for layer in decoding_layers:
        x = layer(x)

    x = Convolution2D(n_classes, 1, 1, border_mode='valid',)(x)
    x = Reshape((n_classes, input_height * input_width))(x)
    x = Activation('softmax')(x)
    outputs = Reshape((n_classes, input_height, input_width))(x)

    model = Model(input=inputs, output=outputs)
    return model
