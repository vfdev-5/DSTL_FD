from keras.layers import merge, Convolution2D, MaxPooling2D, Input, UpSampling2D, Flatten, Reshape, Activation
from keras.models import Model
from keras.backend import set_image_dim_ordering
from keras.regularizers import l2

set_image_dim_ordering('th')


def _conv1(n_filters, size=3, lr=0.01):
    return Convolution2D(n_filters, size, size,
                         activation='relu',
                         W_regularizer=l2(lr),
                         border_mode='same')


def autoencoder_zero(n_classes, n_channels, input_width, input_height):

    inputs = Input((n_channels, input_height, input_width))
    conv1 = _conv1(32)(inputs)
    conv1 = _conv1(32)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # (n_channels, input_height/2, input_width/2)

    conv2 = _conv1(64)(pool1)
    conv2 = _conv1(64)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # (n_channels, input_height/4, input_width/4)

    conv3 = _conv1(128)(pool2)
    conv3 = _conv1(128)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # (n_channels, input_height/8, input_width/8)

    conv4 = _conv1(256)(pool3)
    conv4 = _conv1(256)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # (n_channels, input_height/16, input_width/16)

    conv5 = _conv1(512)(pool4)
    conv5 = _conv1(512)(conv5)
    # (n_channels, input_height/16, input_width/16)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = _conv1(256)(up6)
    conv6 = _conv1(256)(conv6)
    # (n_channels, input_height/8, input_width/8)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = _conv1(128)(up7)
    conv7 = _conv1(128)(conv7)
    # (n_channels, input_height/4, input_width/4)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = _conv1(64)(up8)
    conv8 = _conv1(64)(conv8)
    # (n_channels, input_height/4, input_width/4)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = _conv1(32)(up9)
    conv9 = _conv1(32)(conv9)
    # (n_channels, input_height/2, input_width/2)

    conv10 = Convolution2D(n_classes, 1, 1)(conv9)
    output = Flatten()(conv10)
    output = Activation('softmax')(output)
    output = Reshape((n_classes, input_height, input_width))(output)

    model = Model(input=inputs, output=output)
    return model
