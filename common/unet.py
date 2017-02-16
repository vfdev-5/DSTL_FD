import keras_conf


from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Permute
from keras.layers import UpSampling2D, Reshape, Activation, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer


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


class Inverse(Layer):
    """Inverse Layer : 1/x """
    def __init__(self, **kwargs):
        super(Inverse, self).__init__(**kwargs)
        
    def call(self, x, mask=None):
        x = x + K.min(x) + 100.0
        eps = K.variable(value=K.epsilon())
        x = K.pow(x + eps, -1.0)
        mean = K.mean(x)
        std = K.std(x)
        x -= mean
        x /= std
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


class Normalization(Layer):
    """Normalization layer : x -> (x - mean)/std """
    def __init__(self, **kwargs):
        super(Normalization, self).__init__(**kwargs)
        
    def call(self, x, mask=None):
        mean = K.mean(x)
        std = K.std(x)
        x -= mean
        x /= std
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape           
        

def conv(input_layer, n_filters_0=16, deep=False, l=0.01):
    """
    """
    x = Convolution2D(n_filters_0, 3, 3,
                      activation='elu',
                      init='he_normal',
                      W_regularizer=l2(l),
                      border_mode='same')(input_layer)
    if deep:
        x = Convolution2D(2 * n_filters_0, 3, 3,
                          activation='elu',
                          init='he_normal',
                          W_regularizer=l2(l),
                          border_mode='same')(x)
    return x


def conv_downsample(input_layer, **kwargs):
    """
    """
    x = conv(input_layer, **kwargs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return x

    
def _merge(x1, x2):
    x12 = merge([x1, x2], mode='mul')
    x12 = Normalization()(x12)   
    return x12

    
def multiplication(input_layer, base_unit, **kwargs):    
    """
    x -> (x1 *x2)
    """
    x1 = base_unit(input_layer, **kwargs)
    x2 = base_unit(input_layer, **kwargs)
    return _merge(x1, x2)

    
def ratio(input_layer, base_unit, **kwargs):
    """
    x -> (x1 * 1/x2)
    """
    inverse_layer = Inverse()(input_layer)

    x1 = base_unit(input_layer, **kwargs)
    x2 = base_unit(inverse_layer, **kwargs)
    return _merge(x1, x2)
    

def composition(input_layer, base_unit, **kwargs):
    """
    x -> x1 + (x2 * 1/x3) + (x4 * x5) + 1/x6
    """
    inverse_layer = Inverse()(input_layer)

    x1 = base_unit(input_layer, **kwargs)
    x2 = base_unit(input_layer, **kwargs)
    x3 = base_unit(inverse_layer, **kwargs)
    x4 = base_unit(input_layer, **kwargs)
    x5 = base_unit(input_layer, **kwargs)
    x6 = base_unit(inverse_layer, **kwargs)

    x16 = merge([x1, x6], mode='sum')
    x23 = merge([x2, x3], mode='mul')
    x45 = merge([x4, x5], mode='mul')

    x1236 = merge([x16, x23], mode='sum')
    x123456 = merge([x1236, x45], mode='sum')

    x123456 = Normalization()(x123456)
    return x123456

    
def simple_end_cap(input_layer, n_classes, input_height, input_width):
    """
    
    - last activation by softmax leads to loss and weights equal NaN
    """
    x = Convolution2D(n_classes, 1, 1)(input_layer)
    x = Reshape((n_classes, input_height * input_width))(x)
    x = Permute((2, 1))(x)
    x = Activation('sigmoid')(x)
    x = Permute((2, 1))(x)
    x = Reshape((n_classes, input_height, input_width))(x)  
    return x

    
def end_cap(input_layer, n_classes, input_width, input_height):
    """
    - last activation by softmax leads to loss and weights equal NaN
    """
    return simple_end_cap(x, n_classes, input_width, input_height)

    
def upsample_merge(x_small, x_large):
    x_small = UpSampling2D(size=(2, 2))(x_small)
    return merge([x_small, x_large], mode='concat', concat_axis=1)

    
def unet_one_test(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=8):

    inputs = Input((n_channels, input_height, input_width))
    x = inputs
    #x = ratio(x, conv, n_filters_0=n_filters_0, deep=deep)
    x = multiplication(x, conv, n_filters_0=n_filters_0, deep=deep)    
    outputs = simple_end_cap(x, n_classes, input_width, input_height, activation='sigmoid')
    
    model = Model(input=inputs, output=outputs)
    return model
    
    
def unet_two(n_classes, n_channels, input_width, input_height, deep=False, n_filters_0=8):

    inputs = Input((n_channels, input_height, input_width))
    x = inputs
    
    x = BatchNormalization()(x)
    # Downsample and store
    x = composition(x, conv, n_filters_0=n_filters_0, deep=deep)
    x0 = x
    x = composition(x, conv_downsample, n_filters_0=n_filters_0 * 2, deep=deep)
    x1 = x
    x = composition(x, conv_downsample, n_filters_0=n_filters_0 * 4, deep=deep)
    x2 = x
    x = composition(x, conv_downsample, n_filters_0=n_filters_0 * 8, deep=deep)
    x3 = x
    x = composition(x, conv_downsample, n_filters_0=n_filters_0 * 16, deep=deep)

    x = conv(x, n_filters_0=n_filters_0 * 32)

    # Upsample and merge
    x = upsample_merge(x, x3)
    x = composition(x, conv, n_filters_0=n_filters_0 * 16, deep=deep)
    x = upsample_merge(x, x2)
    x = composition(x, conv, n_filters_0=n_filters_0 * 8, deep=deep)
    x = upsample_merge(x, x1)
    x = composition(x, conv, n_filters_0=n_filters_0 * 4, deep=deep)
    x = upsample_merge(x, x0)
    x = composition(x, conv, n_filters_0=n_filters_0 * 2, deep=deep)

    outputs = Convolution2D(n_classes, 1, 1)(x)
    outputs = Reshape((n_classes, input_height * input_width))(outputs)
    outputs = Activation('softmax')(outputs)
    outputs = Reshape((n_classes, input_height, input_width))(outputs)

    model = Model(input=inputs, output=outputs)
    return model
