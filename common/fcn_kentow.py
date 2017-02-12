# https://raw.githubusercontent.com/k3nt0w/FCN_via_keras/master/FCN.py

from keras.layers import merge, Convolution2D, Deconvolution2D, MaxPooling2D, Input, Reshape, Permute, Cropping2D
from keras.layers.core import Activation
from keras.models import Model
from keras.utils.visualize_util import model_to_dot, plot


def FCN(n_classes, nb_channels, input_width, input_height):

    FCN_CLASSES = n_classes

    # (samples, channels, rows, cols)
    input_img = Input(shape=(nb_channels, input_height, input_width))

    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(input_img)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #  input_height/2, input_width/2
    
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #  input_height/4, input_width/4
    
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #  input_height/4, input_width/4
    
    # split layer
    p3 = x
    p3 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p3)
    

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # split layer
    p4 = x
    p4 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p4)
    p4 = Deconvolution2D(FCN_CLASSES, 4, 4,
            output_shape=(None, FCN_CLASSES, 30, 30),
            subsample=(2, 2),
            border_mode='valid')(p4)
    p4 = Cropping2D(cropping=((1, 1), (1, 1)))(p4)

    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    p5 = x
    p5 = Convolution2D(FCN_CLASSES, 1, 1, activation='relu')(p5)
    p5 = Deconvolution2D(FCN_CLASSES, 8, 8,
            output_shape=(None, FCN_CLASSES, 32, 32),
            subsample=(4, 4),
            border_mode='valid')(p5)
    p5 = Cropping2D(cropping=((2, 2), (2, 2)))(p5)

    # merge scores
    merged = merge([p3, p4, p5], mode='sum')
    x = Deconvolution2D(FCN_CLASSES, 16, 16,
            output_shape=(None, FCN_CLASSES, 232, 232),
            subsample=(8, 8),
            border_mode='valid')(merged)
    x = Cropping2D(cropping=((4, 4), (4, 4)))(x)
    x = Reshape((FCN_CLASSES, input_height*input_width))(x)
    x = Permute((2, 1))(x)
    out = Activation("softmax")(x)

    model = Model(input_img, out)
    return model


def visualize_model(model):
    plot(model, to_file='FCN_model.png', show_shapes=True)


def to_json(model):
    json_string = model.to_json()
    with open('FCN_via_Keras_architecture.json', 'w') as f:
        f.write(json_string)


# if __name__ == "__main__":
    # model = FCN()
    #visualize_model(model)
    #to_json(model)
