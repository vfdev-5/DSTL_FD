#
# Helper methods to visualize layer outputs
#
import numpy as np
import matplotlib.pylab as plt

import keras.backend as K


def get_layer_output_f(layer_name, model):
    inputs = [K.learning_phase()] + model.inputs
    output_layer = model.get_layer(name=layer_name)
    outputs = output_layer.output
    return K.function(inputs, [outputs])
    
    
def compute_layer_output(input_data, layer_output_f):
    return layer_output_f([0] + [input_data])


def compute_layer_outputs(input_data, model, layer_output_f_dict={}):
    """
    Method to compute all layer outputs on `input_data` for a given `model`
    :return: tuple of pairs: [("layer_name_1", ndarray), ...]
    """
    layer_outputs = []
    for layer in model.layers:
        if layer in model.input_layers or layer in model.output_layers:
            continue
        print layer.name
        if layer.name not in layer_output_f_dict:
            layer_output_f_dict[layer.name] = get_layer_output_func(layer.name, model)
        layer_outputs.append((layer.name, compute_layer_output(input_data, layer_output_f_dict[layer.name])))
    return layer_outputs


def display_layer_output(layer_name, layer_output, **kwargs):
    plt.suptitle("%s" % layer_name)
    nc = layer_output.shape[1]
    n_cols = 4 
    n_rows = int(np.floor(nc / n_cols))
    for i in range(nc):
        plt.subplot(n_rows,n_cols,i+1)
        plt.imshow(layer_output[0,i,:,:], interpolation='none')
        plt.colorbar()
        plt.title("%i" % i)