
import numpy as np
import matplotlib.pylab as plt

def scale_percentile(matrix):
    if len(matrix.shape) == 2:
        matrix = matrix.reshape(matrix.shape + (1,))
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    mins = np.percentile(matrix, 0.5, axis=0)
    maxs = np.percentile(matrix, 99.5, axis=0) - mins
    matrix = 255*(matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d] if d > 1 else [w, h])
    matrix = matrix.clip(0, 255).astype(np.uint8)
    return matrix


def display_img_1b(img_1b_data, roi=None):
    if roi is not None:
        y,yh,x,xw = roi
        img_1b_data = img_1b_data[y:yh,x:xw]
    plt.imshow(scale_percentile(img_1b_data), cmap='gray')


def display_img_3b(img_3b_data, roi=None):
    if roi is not None:
        y,yh,x,xw = roi
        img_3b_data = img_3b_data[y:yh,x:xw,:]
    for i in [0,1,2]:
        plt.subplot(1,3,i+1)
        plt.imshow(scale_percentile(img_3b_data[:,:,i]), cmap='gray')
        plt.title("Channel %i" % i)

def display_img_8b(img_ms_data, roi=None):
    if roi is not None:
        y,yh,x,xw = roi
        img_ms_data = img_ms_data[y:yh,x:xw,:]
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.imshow(scale_percentile(img_ms_data[:,:,i]), cmap='gray')
        plt.title("Channel %i" % i)
