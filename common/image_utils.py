import os
import logging

import numpy as np
import cv2

# GDAL
import gdal
import gdalconst
from gdal_pansharpen import gdal_pansharpen


DATA_3_BANDS='../input/three_band/'
DATA_16_BANDS='../input/sixteen_band/'
GENERATED_DATA_16_BANDS = "../input/generated/"
TRAIN_LABELS = '../input/labels'

TRAIN_TILES = '../input/train'
TRAIN_LABEL_TILES = '../input/train/labels'
TRAIN_LABEL_TILES_1D = '../input/train/labels_1d'


if not os.path.exists(GENERATED_DATA_16_BANDS):
    os.makedirs(GENERATED_DATA_16_BANDS)

if not os.path.exists(TRAIN_LABELS):
    os.makedirs(TRAIN_LABELS)

if not os.path.exists(TRAIN_TILES):
    os.makedirs(TRAIN_TILES)

if not os.path.exists(TRAIN_LABEL_TILES):
    os.makedirs(TRAIN_LABEL_TILES)

if not os.path.exists(TRAIN_LABEL_TILES_1D):
    os.makedirs(TRAIN_LABEL_TILES_1D)
   

def generate_aligned_swir(image_id):
    """
    Method to create a swir aligned image file
    :param image_id:
    :return:
    """
    outfname = get_filename(image_id, 'swir_aligned')
    if os.path.exists(outfname):
        logging.warn("File '%s' is already existing" % outfname)
        return

    img_pan = get_image_data(image_id, 'pan')
    img_swir = get_image_data(image_id, 'swir')
    img_swir_aligned = align_images(img_pan, img_swir, roi=[0, 0, 500, 500], warp_mode=cv2.MOTION_EUCLIDEAN)

    img_swir_aligned = img_swir if img_swir_aligned is None else img_swir_aligned
    imwrite(outfname, img_swir_aligned)


def generate_pansharpened(image_id, image_type):
    """
    Method to create pansharpened images from multispectral or swir images
    Created file is placed in GENERATED_DATA_16_BANDS folder
    
    :image_type: 'ms' or 'swir'
    """
    outfname = get_filename(image_id, image_type + '_pan')
    if os.path.exists(outfname):
        logging.warn("File '%s' is already existing" % outfname)
        return 
    
    fname = get_filename(image_id, image_type)
    fname_pan = get_filename(image_id, 'pan')
    gdal_pansharpen(['', fname_pan, fname, outfname])  


def get_tile_filename(image_id, xoffset, yoffset, image_type):
    if image_type == '17b':
        dir_path = TRAIN_TILES
        ext = 'tif'
    elif image_type == 'label':
        dir_path = TRAIN_LABEL_TILES
        ext = 'tif'
    elif image_type == 'label_1d':
        dir_path = TRAIN_LABEL_TILES_1D
        ext = 'tif'
    else:
        raise Exception("Unknown image type: {}".format(image_type))
    return os.path.join(dir_path, "{}_{}_{}.{}".format(image_id, xoffset, yoffset, ext))

    
def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    ext = 'tif'
    if image_type == '3b':
        data_path = DATA_3_BANDS
        suffix = ''
    elif image_type == 'pan':
        data_path = DATA_16_BANDS
        suffix = '_P'
    elif image_type == 'ms':
        data_path = DATA_16_BANDS
        suffix = '_M'
    elif image_type == 'swir':
        data_path = DATA_16_BANDS
        suffix = '_A'
    elif image_type == 'swir_aligned':
        data_path = GENERATED_DATA_16_BANDS
        suffix = '_A_aligned'
    elif image_type == 'ms_pan':
        data_path = GENERATED_DATA_16_BANDS
        suffix = '_M_P'
    elif image_type == 'swir_pan' or \
            image_type == 'swir_aligned_pan':
        data_path = GENERATED_DATA_16_BANDS
        suffix = '_A_P'
    elif image_type == 'label_1d':
        data_path = TRAIN_LABELS
        suffix = '_1d'
    elif image_type == 'label':
        data_path = TRAIN_LABELS
        suffix = ''
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)
        
    return os.path.join(data_path, "{}{}.{}".format(image_id, suffix, ext))


def get_image_tile_data(fname, return_shape_only=False):
    """
    Method to get image tile data as np.array
    """
    img = gdal.Open(fname, gdalconst.GA_ReadOnly)
    assert img, "Image file is not found: {}".format(fname)
    if return_shape_only:
        return img.RasterYSize, img.RasterXSize, img.RasterCount

    img_data = img.ReadAsArray()
    if len(img_data.shape) == 3:
        return img_data.transpose([1, 2, 0])
    return img_data



def get_image_data(image_id, image_type, return_shape_only=False):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = gdal.Open(fname, gdalconst.GA_ReadOnly)
    assert img, "Image is not found: id={}, type={}".format(image_id, image_type)
    if return_shape_only:
        return (img.RasterYSize, img.RasterXSize, img.RasterCount)

    img_data = img.ReadAsArray()
    if len(img_data.shape) == 3:
        return img_data.transpose([1, 2, 0])
    return img_data


def imwrite(filename, data):
    driver = gdal.GetDriverByName("GTiff")
    data_type = to_gdal(data.dtype)
    nb_bands = data.shape[2]
    width = data.shape[1]
    height = data.shape[0]

    dst_dataset = driver.Create(filename, width, height, nb_bands, data_type)
    for band_index in range(1,nb_bands+1):
        dst_band = dst_dataset.GetRasterBand(band_index)
        dst_band.WriteArray(data[:,:,band_index-1], 0, 0)


def to_gdal(dtype):
    """ Method to convert numpy data type to Gdal data type """
    if dtype == np.uint8:
        return gdal.GDT_Byte
    elif dtype == np.int16:
        return gdal.GDT_Int16
    elif dtype == np.int32:
        return gdal.GDT_Int32
    elif dtype == np.uint16:
        return gdal.GDT_UInt16
    elif dtype == np.uint32:
        return gdal.GDT_UInt32
    elif dtype == np.float32:
        return gdal.GDT_Float32
    elif dtype == np.float64:
        return gdal.GDT_Float64
    elif dtype == np.complex64:
        return gdal.GDT_CFloat32
    elif dtype == np.complex128:
        return gdal.GDT_CFloat64
    else:
        return gdal.GDT_Unknown

    
def normalize(in_img, q_min=0.5, q_max=99.5):
    """
    Normalize image in [0.0, 1.0]
    """
    w, h, d = in_img.shape
    img = in_img.copy()
    img = np.reshape(img, [w * h, d]).astype(np.float64)
    mins = np.percentile(img, q_min, axis=0)
    maxs = np.percentile(img, q_max, axis=0) - mins
    maxs[(maxs < 0.0001) & (maxs > -0.0001)] = 0.0001
    img = (img - mins[None, :]) / maxs[None, :]
    img = img.clip(0.0, 1.0)
    img = np.reshape(img, [w, h, d])
    return img    
    

def get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def align_images(img_master, img_slave, roi, warp_mode=cv2.MOTION_EUCLIDEAN):
    """
    Code taken from http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    """
    fx = img_master.shape[1] * 1.0 / img_slave.shape[1]
    fy = img_master.shape[0] * 1.0 / img_slave.shape[0]
    roi_slave = [int(roi[0] / fx), int(roi[1] / fy), int(roi[2] / fx), int(roi[3] / fy)]

    img_slave_roi = img_slave[roi_slave[1]:roi_slave[3], roi_slave[0]:roi_slave[2], :].astype(np.float32)

    img_master_roi = img_master[roi[1]:roi[3], roi[0]:roi[2]].astype(np.float32)
    img_master_roi = cv2.resize(img_master_roi, dsize=(img_slave_roi.shape[1], img_slave_roi.shape[0]))

    img_master_roi = get_gradient(img_master_roi)
    img_slave_roi = get_gradient(img_slave_roi)

    img_slave_aligned = np.zeros_like(img_slave)

    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    ll = img_slave.shape[2]
    for i in range(ll):
        # Set the stopping criteria for the algorithm.
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-1)
        try:
            cc, warp_matrix = cv2.findTransformECC(img_master_roi, img_slave_roi[:, :, i], warp_matrix, warp_mode,
                                                   criteria)

            height, width, _ = img_slave.shape
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                # Use Perspective warp when the transformation is a Homography
                img_slave_aligned[:, :, i] = cv2.warpPerspective(img_slave[:, :, i],
                                                                 warp_matrix,
                                                                 (width, height),
                                                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                                                 borderMode=cv2.BORDER_REFLECT
                                                                 )
            else:
                # Use Affine warp when the transformation is not a Homography
                img_slave_aligned[:, :, i] = cv2.warpAffine(img_slave[:, :, i],
                                                            warp_matrix,
                                                            (width, height),
                                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                                            borderMode=cv2.BORDER_REFLECT
                                                            );
        except Exception as e:
            logging.error("Failed to find warp matrix: %s" % str(e))
            return None

    return img_slave_aligned
