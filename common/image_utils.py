import os
import logging

import numpy as np
import cv2

# GDAL
import gdal
import gdalconst
from gdal_pansharpen import gdal_pansharpen


DATA_3_BANDS='../input/three_band'
DATA_16_BANDS='../input/sixteen_band'
GENERATED_DATA = '../input/generated'
TRAIN_DATA = '../input/train'
TEST_DATA = '../input/test'


GENERATED_LABELS = os.path.join(GENERATED_DATA, 'labels')
TRAIN_LABEL_TILES = os.path.join(TRAIN_DATA, 'labels')
TRAIN_LABEL_TILES_1D = os.path.join(TRAIN_DATA, 'labels_1d')


if not os.path.exists(GENERATED_DATA):
    os.makedirs(GENERATED_DATA)

if not os.path.exists(GENERATED_LABELS):
    os.makedirs(GENERATED_LABELS)

if not os.path.exists(TRAIN_DATA):
    os.makedirs(TRAIN_DATA)

if not os.path.exists(TEST_DATA):
    os.makedirs(TEST_DATA)

# if not os.path.exists(TRAIN_LABEL_TILES):
#     os.makedirs(TRAIN_LABEL_TILES)
#
# if not os.path.exists(TRAIN_LABEL_TILES_1D):
#     os.makedirs(TRAIN_LABEL_TILES_1D)


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
    
    img_swir_aligned = compute_aligned_image(img_pan, img_swir)
    imwrite(outfname, img_swir_aligned)


def generated_upsampled_swir(image_id, image_type):
    """
    Method to generate an upsampled swir image
    :param image_id:
    :param image_type: 'swir' or 'swir*'
    """
    assert 'swir' in image_type, "Image type should be derived from 'swir'"
    outfname = get_filename(image_id, image_type + '_upsampled')
    if os.path.exists(outfname):
        logging.warn("File '%s' is already existing" % outfname)
        return

    h, w, _ = get_image_data(image_id, 'pan', return_shape_only=True)
    img_swir = get_image_data(image_id, image_type)
    img_swir_upsampled = cv2.resize(img_swir, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    imwrite(outfname, img_swir_upsampled)


def generate_pansharpened(image_id, image_type):
    """
    Method to create pansharpened images from multispectral or swir images
    Created file is placed in GENERATED_DATA folder
    
    :image_type: 'ms' or 'ms*'
    """
    assert 'ms' in image_type, "Image type should be derived from 'ms'"
    outfname = get_filename(image_id, image_type + '_pan')
    if os.path.exists(outfname):
        logging.warn("File '%s' is already existing" % outfname)
        return 
    
    fname = get_filename(image_id, image_type)
    fname_pan = get_filename(image_id, 'pan')
    gdal_pansharpen(['', fname_pan, fname, outfname])  


def get_tile_filename(image_id, xoffset, yoffset, image_type):
    if image_type == '17b':
        dir_path = TRAIN_DATA
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
        data_path = GENERATED_DATA
        suffix = '_A_aligned'
    elif image_type == 'ms_pan':
        data_path = GENERATED_DATA
        suffix = '_M_P'
    elif image_type == 'swir_upsampled' or \
            image_type == 'swir_aligned_upsampled':
        data_path = GENERATED_DATA
        suffix = '_A_P'
    elif image_type == 'label_1d':
        data_path = GENERATED_LABELS
        suffix = '_1d'
    elif image_type == 'label':
        data_path = GENERATED_LABELS
        suffix = ''
    elif image_type == '17b':
        data_path = TRAIN_DATA
        suffix = ''
    elif image_type == '17b_test':
        data_path = TEST_DATA
        suffix = ''
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)
        
    return os.path.join(data_path, "{}{}.{}".format(image_id, suffix, ext))


def print_image_info(image_id, image_type):
    
    fname = get_filename(image_id, image_type)
    img = gdal.Open(fname, gdalconst.GA_ReadOnly)
    assert img, "Image file is not found: {}".format(fname)

    print("Image size:", img.RasterYSize, img.RasterXSize, img.RasterCount)
    print("Metadata:", img.GetMetadata_List())
    print("MetadataDomainList:", img.GetMetadataDomainList())
    print("Description:", img.GetDescription())
    print("ProjectionRef:", img.GetProjectionRef())
    print("GeoTransform:", img.GetGeoTransform())


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
    assert dst_dataset is not None, "File '%s' is not created" % filename
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


def compute_alignment_warp_matrix(img_master, img_slave, roi, warp_mode=cv2.MOTION_TRANSLATION):
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

    height, width, ll = img_slave.shape

    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        mean_warp_matrix = np.zeros((3, 3), dtype=np.float32)
    else:
        mean_warp_matrix = np.zeros((2, 3), dtype=np.float32)

    for i in range(ll):

        # Set the warp matrix to identity.
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        # Set the stopping criteria for the algorithm.
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 0.01)
        try:
            cc, warp_matrix = cv2.findTransformECC(img_master_roi,
                                                   img_slave_roi[:, :, i],
                                                   warp_matrix,
                                                   warp_mode,
                                                   criteria)
        except Exception as e:
            logging.error("Failed to find warp matrix: %s" % str(e))
            return warp_matrix

        mean_warp_matrix += warp_matrix

    mean_warp_matrix *= 1.0/ll
    return mean_warp_matrix


def compute_aligned_image(img_master, img_slave):
    # Compute mean warp matrix
    roi=[0,0,500,500]
    warp_mode = cv2.MOTION_TRANSLATION
    mean_warp_matrix = np.zeros((2, 3), dtype=np.float32)
    mean_warp_matrix[0, 0] = 1.0
    mean_warp_matrix[1, 1] = 1.0
    tx = []
    ty = []
    n = 3
    for i in range(n):
        for j in range(n):
            warp_matrix = compute_alignment_warp_matrix(img_master, img_slave, roi=roi, warp_mode=warp_mode)
            tx.append(warp_matrix[0, 2])
            ty.append(warp_matrix[1, 2])
            roi[0] = i * 500
            roi[1] = j * 500
            roi[2] += roi[0]
            roi[3] += roi[1]    
    
    tx = np.median(tx)
    ty = np.median(ty)
    mean_warp_matrix[0, 2] = tx
    mean_warp_matrix[1, 2] = ty
    
    #print "mean_warp_matrix :"
    #print mean_warp_matrix
    
    img_slave_aligned = np.zeros_like(img_slave)
    height, width, ll = img_slave.shape
    for i in range(ll):
        # Use Affine warp when the transformation is not a Homography
        img_slave_aligned[:, :, i] = cv2.warpAffine(img_slave[:, :, i],
                                                    mean_warp_matrix,
                                                    (width, height),
                                                    flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                                                    borderMode=cv2.BORDER_REPLICATE
                                                    )
   
    img_slave_aligned = img_slave if img_slave_aligned is None else img_slave_aligned
    return img_slave_aligned


def align_images(img_master, img_slave, roi, warp_mode=cv2.MOTION_TRANSLATION):
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
    height, width, ll = img_slave.shape
    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        mean_warp_matrix = np.zeros((3, 3), dtype=np.float32)
    else:
        mean_warp_matrix = np.zeros((2, 3), dtype=np.float32)

    for i in range(ll):

        # Set the warp matrix to identity.
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        # Set the stopping criteria for the algorithm.
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 0.01)
        try:
            cc, warp_matrix = cv2.findTransformECC(img_master_roi,
                                                   img_slave_roi[:, :, i],
                                                   warp_matrix,
                                                   warp_mode,
                                                   criteria)
        except Exception as e:
            logging.error("Failed to find warp matrix: %s" % str(e))
            return None

        mean_warp_matrix += warp_matrix

    mean_warp_matrix *= 1.0/ll
    #print "Mean Warp matrix: ",
    #print mean_warp_matrix

    for i in range(ll):

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use Perspective warp when the transformation is a Homography
            img_slave_aligned[:, :, i] = cv2.warpPerspective(img_slave[:, :, i],
                                                             mean_warp_matrix,
                                                             (width, height),
                                                             flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                                                             borderMode=cv2.BORDER_REPLICATE
                                                             )
        else:
            # Use Affine warp when the transformation is not a Homography
            img_slave_aligned[:, :, i] = cv2.warpAffine(img_slave[:, :, i],
                                                        mean_warp_matrix,
                                                        (width, height),
                                                        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                                                        borderMode=cv2.BORDER_REPLICATE
                                                        )

    return img_slave_aligned
    
    
def make_ratios_vegetation(img_17b):
    """
        Method creates an image of all possible band ratios
        
        - panchromatic[0] / MS[5] = Trees, Crops, Misc manmade structures (of trees) 
        - panchromatic[0] / MS[4] = Trees, Crops, Misc manmade structures (of trees) 
        - MS[1] / MS[5] = Trees, Crops, Misc manmade structures (of trees) 
        - MS[2] / MS[5] = Trees, Crops, Misc manmade structures (of trees) 
        - MS[6] / MS[4] = Trees, Crops, Misc manmade structures (of trees) 
        - MS[6] / MS[5] = Trees, Crops, Misc manmade structures (of trees) 
        - MS[7] / MS[4] = Trees, Crops, Misc manmade structures (of trees) 
        - MS[7] / MS[5] = Trees, Crops, Misc manmade structures (of trees) 
        - MS[7] / MS[10:17] = Trees, Crops, Misc manmade structures (of trees) 
        - MS[8] / MS[4] = Trees, Crops, Misc manmade structures (of trees) 
        - MS[8:17] / MS[5] = Trees, Crops, Misc manmade structures (of trees) 
    """
    h, w, n = img_17b.shape
    
    out_n = 23
    out = np.zeros((h, w, out_n), dtype=np.float32)
    def _ratio(i, j):
        return img_17b[:,:,i] / (img_17b[:,:,j] + 0.00001)

    c = 0
    out[:,:,c] = _ratio(0, 5); c+= 1
    out[:,:,c] = _ratio(0, 4); c+= 1
    out[:,:,c] = _ratio(1, 5); c+= 1
    out[:,:,c] = _ratio(2, 5); c+= 1
    out[:,:,c] = _ratio(6, 4); c+= 1
    out[:,:,c] = _ratio(6, 5); c+= 1
    out[:,:,c] = _ratio(7, 5); c+= 1
    out[:,:,c] = _ratio(7, 4); c+= 1
    out[:,:,c] = _ratio(8, 4); c+= 1
    out[:,:,c] = _ratio(7, 10); c+= 1
    out[:,:,c] = _ratio(7, 11); c+= 1
    out[:,:,c] = _ratio(7, 12); c+= 1
    out[:,:,c] = _ratio(7, 13); c+= 1
    out[:,:,c] = _ratio(7, 14); c+= 1
    out[:,:,c] = _ratio(7, 15); c+= 1
    out[:,:,c] = _ratio(7, 16); c+= 1
    out[:,:,c] = _ratio(9, 5); c+= 1
    out[:,:,c] = _ratio(10, 5); c+= 1
    out[:,:,c] = _ratio(11, 5); c+= 1
    out[:,:,c] = _ratio(12, 5); c+= 1
    out[:,:,c] = _ratio(13, 5); c+= 1
    out[:,:,c] = _ratio(14, 5); c+= 1
    out[:,:,c] = _ratio(15, 5); c+= 1
    return out

    
def compute_mean_std_on_tiles(trainset_ids):
    """
    Method to compute mean/std tile image
    :return: mean_tile_image, std_tile_image
    """
    ll = len(trainset_ids)
    tile_id = trainset_ids[0]
    mean_tile_image = get_image_tile_data(os.path.join(TRAIN_DATA,tile_id)).astype(np.float)
    # Init mean/std images
    std_tile_image = np.power(mean_tile_image, 2)

    for i, tile_id in enumerate(trainset_ids[1:]):
        logging.info("-- %i/%i | %s" % (i+2, ll, tile_id))
        tile = get_image_tile_data(os.path.join(TRAIN_DATA,tile_id)).astype(np.float)
        mean_tile_image += tile
        std_tile_image += np.power(tile, 2)
        
    mean_tile_image *= 1.0/ll
    std_tile_image *= 1.0/ll
    std_tile_image -= np.power(mean_tile_image, 2)
    std_tile_image = np.sqrt(std_tile_image)
    return mean_tile_image, std_tile_image


def compute_mean_std_on_images(trainset_ids):
    """
    Method to compute mean/std input image
    :return: mean_image, std_image
    """
    ll = len(trainset_ids)
    image_id = trainset_ids[0]
    img_17b = get_image_data(image_id, '17b').astype(np.float)
    # Init mean/std images
    mean_image = np.zeros((3349, 3404, 17))
    h, w, _ = img_17b.shape
    mean_image[:h, :w, :] += img_17b
    std_image = np.power(mean_image, 2)

    for i, image_id in enumerate(trainset_ids[1:]):
        logging.info("-- %i/%i | %s" % (i + 2, ll, image_id))
        img_17b = get_image_data(image_id, '17b').astype(np.float)
        h, w, _ = img_17b.shape
        mean_image[:h, :w, :] += img_17b
        std_image[:h, :w, :] += np.power(img_17b, 2)

    mean_image *= 1.0 / ll
    std_image *= 1.0 / ll
    std_image -= np.power(mean_image, 2)
    std_image = np.sqrt(std_image)
    return mean_image, std_image

