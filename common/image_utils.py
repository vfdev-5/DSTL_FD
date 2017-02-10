import os
import logging

# GDAL
import gdal
import gdalconst
from gdal_pansharpen import gdal_pansharpen

DATA_3_BANDS='../input/three_band/'
DATA_16_BANDS='../input/sixteen_band/'
GENERATED_DATA_16_BANDS = "../input/generated/"


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
    
    
def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
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
    elif image_type == 'ms_pan':
        data_path = GENERATED_DATA_16_BANDS
        suffix = '_M_P'
    elif image_type == 'swir_pan':
        data_path = GENERATED_DATA_16_BANDS
        suffix = '_A_P'
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)
        
    return os.path.join(data_path, "{}{}.tif".format(image_id, suffix))


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
        