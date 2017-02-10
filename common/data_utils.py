import os
import pandas as pd
import gdal
import gdalconst

DATA_3_BANDS='../input/three_band/'
DATA_16_BANDS='../input/sixteen_band/'

GRID_SIZE = pd.read_csv('../input/grid_sizes.csv',
                        names=['ImageId', 'Xmax', 'Ymin'],
                        skiprows=1)
GRID_SIZE.columns = ['ImageId','Xmax','Ymin']
TRAIN_WKT = pd.read_csv('../input/train_wkt_v4.csv',
                        dtype={'ClassType':int})

TRAIN_IMAGE_IDS = TRAIN_WKT['ImageId'].unique()

LABELS = [
    None,
    # 1
    "Buildings - large building, residential, non-residential, fuel storage facility, fortified building",
    # 2
    "Misc. Manmade structures",
    # 3
    "Road",
    # 4
    "Track - poor/dirt/cart track, footpath/trail",
    # 5
    "Trees - woodland, hedgerows, groups of trees, standalone trees",
    # 6
    "Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops",
    # 7
    "Waterway",
    # 8
    "Standing water",
    # 9
    "Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle",
    # 10
    "Vehicle Small - small vehicle (car, van), motorbike",
]


def get_image_data(image_id, image_type='3b', return_shape_only=False):

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

    fname = os.path.join(data_path, "{}{}.tif".format(image_id, suffix))
    img = gdal.Open(fname, gdalconst.GA_ReadOnly)
    assert img, "Image is not found: id={}, type={}".format(image_id, image_type)
    if return_shape_only:
        return (img.RasterYSize, img.RasterXSize, img.RasterCount)

    img_data = img.ReadAsArray()
    if len(img_data.shape) == 3:
        return img_data.transpose([1, 2, 0])
    return img_data
