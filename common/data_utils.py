import os
import pandas as pd

from shapely.wkt import loads
from shapely.affinity import scale

from image_utils import get_image_data

assert os.path.exists('../input'), "Please download Kaggle input data into 'input' folder"

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
    "Buildings",
    # 2
    "Misc. Manmade structures",
    # 3
    "Road",
    # 4
    "Track",
    # 5
    "Trees",
    # 6
    "Crops",
    # 7
    "Waterway",
    # 8
    "Standing water",
    # 9
    "Vehicle Large",
    # 10
    "Vehicle Small",
]

FULL_LABELS = [
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

def get_unit_polygons(image_id):
    polygons = {}
    image_polygons = TRAIN_WKT[TRAIN_WKT['ImageId'] == image_id]
    assert not image_polygons.empty, "This is a test image"
    for class_type in range(1,len(LABELS)):
        polygons[class_type] = loads(image_polygons[image_polygons['ClassType'] == class_type].MultipolygonWKT.values[0])
    return polygons


def get_grid_params(image_id):
    params = GRID_SIZE[GRID_SIZE['ImageId'] == image_id][['Xmax', 'Ymin']]
    assert not params.empty, "No grid parameters for this image id : {}".format(image_id)
    return params.values[0]


def get_scalers(image_id, image_type):
    x_max, y_min = get_grid_params(image_id)
    image_shape = get_image_data(image_id, image_type, return_shape_only=True)   
    h, w = image_shape[:2]  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w * 1.0 / (w + 1.0))
    h_ = h * (h * 1.0 / (h + 1.0))
    return w_ / x_max, h_ / y_min


def get_resized_polygons(image_id, image_type):
    polygons = get_unit_polygons(image_id)
    x_scaler, y_scaler = get_scalers(image_id, image_type)
    resized_polygons = {}
    for class_type in polygons:
        poly = polygons[class_type]
        rpoly = scale(poly, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))  
        resized_polygons[class_type] = rpoly
    return resized_polygons
    

