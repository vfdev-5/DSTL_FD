
#
# Helper method to perform a post-processing on predicted data
#
import cPickle

import numpy as np
import cv2

from shapely.geometry import MultiPolygon

import sys
sys.path.append("../common/")
from data_utils import LABELS


def sieve(image, size=None, compactness=None, use_convex_hull=False):
    """
    Filter removes small objects of 'size' from binary image
    Input image should be a single band image of type np.uint8

    Idea : use Opencv findContours
    """
    assert image.dtype == np.uint8, "Input should be a Numpy array of type np.uint8"
    assert size is not None or compactness is not None, "Either size or compactness should be defined"

    if size is not None:
        sq_limit = size**2
        lin_limit = size*4

    out_image = image.copy()
    image, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if hierarchy is not None and len(hierarchy) > 0:
        hierarchy = hierarchy[0]
        index = 0
        while index >= 0:
            contour = contours[index]

            if use_convex_hull:
                contour = cv2.convexHull(contour)
            p = cv2.arcLength(contour, True)
            s = cv2.contourArea(contour)
            r = cv2.boundingRect(contour)
            if size is not None:
                if s <= sq_limit and p <= lin_limit:
                    out_image[r[1]:r[1]+r[3],r[0]:r[0]+r[2]] = 0
            if compactness is not None:
                if np.sign(compactness) * 4.0 * np.pi * s / p ** 2 > np.abs(compactness):
                    out_image[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = 0
            # choose next contour of the same hierarchy
            index = hierarchy[index][0]

    return out_image


def normalize(img):
    assert len(img.shape) == 2, "Image should have one channel"
    out = img.astype(np.float32)
    mins = out.min()
    maxs = out.max() - mins
    return (out - mins) / (maxs + 0.00001)


def binarize(img, threshold_low=0.0, threshold_high=1.0, size=10, iters=1):
    res = ((img >= threshold_low) & (img <= threshold_high)).astype(np.uint8)
    #res = sieve(res, size)
    #res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=iters)
    #res = cv2.morphologyEx(res, cv2.MORPH_DILATE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return res


def crop_postprocessing(bin_img, **kwargs):
    """
    Mask post-processing for 'Crops'

    - Enlarge pathes and erode boundaries <-> Morpho Erode
    - Smooth forms <-> Smooth countours with median filter
    - No small fields <-> Remove small detections with sieve, linear size < 100 pixels
    - No small holes, do not touch pathes <-> Remove small non-detections with sieve, linear size < 50 pixels

    """

    x = cv2.morphologyEx(bin_img, cv2.MORPH_ERODE, kernel=np.ones((3, 3), dtype=np.uint8), iterations=2)

    x = cv2.medianBlur(x, ksize=7)
    x = cv2.medianBlur(x, ksize=7)
    x = (x > 0.5).astype(np.uint8)

    x = sieve(x, 100)
    h, w = x.shape

    inv_x = (x < 0.5).astype(np.uint8)
    inv_x = inv_x[1:h, 1:w]
    inv_x = sieve(inv_x, 75)
    x[1:h, 1:w] = (inv_x < 0.5).astype(np.uint8)

    x[0:5, :] = bin_img[0:5, :]
    x[:, 0:5] = bin_img[:, 0:5]
    x[-6:, :] = bin_img[-6:, :]
    x[:, -6:] = bin_img[:, -6:]
    return x


def roads_postprocessing(bin_img, **kwargs):
    """
    Mask post-processing for 'Roads' (label 3")

    - Remove non linear formes <-> 4 * pi * surface / perimeter^2 < 0.25
    """
    bin_img = sieve(bin_img, compactness=0.25, use_convex_hull=False)
    return bin_img


# def pca(points, n_components=2):
#     covar, mean = cv2.calcCovarMatrix(points, None, cv2.COVAR_SCALE | cv2.COVAR_ROWS | cv2.COVAR_SCRAMBLED)
#     ret, e_vals, e_vecs = cv2.eigen(covar)
#     # Conversion + normalisation required due to 'scrambled' mode
#     e_vecs = cv2.gemm(e_vecs, points - mean, 1, None, 0)
#     # apply_along_axis() slices 1D rows, but normalize() returns 4x1 vectors
#     e_vecs = np.apply_along_axis(lambda n: cv2.normalize(n, None).flat, 1, e_vecs)
#     return e_vecs[:n_components,:], e_vals[:n_components,:], mean


round_coords = lambda x: np.array(x).round().astype(np.int32)


def compute_features(p):
    cnt = round_coords(p.exterior.coords)
    defects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
    p2 = p.convex_hull
    ha = p2.area
    hl = p2.length
    dims = (p.length, p.area)
    indices = (4.0 * np.pi * p.area / p.length ** 2, p.area/p.length)
    indices2 = (ha/p.area,
                p.length/hl,
                p.area/hl,
                ha/p.length,
                p.area/hl**2,
                ha/p.length**2,
                ha - p.area,
                p.length - hl)
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments)
    if defects is not None:
        defs = (np.min(defects[:, 0, -1]), np.max(defects[:, 0, -1]), np.mean(defects[:, 0, -1]))
    else:
        defs = (0, 0, 0)
    return dims + indices + indices2 + defs + tuple(hu_moments.ravel().tolist())


with open('weights/roads_form_classifier', 'rb') as f:
    roads_classifier = cPickle.load(f)


def roads_shape_postprocessing(polygons, **kwargs):
    """
    Mask post-processing for 'Roads' (label 3")

    - Remove non linear formes <->
        a) Compactness index : 4 * pi * surface / perimeter^2 < 0.25
        b) If surface > 1000
    """
    road_properties = []
    for i, p in enumerate(polygons):
        road_properties.append(compute_features(p))
    road_properties = np.array(road_properties)
    road_target_labels = roads_classifier.predict(road_properties)

    new_polygons = []
    for i, p in enumerate(polygons):
        if road_target_labels[i]:
            new_polygons.append(p)
    return MultiPolygon(new_polygons)


def path_postprocessing(bin_img, **kwargs):
    """
    Mask post-processing for 'Path' (label 4)

    - Enlarge pathes <-> Morpho dilate + close
    - Smooth forms <-> Smooth countours with median filter
    """
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel=np.ones((3, 3), dtype=np.uint8), iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), dtype=np.uint8), iterations=2)

    bin_img = cv2.medianBlur(bin_img, ksize=3)
    bin_img = (bin_img > 0.55).astype(np.uint8)

    return bin_img


def trees_postprocessing(bin_img, **kwargs):
    """
    Mask post-processing for 'Trees' (label 5)

    - Enlarge trees <-> Morpho dilate
    """
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel=np.ones((3, 3), dtype=np.uint8), iterations=1)
    return bin_img


def buildings_postprocessing(bin_img, **kwargs):
    """
    Mask post-processing for 'Buildings' (label 1)

    - Enlarge trees <-> Morpho close
    """

    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), dtype=np.uint8), iterations=1)
    return bin_img


def buildings_shape_postprocessing(polygons, **kwargs):
    """
    Mask post-processing for 'Buildings' (label 1)

    - Remove detections all of 6080
    - Keep only detections of 6050_4_4, 6050_4_3
    """
    image_id = kwargs['image_id']
    if image_id is not None:
        if "6080" in image_id:
            return MultiPolygon([])
        elif "6050" in image_id and (image_id != "6050_4_4" and image_id !="6050_4_3"):
            return MultiPolygon([])
        else:
            return polygons
    return polygons


def standing_water_postprocessing(bin_img, **kwargs):
    """
    Mask post-processing for 'Standing water' (label 8)

    - Remove small detections <-> remove small objects with sieve, size < 30
    - Enlarge trees <-> Morpho dilate
    - Smooth contours
    """
    bin_img = sieve(bin_img, size=20)
    bin_img = cv2.medianBlur(bin_img, ksize=3)
    bin_img = (bin_img > 0.55).astype(np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel=np.ones((5, 5), dtype=np.uint8), iterations=1)
    return bin_img


def mask_postprocessing(labels_image, class_pp_func_list):
    out = np.zeros_like(labels_image)
    for i, l in enumerate(LABELS):
        if i in class_pp_func_list:
            out[:,:,i] = class_pp_func_list[i](labels_image[:,:,i])
        else:
            out[:,:,i] = labels_image[:,:,i]
    return out


def shape_postprocessing(all_classes_polygons, class_shape_pp_func_list):
    output = {}
    for k in all_classes_polygons:
        if k in class_shape_pp_func_list:
            output[k] = class_shape_pp_func_list[k](all_classes_polygons[k])
        else:
            output[k] = all_classes_polygons[k]
    return output