
#
# Helper method to perform a post-processing on predicted data
#
import numpy as np
import cv2


def sieve(image, size):
    """
    Filter removes small objects of 'size' from binary image
    Input image should be a single band image of type np.uint8

    Idea : use Opencv findContours
    """
    assert image.dtype == np.uint8, "Input should be a Numpy array of type np.uint8"

    sq_limit = size**2
    lin_limit = size*4

    out_image = image.copy()
    image, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(hierarchy) > 0:
        hierarchy = hierarchy[0]
        index = 0
        while index >= 0:
            contour = contours[index]
            p = cv2.arcLength(contour, True)
            s = cv2.contourArea(contour)
            r = cv2.boundingRect(contour)
            if s <= sq_limit and p <= lin_limit:
                out_image[r[1]:r[1]+r[3],r[0]:r[0]+r[2]] = 0
            # choose next contour of the same hierarchy
            index = hierarchy[index][0]

    return out_image


def normalize(img):
    assert len(img.shape) == 2, "Image should have one channel"
    out = img.astype(np.float32)
    mins = out.min()
    maxs = out.max() - mins
    return (out - mins) / (maxs + 0.00001)


def binarize(img, threshold=0.1, size=10, iters=1):
    img = normalize(img)
    res = (img > threshold).astype(np.uint8)    
    res = sieve(res, size)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((3,3), dtype=np.uint8), iterations=iters)
    return res