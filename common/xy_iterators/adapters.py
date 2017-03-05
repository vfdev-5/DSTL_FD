
import sys
sys.path.append("../")

import numpy as np

from geo_utils.GeoImage import GeoImage, from_ndarray
from submission_utils import get_data_csv, get_scaled_polygons, compute_label_image
from image_utils import get_image_data
from training_utils import normalize_image


def compute_X_csv_mean_std(image_ids, csv_file, out_shape, X_channels, feature_wise=False):

    assert out_shape[2] == len(X_channels), "WTF"
    ll = len(image_ids)
    X_mean = np.zeros(out_shape, dtype=np.float32)
    X_std = np.zeros(out_shape, dtype=np.float32)

    for image_id in image_ids:
        data_csv = get_data_csv(image_id, csv_file)
        polygons = get_scaled_polygons(data_csv)
        X = compute_label_image(image_id, polygons)
        X = X[:, :, X_channels]
        h, w, _ = X.shape
        if feature_wise:
            X_mean[:, :, :] += np.mean(X, axis=(0, 1))
            X_std[:, :, :] += np.std(X, axis=(0, 1))
        else:
            X_mean[:h, :w, :] += X
            X_std[:h, :w, :] += np.power(X, 2.0)

    X_mean *= 1.0 / ll
    X_std *= 1.0 / ll
    if not feature_wise:
        X_std -= np.power(X_mean, 2.0)
        X_std = np.sqrt(X_std)

    return X_mean, X_std


def X_csv_adapter(image_id, csv_file, X_channels=None, X_mean=None, X_std=None):
    data_csv = get_data_csv(image_id, csv_file)
    polygons = get_scaled_polygons(data_csv)
    X = compute_label_image(image_id, polygons).astype(np.float32)
    if X_channels is not None:
        X = X[:, :, X_channels]

    if X_mean is not None and X_std is not None:
        X = normalize_image(X, X_mean, X_std)

    gimg = from_ndarray(X)
    return gimg


def label_11d_adapter(image_id, Y_channels):
    y = get_image_data(image_id, 'label')
    y = y[:, :, Y_channels]

    gimg = from_ndarray(y)
    return gimg
