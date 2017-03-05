
#
#
#

import numpy as np
import cv2


total_classes_freq = None


def balance_classes_reset():
    global total_classes_freq
    total_classes_freq = None


def balance_classes(Y, presence_percentage=5, class_freq_tol=2, verbose=0):

    global total_classes_freq
    h, w, ll = Y.shape

    if total_classes_freq is None:
        # total_freq = [class_1, class_2, ...]
        total_classes_freq = np.array([0] * ll)

    if verbose > 2:
        print("DEBUG 2: balance_classes: total_classes_freq=", total_classes_freq)

    classes_freq = np.array([0] * ll)
    for ci in range(ll):
        s = np.sum(Y[:, :, ci], axis=(0, 1))
        if s * 100.0 / (h * w) > presence_percentage:
            classes_freq[ci] += 1

    if verbose > 2:
        print("DEBUG 2: balance_classes: classes_freq=", classes_freq)

    if np.sum(classes_freq) < 1:
        return False

    new_total_classes_freq = total_classes_freq + classes_freq
    m = np.median(new_total_classes_freq)

    for v in new_total_classes_freq:
        if np.abs(v - m) > class_freq_tol:
            if verbose > 2:
                print("DEBUG 2: balance_classes: total_freq=", total_classes_freq)
            return False

    total_classes_freq += classes_freq
    if verbose > 2:
        print("DEBUG 2: balance_classes: total_freq=", total_classes_freq)

    return True


def balance_classes_with_none(Y, presence_percentage=5, class_freq_tol=2, verbose=0):

    global total_classes_freq
    h, w, ll = Y.shape

    if total_classes_freq is None:
        # total_freq = [None, class_1, class_2, ...]
        total_classes_freq = np.array([0] * (ll + 1))

    if verbose > 2:
        print("DEBUG 2: balance_classes: total_classes_freq=", total_classes_freq)

    classes_freq = np.array([0] * (ll+1))
    for ci in range(ll):
        s = np.sum(Y[:, :, ci], axis=(0, 1))
        if s * 100.0 / (h * w) > presence_percentage:
            classes_freq[ci+1] += 1
        else:
            classes_freq[0] += 1

    if verbose > 2:
        print("DEBUG 2: balance_classes: classes_freq=", classes_freq)

    new_total_classes_freq = total_classes_freq + classes_freq
    m = np.median(new_total_classes_freq)

    for v in new_total_classes_freq:
        if np.abs(v - m) > class_freq_tol:
            if verbose > 2:
                print("DEBUG 2: balance_classes: total_freq=", total_classes_freq)

            return False

    total_classes_freq += classes_freq
    if verbose > 2:
        print("DEBUG 2: balance_classes: total_freq=", total_classes_freq)

    return True


def random_rotations(X, Y, angles=(15.0, -15.0, 90.0, -90.0), verbose=0):
    a = angles[np.random.randint(len(angles))] if len(angles) > 0 else 0.0
    if 0 < np.abs(a) < 90:
        sc = 1.2
    else:
        sc = 1.0

    if verbose > 2:
        print("DEBUG 2: random_rotations: a, sc=", a, sc)

    h, w, _ = X.shape
    warp_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), a, sc)
    X = cv2.warpAffine(X, warp_matrix, dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    Y = cv2.warpAffine(Y, warp_matrix, dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if len(X.shape) == 2:
        X = np.expand_dims(X, 2)
    if len(Y.shape) == 2:
        Y = np.expand_dims(Y, 2)
    return X, Y


def random_noise(X, Y, mean=0.0, std=0.25):
    X_noise = std*np.random.randn(*X.shape) + mean
    X += X_noise
    return X, Y
