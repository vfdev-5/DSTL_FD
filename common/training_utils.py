
import numpy as np
import cv2

from data_utils import TRAIN_WKT, get_image_ids, get_filename
from geo_utils.GeoImage import GeoImage
from geo_utils.GeoImageTilers import GeoImageTilerConstSize


def tile_iterator(image_ids_to_use, classes, presence_percentage=2, tile_size=(256, 256), return_tile_info=False):
    """
    Method returns a random tile in which at least one class of `classes` is present more than `presence_percentage`

    Random tile generation is a uniform tile selection.
    5 random images containing `classes` are selected. Overlapping tiles are searched that contain any of class.

    To uniformize tile generation, total pixel number of each class is counted and a generated tile is selected in
    a way to keep total pixel numbers balanced.


    """
    gb = TRAIN_WKT[~TRAIN_WKT['MultipolygonWKT'].str.contains("EMPTY")].groupby('ClassType')
    overlapping = int(min(tile_size[0], tile_size[1]) * 0.25)

    total_n_pixels = np.array([0] * len(classes))

    while True:

        image_ids = get_image_ids(classes, gb)
        image_ids = list(set(image_ids) & set(image_ids_to_use))
        np.random.shuffle(image_ids)
        step = 5

        for i, _ in enumerate(image_ids[::step]):
            e = min(step * i + step, len(image_ids))
            ids = image_ids[step*i:e]
            # Open 5 labels images
            gimg_tilers = []
            for image_id in ids:
                gimg_label = GeoImage(get_filename(image_id, 'label'))
                gimg_label_tiles = GeoImageTilerConstSize(gimg_label, tile_size=tile_size, min_overlapping=overlapping)
                gimg_tilers.append(gimg_label_tiles)

            counter = 0
            max_counter = gimg_tilers[0].nx * gimg_tilers[0].ny
            while counter < max_counter:
                all_done = True

                for tiler_index, tiles in enumerate(gimg_tilers):
                    for tile_info_label in tiles:
                        all_done = False
                        tile_label, xoffset_label, yoffset_label = tile_info_label
                        h, w, _ = tile_label.shape
                        class_freq = np.array([0 ] *len(classes))
                        for ci, cindex in enumerate(classes):
                            class_freq[ci] += cv2.countNonZero(tile_label[:, :, cindex])
                        # If class representatifs are less than presence_percentage in the tile -> discard the tile
                        if np.sum(class_freq) * 100.0 / (h*w) < presence_percentage:
                            continue

                        if np.sum(total_n_pixels) > 1:
                            old_argmax = np.argmax(total_n_pixels)
                            new_argmax = np.argmax(class_freq)
                            if old_argmax == new_argmax:
                                continue
                        total_n_pixels += class_freq

                        tile_label = tile_label[:, :, classes]

                        gimg_17b = GeoImage(get_filename(ids[tiler_index], '17b'))
                        width = min(tile_size[0], gimg_17b.shape[1] - xoffset_label)
                        height = min(tile_size[1], gimg_17b.shape[0] - yoffset_label)
                        tile_17b = gimg_17b.get_data([xoffset_label, yoffset_label, width, height])
                        assert tile_label.shape[:2] == tile_17b.shape[:2], "Tile sizes are not equal: {} != {}".format \
                            (tile_label.shape[:2], tile_17b.shape[:2])
                        if return_tile_info:
                            yield tile_17b, tile_label, xoffset_label, yoffset_label, ids[tiler_index]
                        else:
                            yield tile_17b, tile_label
                        break

                counter += 1
                # Check if all tilers have done the iterations
                if all_done:
                    break
