
import numpy as np
import cv2

from data_utils import TRAIN_WKT, get_image_ids, get_filename
from geo_utils.GeoImage import GeoImage
from geo_utils.GeoImageTilers import GeoImageTilerConstSize


def tile_iterator(image_ids_to_use, classes,
                  image_type='input',
                  label_type='label',
                  balance_classes=True,
                  presence_percentage=2,
                  tile_size=(256, 256),
                  mean_image=None,
                  std_image=None,
                  random_rotation_angles=(0.0, 5.0, 0.0, -5.0, 0.0, 15.0, 0.0, -15.0),
                  random_scales=(),
                  resolution_levels=(1, 2, 4)
                ):
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

    apply_random_transformation = len(random_rotation_angles) > 0 or len(random_scales) > 0

    if len(resolution_levels) == 0:
        resolution_levels = (1)

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
            gimg_labels = []
            for image_id in ids:
                gimg_labels.append(GeoImage(get_filename(image_id, label_type)))

            for res_level in resolution_levels:
                for i in range(len(ids)):
                    gimg_label_tiles = GeoImageTilerConstSize(gimg_labels[i],
                                                              tile_size=tile_size,
                                                              scale=res_level,
                                                              min_overlapping=overlapping)
                    gimg_tilers.append(gimg_label_tiles)

            # gimg_tilers has 5*len(resolution_levels) instances
            # gimg_tilers ~ [img1_res1, img2_res1, ..., img5_res1, img1_res2, ...]

            counter = 0
            max_counter = gimg_tilers[0].nx * gimg_tilers[0].ny
            while counter < max_counter:
                all_done = True

                gimg_inputs = []
                for i in ids:
                    gimg_inputs.append(GeoImage(get_filename(i, image_type)))

                for tiler_index, tiles in enumerate(gimg_tilers):
                    for tile_info_label in tiles:
                        all_done = False
                        tile_label, xoffset_label, yoffset_label = tile_info_label

                        h, w, _ = tile_label.shape
                        if balance_classes:
                            class_freq = np.array([0] *len(classes))
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

                        if label_type == 'label':
                            tile_label = tile_label[:, :, classes]

                        gimg_input = gimg_inputs[tiler_index % 5]
                        scale = resolution_levels[int(np.floor(tiler_index / 5))]

                        #print "scale, xoffset_label, yoffset_label: ", scale, xoffset_label, yoffset_label
                        #print "tile_size[0], gimg_input.shape[1], gimg_input.shape[1] - xoffset_label", tile_size[0], gimg_input.shape[1], gimg_input.shape[1] - xoffset_label

                        tile_size_s = (tile_size[0]*scale, tile_size[1]*scale)
                        extent = [xoffset_label, yoffset_label, tile_size_s[0], tile_size_s[1]]
                        tile_input = gimg_input.get_data(extent, *tile_size).astype(np.float)

                        if mean_image is not None and std_image is not None:
                            mean_tile_image = mean_image[yoffset_label:yoffset_label + tile_size_s[1],
                                              xoffset_label:xoffset_label + tile_size_s[0], :]
                            std_tile_image = std_image[yoffset_label:yoffset_label + tile_size_s[1],
                                             xoffset_label:xoffset_label + tile_size_s[0], :]
                            if scale > 1:
                                mean_tile_image = cv2.resize(mean_tile_image, dsize=tile_size, interpolation=cv2.INTER_LINEAR)
                                std_tile_image = cv2.resize(std_tile_image, dsize=tile_size, interpolation=cv2.INTER_LINEAR)
                            tile_input -= mean_tile_image
                            tile_input /= (std_tile_image + 0.00001)

                        # Add random rotation and scale
                        if apply_random_transformation:
                            sc = random_scales[np.random.randint(len(random_scales))] if len(random_scales) > 0 else 1.0
                            a = random_rotation_angles[np.random.randint(len(random_rotation_angles))] if len(random_rotation_angles) > 0 else 0.0
                            if a != 0 and sc < 1.15:
                                sc = 1.15
                            warp_matrix = cv2.getRotationMatrix2D((tile_size[0] // 2, tile_size[1] // 2), a, sc)
                            h, w, _ = tile_input.shape
                            tile_input = cv2.warpAffine(tile_input,
                                                      warp_matrix,
                                                      dsize=(w, h),
                                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                            
                            tile_label = cv2.warpAffine(tile_label,
                                                      warp_matrix,
                                                      dsize=(w, h),
                                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                            if len(tile_label.shape) == 2:
                                tile_label = np.expand_dims(tile_label, 2)
  
                        assert tile_label.shape[:2] == tile_input.shape[:2], "Tile sizes are not equal: {} != {}".format \
                            (tile_label.shape[:2], tile_input.shape[:2])
                        yield tile_input, tile_label
                        break

                counter += 1
                # Check if all tilers have done the iterations
                if all_done:
                    break
