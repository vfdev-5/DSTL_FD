
#
#
#

import sys
sys.path.append("../")

import numpy as np

from geo_utils.GeoImage import GeoImage
from geo_utils.GeoImageTilers import GeoImageTilerConstSize


def tile_iterator(image_ids, X_adapter, Y_adapter, **params):
    """
    X, Y tiles iterator
    :param image_ids:
    :param X_adapter: function that returns GeoImage of X data corresponding to image_id. Function of signature : foo(image_id) -> GeoImage
    :param Y_adapter: function that returns GeoImage of Y data corresponding to image_id. Function of signature : foo(image_id) -> GeoImage
    :param params:
    :return:
    """
    n_images_same_time = params.get('n_images_same_time') if 'n_images_same_time' in params else 5
    tile_size = params.get('tile_size') if 'tile_size' in params else (256, 256)
    overlapping = int(min(tile_size[0], tile_size[1]) * 0.5)

    balance_classes_func = params.get('balance_classes_func')
    transformation_func = params.get('transformation_func')
    random_transformation_func = params.get('random_transformation_func')
    verbose = params.get('verbose') if 'verbose' in params else 0

    _step = n_images_same_time
    # Loop forever:
    while True:
        np.random.shuffle(image_ids)
        for i, _ in enumerate(image_ids[::_step]):
            e = min(_step * (i + 1), len(image_ids))
            ids = image_ids[_step * i:e]

            if verbose > 2:
                print("DEBUG 3: ids: ", ids)

            X_batch = []
            Y_tiles_batch = []
            for image_id in ids:
                X = X_adapter(image_id)
                assert isinstance(X, GeoImage), "X_adapter should return a GeoImage"
                X_batch.append(X)
                Y = Y_adapter(image_id)
                assert isinstance(Y, GeoImage), "Y_adapter should return a GeoImage"
                Y_tiles_batch.append(GeoImageTilerConstSize(Y, tile_size=tile_size, min_overlapping=overlapping))

            counter = 0
            max_counter = Y_tiles_batch[0].nx * Y_tiles_batch[0].ny
            # Iterate over all tiles of label images
            while counter < max_counter:
                all_done = True

                # Get X,Y batch of tiles from `n_images_same_time` files
                XY_tiles_batch = []
                for Y_tiles_index, Y_tiles in enumerate(Y_tiles_batch):

                    if verbose > 1:
                        print("DEBUG 2: Y_tiles_index=", Y_tiles_index)
                    # for + break = next()
                    for Y_tile_info in Y_tiles:
                        all_done = False
                        Y_tile, xoffset, yoffset = Y_tile_info

                        if verbose > 2:
                            print("DEBUG 2: Y_tile, xoffset, yoffset : ", Y_tile.shape, Y_tile.dtype, xoffset, yoffset)

                        if balance_classes_func is not None:
                            if not balance_classes_func(Y_tile):
                                continue

                        X = X_batch[Y_tiles_index]
                        extent = [xoffset, yoffset, tile_size[0], tile_size[1]]
                        X_tile = X.get_data(extent, *tile_size).astype(np.float32)

                        if verbose > 2:
                            print("DEBUG 2: X_tile: ", X_tile.shape, X_tile.dtype)

                        XY_tiles_batch.append((X_tile, Y_tile))
                        break

                # Apply random transformations if any required on tiles
                if random_transformation_func is not None:
                    if verbose > 2:
                        print("DEBUG 2: XY_tiles_batch: ", len(XY_tiles_batch))
                    new_XY_tiles_batch = []
                    for X_tile, Y_tile in XY_tiles_batch:
                        new_XY_tiles_batch.append(random_transformation_func(X_tile, Y_tile))
                    XY_tiles_batch.extend(new_XY_tiles_batch)

                if verbose > 2:
                    print("DEBUG 2: XY_tiles_batch: ", len(XY_tiles_batch))

                for X_tile, Y_tile in XY_tiles_batch:
                    # Apply transformation to all produced tiles
                    if transformation_func is not None:
                        yield transformation_func(X_tile, Y_tile)
                    else:
                        yield X_tile, Y_tile

                counter += 1
                # Check if all tilers have done the iterations
                if all_done:
                    break
