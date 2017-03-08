
#
# Methods to work with submissions
#

import numpy as np
import cv2

from shapely.wkt import loads
from shapely.affinity import scale
from shapely.geometry import mapping, Polygon, MultiPolygon
from shapely.wkt import dumps
from shapely.affinity import translate

import fiona


import sys
sys.path.append("../common/")
from data_utils import get_scalers, LABELS
from image_utils import get_image_data
from data_utils import mask_to_polygons
from image_utils import compute_mean_warp_matrix


def _process(line):
    data = line.split(',', 2)
    data[2] = data[2].replace("\"", "")
    data[2] = data[2].replace("\n", "")
    data[2] = data[2].replace("\r", "")
    return data


def _unprocess(poly_info):
    return ",".join([poly_info[0], str(poly_info[1]), "\"" + poly_info[2] + "\""]) + "\r\n"


def submission_iterator(csv_file):
    f_in = open(csv_file)
    _ = f_in.readline()

    data_csv = []
    next_data_csv = []

    while True:

        # First line
        if len(next_data_csv) == 0:
            line = f_in.readline()
            # print "First line: ", line[:35], len(data_csv), len(next_data_csv)
            if len(line) == 0:
                return
            image_id = line[:8]
            data_csv.append(_process(line))
        else:
            data_csv = []
            data_csv.append(next_data_csv)
            # print "Copy next_data_csv -> data_csv", next_data_csv[:2], data_csv[0][:2]
            image_id = next_data_csv[0]
            next_data_csv = []

        # Loop
        counter = 0
        while counter < 15:
            prev_image_id = image_id
            line = f_in.readline()
            if len(line) == 0:
                if len(data_csv) > 0:
                    yield data_csv
                return
            image_id = line[:8]
            if image_id == prev_image_id:
                data_csv.append(_process(line))
            else:
                next_data_csv = _process(line)
                # print "End of unique ImageId : ", len(data_csv), "Next is", next_data_csv[0], next_data_csv[1]
                yield data_csv
                break


def write_shp_from_csv(filename, data_csv, simplify=False, tol=5, n_pts=15):
    # Write a new Shapefile
    image_id = data_csv[0][0]
    h, w, _ = get_image_data(image_id, '3b', return_shape_only=True)
    xs, ys = get_scalers(image_id, h, w)

    all_scaled_polygons = {}
    for poly_info in data_csv:
        if "MULTIPOLYGON" not in poly_info[2][:20]:
            continue
        polygons = loads(poly_info[2])
        scaled_polygons = scale(polygons, xfact=xs, yfact=ys, origin=(0, 0, 0))
        all_scaled_polygons[int(poly_info[1])] = scaled_polygons
        if simplify:
            for k in scaled_polygons:
                if len(scaled_polygons[k].exterior.coords) > n_pts:
                    scaled_polygons[k] = scaled_polygons[k].simplify(tolerance=5)
    write_shp_from_polygons(filename, image_id, all_scaled_polygons)


def write_shp_from_polygons(filename, image_id, all_scaled_polygons):
    schema = {
        'geometry': 'MultiPolygon',
        'properties': {'image_id': 'str', 'class': 'int'},
    }
    with fiona.open(filename, 'w', 'ESRI Shapefile', schema) as c:

        def _write_poly(polygons, image_id, k):
            # To align with images in QGis
            polygons = scale(polygons, xfact=1.0, yfact=-1.0, origin=(0, 0, 0))
            c.write({
                'geometry': mapping(polygons),
                'properties': {'image_id': image_id, 'class': k},
            })

        for k in all_scaled_polygons:
            polygons = all_scaled_polygons[k]
            if isinstance(polygons, list):
                for p in polygons:
                    _write_poly(p, image_id, k)
            else:
                _write_poly(polygons, image_id, k)
    print "Written succesfully file : ", filename


def get_data_csv(image_id, csv_filename):
    for data_csv in submission_iterator(csv_filename):
        if image_id == data_csv[0][0]:
            return data_csv


def get_image_ids(csv_filename):
    out = []
    for data_csv in submission_iterator(csv_filename):
        out.append(data_csv[0][0])
    return out


def compute_label_image(image_id, scaled_polygons, image_type='3b'):
    image_shape = get_image_data(image_id, image_type, return_shape_only=True)
    out = np.zeros((image_shape[0], image_shape[1], len(LABELS)), np.uint8)
    out[:, :, 0] = 1
    round_coords = lambda x: np.array(x).round().astype(np.int32)
    for class_type in range(1, len(LABELS)):
        if class_type not in scaled_polygons:
            continue
        one_class_mask = np.zeros((image_shape[0], image_shape[1]), np.uint8)
        for polygon in scaled_polygons[class_type]:
            exterior = [round_coords(polygon.exterior.coords)]
            cv2.fillPoly(one_class_mask, exterior, 1)
            if len(polygon.interiors) > 0:
                interiors = [round_coords(poly.coords) for poly in polygon.interiors]
                cv2.fillPoly(one_class_mask, interiors, 0)
        out[:, :, class_type] = one_class_mask
        out[:, :, 0] = np.bitwise_xor(out[:, :, 0], np.bitwise_and(out[:, :, 0], one_class_mask))  # =x ^ (x & y)
    return out


def get_polygons(data_csv):
    out = {}
    for poly_info in data_csv:
        if "MULTIPOLYGON" not in poly_info[2][:20]:
            continue
        polygons = loads(poly_info[2])
        out[poly_info[1]] = polygons
    return out


def get_scaled_polygons(data_csv, image_type='3b'):
    out = {}
    image_id = data_csv[0][0]
    h, w, _ = get_image_data(image_id, image_type, return_shape_only=True)
    xs, ys = get_scalers(image_id, h, w)
    for poly_info in data_csv:
        if "MULTIPOLYGON" not in poly_info[2][:20]:
            continue
        polygons = loads(poly_info[2])
        scaled_polygons = scale(polygons, xfact=xs, yfact=ys, origin=(0, 0, 0))
        out[int(poly_info[1])] = scaled_polygons
    return out


def write_shp_from_mask(filename, image_id, labels_image, epsilon=0, min_area=0.1):
    all_scaled_polygons = {}
    for class_index in range(1, len(LABELS)):
        polygons = mask_to_polygons(labels_image[:, :, class_index], epsilon=epsilon, min_area=min_area)
        all_scaled_polygons[class_index] = polygons
    write_shp_from_polygons(filename, image_id, all_scaled_polygons)


def rewrite_submission(input_csv_filename, output_csv_file,
                       postproc_single_class_mask_functions,
                       postproc_single_class_shape_functions):
    """

    :param input_csv_filename:
    :param output_csv_file:
    :param postproc_single_class_mask_functions: { i: pp_class_1_func } work with mask images
    :param postproc_single_class_shape_functions: { i: pp_class_1_func } work on polygons
    :return:
    """
    empty_polygon = 'MULTIPOLYGON EMPTY'

    data_iterator = submission_iterator(input_csv_filename)

    f_out = open(output_csv_file, 'w')

    f_out.write("ImageId,ClassType,MultipolygonWKT\r\n")
    try:
        index = 0
        round_coords = lambda x: np.array(x).round().astype(np.int32)
        for data_csv in data_iterator:
            print "--", data_csv[0][0], len(data_csv), index
            image_id = data_csv[0][0]

            h, w, _ = get_image_data(image_id, '3b', return_shape_only=True)
            x_scaler, y_scaler = get_scalers(image_id, h, w)
            for i, class_index in enumerate(range(1, len(LABELS))):
                if "EMPTY" in data_csv[i][2][:50]:
                    f_out.write(_unprocess(data_csv[i]))
                    continue

                b1 = class_index in postproc_single_class_shape_functions
                b2 = class_index in postproc_single_class_mask_functions

                if b1 or b2:
                    polygons = loads(data_csv[i][2])
                    polygons = scale(polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

                    if b1:
                        if isinstance(polygons, Polygon):
                            polygons = MultiPolygon([polygons])
                        polygons = postproc_single_class_shape_functions[class_index](polygons, image_id=image_id)

                    if b2:
                        one_class_mask = np.zeros((h, w), np.uint8)
                        for polygon in polygons:
                            exterior = [round_coords(polygon.exterior.coords)]
                            cv2.fillPoly(one_class_mask, exterior, 1)
                            if len(polygon.interiors) > 0:
                                interiors = [round_coords(poly.coords) for poly in polygon.interiors]
                                cv2.fillPoly(one_class_mask, interiors, 0)
                        pp_one_class_mask = postproc_single_class_mask_functions[class_index](one_class_mask, image_id=image_id)
                        polygons = mask_to_polygons(pp_one_class_mask, epsilon=1.0, min_area=0.1)

                    if len(polygons) == 0:
                        line = ",".join([image_id, str(class_index), empty_polygon]) + "\r\n"
                        f_out.write(line)
                    else:
                        unit_polygons = scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))
                        line = ",".join([image_id, str(class_index), "\"" + dumps(unit_polygons) + "\""]) + "\r\n"
                        f_out.write(line)
                else:
                    f_out.write(_unprocess(data_csv[i]))
            index += 1
    except KeyboardInterrupt:
        pass

    f_out.close()


def merge_pair_images(pair_images, csv_filename, class_merge_polygons_funcs, merge_labels=(7,)):
    def _translate(polygons, tx, ty):
        polygons_out = {}
        for k in polygons:
            polygons_out[k] = translate(polygons[k], xoff=tx, yoff=ty)
        return polygons_out

    n, m = (5, 5)
    l = len(pair_images)
    data_csv_1_array = np.empty((l, n, m), dtype=list)
    data_csv_2_array = np.empty((l, n, m), dtype=list)
    for data_csv in submission_iterator(csv_filename):
        for k, pair in enumerate(pair_images):
            for i in range(n):
                for j in range(m):
                    image_id_1 = pair[0] + "_%i_%i" % (i, j)
                    if data_csv[0][0] == image_id_1:
                        data_csv_1_array[k, i, j] = list(data_csv)
                    image_id_2 = pair[1] + "_%i_%i" % (i, j)
                    if data_csv[0][0] == image_id_2:
                        data_csv_2_array[k, i, j] = list(data_csv)

    n, m = (5, 5)
    results = {}
    for k, pair in enumerate(pair_images):
        for i in range(n):
            for j in range(m):

                image_id_1 = pair[0] + "_%i_%i" % (i, j)
                image_id_2 = pair[1] + "_%i_%i" % (i, j)
                print "---", image_id_1, image_id_2

                # Get polygons
                #data_csv_1 = get_data_csv(image_id_1, csv_filename)
                #data_csv_2 = get_data_csv(image_id_2, csv_filename)
                data_csv_1 = data_csv_1_array[k, i, j]
                data_csv_2 = data_csv_2_array[k, i, j]

                if data_csv_1 is None or data_csv_2 is None:
                    continue

                assert data_csv_1[0][0] == image_id_1 and data_csv_2[0][0] == image_id_2, \
                    "Wrong data: {}, {}, {}, {}".format(data_csv_1[0][0], image_id_1, data_csv_2[0][0], image_id_2)

                polygons_1 = get_scaled_polygons(data_csv_1, '3b')
                polygons_2 = get_scaled_polygons(data_csv_2, '3b')

                _proceed = False
                for index in merge_labels:
                    if index in polygons_1 and index in polygons_2:
                        _proceed = True
                        break

                if not _proceed:
                    continue

                print "--- merge ..."
                # Compute translations from image_3b_2 to image_3b_1
                image_3b_1 = get_image_data(image_id_1, '3b')
                image_3b_2 = get_image_data(image_id_2, '3b')
                roi_size = (150, 150)
                warp_matrix_12 = compute_mean_warp_matrix(image_3b_1, image_3b_2,
                                                          roi_size=roi_size, err=0.01,
                                                          use_gradients=False)
                tx_12 = warp_matrix_12[0, 2]
                ty_12 = warp_matrix_12[1, 2]
                # print " tx_12, ty_12 :", tx_12, ty_12

                # Merge polygons of data_csv_2 to data_csv_1
                polygons_2_1 = _translate(polygons_2, -tx_12, -ty_12)
                polygons_1_2 = _translate(polygons_1, tx_12, ty_12)
                polygons_1_merged = {}
                polygons_2_merged = {}
                for index in merge_labels:

                    if index in polygons_1 and index in polygons_2:
                        # Merge polygons of data_csv_2 to data_csv_1
                        if not polygons_1[index].is_valid:
                            polygons_1[index] = polygons_1[index].buffer(0)
                        if not polygons_2[index].is_valid:
                            polygons_2[index] = polygons_2[index].buffer(0)
                        if not polygons_2_1[index].is_valid:
                            polygons_2_1[index] = polygons_2_1[index].buffer(0)
                        if not polygons_1_2[index].is_valid:
                            polygons_1_2[index] = polygons_1_2[index].buffer(0)

                        polygons_1_merged[index] = class_merge_polygons_funcs[index](polygons_1[index], polygons_2_1[index])
                        polygons_2_merged[index] = class_merge_polygons_funcs[index](polygons_2[index], polygons_1_2[index])

                results[image_id_1] = polygons_1_merged
                results[image_id_2] = polygons_2_merged
    return results


def rewrite_submission_with_new_polygons(input_csv_filename, output_csv_file, image_id_new_polygons):
    """
    :param image_id_new_polygons: {'image_id': {'class_label': MultiPolygon, ...}, ... }
    """
    empty_polygon = 'MULTIPOLYGON EMPTY'

    data_iterator = submission_iterator(input_csv_filename)

    f_out = open(output_csv_file, 'w')

    f_out.write("ImageId,ClassType,MultipolygonWKT\r\n")
    try:
        index = 0
        for data_csv in data_iterator:
            print "--", data_csv[0][0], len(data_csv), index
            index += 1
            image_id = data_csv[0][0]

            if image_id not in image_id_new_polygons:
                for i, class_index in enumerate(range(1, len(LABELS))):
                    # Write as it is
                    f_out.write(_unprocess(data_csv[i]))
                    # print "---> ", i, class_index
                continue

            h, w, _ = get_image_data(image_id, '3b', return_shape_only=True)
            x_scaler, y_scaler = get_scalers(image_id, h, w)
            new_polygons = image_id_new_polygons[image_id]
            for i, class_index in enumerate(range(1, len(LABELS))):

                if class_index not in new_polygons:
                    # Write as it is
                    f_out.write(_unprocess(data_csv[i]))
                    # print "===> ", i, class_index
                    continue

                polygons = new_polygons[class_index]
                # print class_index, new_polygons, type(new_polygons[class_index]), len(new_polygons[class_index])

                if polygons.area == 0.0: #len(polygons) == 0:
                    line = ",".join([image_id, str(class_index), empty_polygon]) + "\r\n"
                    f_out.write(line)
                    # print "++++ empty", class_index
                else:
                    unit_polygons = scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))
                    line = ",".join([image_id, str(class_index), "\"" + dumps(unit_polygons) + "\""]) + "\r\n"
                    f_out.write(line)
                    # print "++++ new", class_index
            index += 1
    except KeyboardInterrupt:
        pass

    f_out.close()

