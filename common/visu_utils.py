import logging 

import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Polygon, Patch

from shapely.geometry import box
from shapely.affinity import translate
from shapely.validation import explain_validity

from data_utils import ORDERED_LABEL_IDS, LABELS


def scale_percentile(matrix):
    if len(matrix.shape) == 2:
        matrix = matrix.reshape(matrix.shape + (1,))
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    mins = np.percentile(matrix, 0.5, axis=0)
    maxs = np.percentile(matrix, 99.5, axis=0) - mins
    matrix = 255*(matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d] if d > 1 else [w, h])
    matrix = matrix.clip(0, 255).astype(np.uint8)
    return matrix


def display_img_1b(img_1b_data, roi=None):
    if roi is not None:
        # roi is [minx, miny, maxx, maxy]
        x,y,xw,yh = roi
        img_1b_data = img_1b_data[y:yh,x:xw]
    plt.imshow(scale_percentile(img_1b_data), cmap='gray')


def display_img_3b(img_3b_data, roi=None):
    if roi is not None:
        # roi is [minx, miny, maxx, maxy]
        x,y,xw,yh = roi
        img_3b_data = img_3b_data[y:yh,x:xw,:]
    ax_array = []
    for i in [0,1,2]:
        ax = plt.subplot(1,3,i+1)
        ax_array.append(ax)
        plt.imshow(scale_percentile(img_3b_data[:,:,i]), cmap='gray')
        plt.title("Channel %i" % i)


def display_img_8b(img_ms_data, roi=None):
    if roi is not None:
        # roi is [minx, miny, maxx, maxy]
        x,y,xw,yh = roi
        img_ms_data = img_ms_data[y:yh,x:xw,:]
    ax_array = []
    for i in range(8):
        ax = plt.subplot(2,4,i+1)
        ax_array.append(ax)
        plt.imshow(scale_percentile(img_ms_data[:,:,i]), cmap='gray')
        plt.title("Channel %i" % i)
    return ax_array


def display_labels(label_img, alpha=0.5, roi=None, ax_array=None, show_legend=True):
    if roi is not None:
        # roi is [minx, miny, maxx, maxy]
        x,y,xw,yh = roi
        label_img = label_img[y:yh,x:xw]
    
    cmap = plt.get_cmap('Paired', 10)
    if ax_array is None:
        ax_array = [plt.gca()]
        
    for ax in ax_array:
        ax.imshow(label_img, cmap=cmap, alpha=alpha)
        
    if show_legend:
        legend_handles = []
        for i in range(len(ORDERED_LABEL_IDS)):
            class_type = LABELS[ORDERED_LABEL_IDS[i]]
            legend_handles.append(Patch(color=cmap(i), label='{}'.format(class_type)))
            
        index = 0 if len(ax_array) == 1 else len(ax_array)//2 - 1
        ax = ax_array[index]
        ax.legend(handles=legend_handles, 
                  bbox_to_anchor=(1.05, 1), 
                  loc=2, 
                  borderaxespad=0.,
                  fontsize='x-small',
                  title='Objects:',
                  framealpha=0.3)


def display_polygons(polygons, roi=None, ax_array=None, show_legend=True):
    if roi is not None:
        # roi is [minx, miny, maxx, maxy]
        b = box(*roi)
    if ax_array is None:
        ax_array = [plt.gca()]

    cmap = plt.get_cmap('Paired', len(LABELS))
    
    def _draw_polygon(ax, polygon, class_type):
        _add_mpl_polygon(ax, polygon.exterior, cmap(class_type))
        for lin_ring in polygon.interiors:
            _add_mpl_polygon(ax, lin_ring, 'k')
    
    def _add_mpl_polygon(ax, linear_ring, color):
        mpl_poly = Polygon(np.array(linear_ring), color=color, lw=0, alpha=0.5)
        ax.add_patch(mpl_poly)
        
    legend_handles = []
    for class_type in ORDERED_LABEL_IDS:
        if class_type not in polygons:
            continue
        for i, polygon in enumerate(polygons[class_type]):     
            draw_polygon = roi is None
            if roi is not None and polygon.intersects(b):
                if not polygon.is_valid:
                    logging.warn("Polygon (%i, %i) is not valid: %s" % (class_type, i, explain_validity(polygon)))
                    continue
                polygon = polygon.intersection(b)
                polygon = translate(polygon, -roi[0], -roi[1])
                draw_polygon = True
            if draw_polygon:
                if polygon.type == 'MultiPolygon':
                    for p in polygon:
                        for ax in ax_array:
                            _draw_polygon(ax, p, class_type)
                else:
                    for ax in ax_array:
                        _draw_polygon(ax, polygon, class_type)

        legend_handles.append(Patch(color=cmap(class_type), label='{} ({})'.format(LABELS[class_type], len(polygons[class_type]))))

    for ax in ax_array:
        ax.relim()
        ax.autoscale_view()    
    if show_legend:
        index = 0 if len(ax_array) == 1 else len(ax_array)//2 - 1
        ax = ax_array[index]
        ax.legend(handles=legend_handles, 
                  bbox_to_anchor=(1.05, 1), 
                  loc=2, 
                  borderaxespad=0.,
                  fontsize='x-small',
                  title='Objects:',
                  framealpha=0.3)