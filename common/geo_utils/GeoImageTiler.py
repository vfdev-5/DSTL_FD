# -*- coding:utf-8 -*-

# Python
import logging

# Numpy
import numpy as np

# Project
from GeoImage import GeoImage

logger = logging.getLogger(__name__)


class GeoImageTiler:
    """

        Helper class to iterate over GeoImage
        Tiles of size (tile_size[0], tile_size[1]) are extracted with a four-sided overlapping (overlapping)

        Usage :

            gimage = GeoImage('path/to/file')
            tiles = GeoImageTiler(gimage)

            for tile, xoffset, yoffset in tiles:
                assert instanceof(tile, np.ndarray), "..."


        <--- tile_size[0] -->
      overlapping     overlapping
        <--->           <--->
         ____________________
        |   |           |   |
        |---*************---|
        |   *   *   *   *   |
        |   *   *   *   *   |
        |   *   *   *   *   |
        |---*************---|
        |___|___________|___|


        If the option include_nodata=False, then at the boundaries the outside image overlapping is not added.
        For example, the 1st top-left tile looks like
         tile_size[0] - overlapping
            <--------------->
                       overlapping
                        <--->
            *************---|
            *   *   *   *   |
            *   *   *   *   |
            *   *   *   *   |
            *************---|
            |___________|___|


        x_tile_offset = { i == 0 :  0,
                        { i > 0  : i*(tile_size[0] - overlapping) - overlapping
        y_tile_offset = { j == 0 :  0,
                        { j > 0  : j*(tile_size[1] - overlapping) - overlapping

    """

    def __init__(self, geo_image, tile_size=(1024, 1024), overlapping=256, include_nodata=False, nodata_value=np.nan):

        assert isinstance(geo_image, GeoImage), logger.error("geo_image argument should be instance of GeoImage")
        assert len(tile_size) == 2, logger.error("tile_size argument should be a tuple (sx, sy)")
        assert 2*overlapping < min(tile_size[0], tile_size[1]), \
            logger.error("overlapping argument should be less than half of the min of tile_size")

        self._geo_image = geo_image
        self.tile_size = tile_size
        self.overlapping = overlapping

        self.nx = GeoImageTiler._compute_number_of_tiles(self.tile_size[0], self._geo_image.shape[1], overlapping)
        self.ny = GeoImageTiler._compute_number_of_tiles(self.tile_size[1], self._geo_image.shape[0], overlapping)

        self._index = 0
        self._maxIndex = self.nx * self.ny

        self.include_nodata = include_nodata
        self.nodata_value = nodata_value

    @staticmethod
    def _compute_number_of_tiles(tile_size, image_size, overlapping):
        """
            Method to compute number of overlapping tiles for a given image size
            n = ceil((imageSize + overlapping)/(tileSize - overlapping ))
            imageSize :  [01234567891] (isFourSided=true), tileSize=6, overlapping=2
            tile   0  :[xx0123]
                   1  :    [234567]
                   2  :        [67891x]
                   3  :            [1xxxxx]
              n = ceil ( (11+2) / (6 - 2) ) = 4

            imageSize :  [012345678901234] (isFourSided=true), tileSize=7, overlapping=2
            tile   0  :[xx01234]
                   1  :     [3456789]
                   2  :          [8901234]
                   3  :               [34xxxxx]
              n = ceil ( (16+2) / (7 - 2) ) = 4
        """
        return int(np.ceil((image_size + overlapping)*1.0/(tile_size - overlapping)))

    def __iter__(self):
        return self

    def get_lin_index(self):
        return self._index - 1

    def _get_current_tile_extent(self):

        image_width = self._geo_image.shape[1]
        image_height = self._geo_image.shape[0]
        x_tile_index = self._index % self.nx
        y_tile_index = int(np.floor(self._index * 1.0 / self.nx))

        x_tile_size = self.tile_size[0]
        y_tile_size = self.tile_size[1]
        x_tile_offset = x_tile_index * (self.tile_size[0] - self.overlapping) - self.overlapping
        y_tile_offset = y_tile_index * (self.tile_size[1] - self.overlapping) - self.overlapping

        if not self.include_nodata:
            if x_tile_index == 0:
                x_tile_offset = 0
                x_tile_size -= self.overlapping
            if y_tile_index == 0:
                y_tile_offset = 0
                y_tile_size -= self.overlapping
            x_tile_size = image_width - x_tile_offset if x_tile_offset + x_tile_size >= image_width else x_tile_size
            y_tile_size = image_height - y_tile_offset if y_tile_offset + y_tile_size >= image_height else y_tile_size

        return [x_tile_offset, y_tile_offset, x_tile_size, y_tile_size], x_tile_index, y_tile_index

    def compute_geo_extent(self, tile, x, y):
        """ Compute tile geo extent """
        points = np.array([[x, y],
                           [x + tile.shape[1] - 1, y],
                           [x + tile.shape[1] - 1, y + tile.shape[0] - 1],
                           [x, y + tile.shape[0] - 1]])
        return self._geo_image.transform(points)

    def next(self):
        """
            Method to get current tile

            isFourSided=true :
            tileSize=6, overlapping=1

            offset:   0   4    9    14   19
            Image :  [----------------------]
            Tiles : [x....O]                ]
                     [   [O....O]           ]
                     [        [O....O]      ]
                     [             [O....O] ]
                     [                  [O..xxx]

            offset(i) = { i == 0 :  0,
                        { i > 0  : i*(tileSize - overlapping) - overlapping

            size(i) = { i == 0 : tileSize - overlapping,
                      { i > 0 :  offset(i) + tileSize < imageWidth ? tileSize : imageWidth - offset(i)

            bufferOffset(i) = { i == 0 : dataPtr + overlapping
                              { i > 0 : dataPtr + 0
        """
        if self._index < 0 or self._index >= self._maxIndex:
            raise StopIteration()

        # Define ROI to extract
        extent, x_tile_index, y_tile_index = self._get_current_tile_extent()
        # print "{} = ({},{}) | extent={}".format(self._index, x_tile_index, y_tile_index, extent)

        # Extract data
        data = self._geo_image.get_data(extent, nodata_value=self.nodata_value)

        # ++
        self._index += 1

        return data, extent[0], extent[1]

    __next__ = next


