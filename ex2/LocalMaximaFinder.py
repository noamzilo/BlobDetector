import scipy.ndimage.filters as filters
import numpy as np
import cv2


class LocalMaximaFinder(object):
    def __init__(self, mat):
        self._mat = mat
        self._is_debug = False

    def find_local_maxima(self, nhood):
        if self._is_debug:
            color_mat = np.zeros((self._mat.shape[0], self._mat.shape[1], 3), dtype=self._mat.dtype)
            color_mat[:, :, 0] = self._mat
            color_mat[:, :, 1] = self._mat
            color_mat[:, :, 2] = self._mat
            cv2.imshow("finding max for ", color_mat)

        data_max = filters.maximum_filter(self._mat, nhood * 2)
        maxima_mask = (self._mat == data_max)
        maxima_locations = np.where(maxima_mask)

        # TODO suppress non maxima
        if self._is_debug:
            for maximum_location_X, maximum_location_y in zip(maxima_locations[1], maxima_locations[0]):
                cv2.circle(color_mat, (maximum_location_X, maximum_location_y), nhood, (0, 255, 0), 1)


            cv2.imshow("maximums: ", color_mat)

        return maxima_locations
