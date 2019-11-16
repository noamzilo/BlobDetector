from LoG_filter import log_filt
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d


class Pyramids(object):
    def __init__(self, absolute_path_to_image, num_pyramids, initial_scale, scale_multiply_per_level):
        self._absolute_path_to_image = absolute_path_to_image
        assert os.path.isfile(self._absolute_path_to_image)
        self._num_pyramids = num_pyramids
        assert self._num_pyramids > 0
        self._initial_scale = initial_scale
        assert self._initial_scale > 0
        self._scale_multiply_per_level = scale_multiply_per_level
        assert self._scale_multiply_per_level > 0

        self._create_filters()
        self._pyramids = None
        self._create_pyramids()

    def get_pyramids(self):
        return self._pyramids.copy()

    def _create_filters(self):
        self._filters_array = []
        current_scale = self._initial_scale
        for i in range(self._num_pyramids):
            filter_size = self._calculate_filter_size(current_scale)
            filt = log_filt(ksize=filter_size, sig=current_scale)
            # filt *= self._normalize_filter(filt, sigma=current_scale)
            self._filters_array.append(filt)
            current_scale *= self._scale_multiply_per_level

        for filt in self._filters_array:
            plt.imshow(filt, interpolation='nearest')

    @staticmethod
    def _show_matrix_as_grayscale(title, mat):
        show_this = (mat - np.min(mat))
        show_this = show_this / np.max(show_this)

        cv2.imshow(title, show_this)
        cv2.waitKey(0)

    @staticmethod
    def _normalize_filter(filt, sigma):
        # Laplacian response decays as scale increases
        # We want the same response for every scale, so need to normalize up by sigma per derivative.
        # Laplacian is a second derivative, so need to normalize twice by sigma, hence sigma ** 2
        return filt * (sigma ** 2)

        # return filt * (sigma)

    @staticmethod
    def _calculate_filter_size(sigma):
        return 2 * np.ceil(3 * sigma) + 1

    def _create_pyramids(self):
        self._read_image_from_disk()
        self._pyramids = np.zeros((self._h, self._w, self._num_pyramids), dtype=float)
        for i, filt in enumerate(self._filters_array):
            self._pyramids[:, :, i] = convolve2d(in1=self._grayscale_image, in2=filt, mode='same')

        for i in range(self._num_pyramids):
            self._show_matrix_as_grayscale(title=f"pyramid {i}", mat=self._pyramids[:, :, i])

    def _read_image_from_disk(self):
        self._grayscale_image = cv2.imread(self._absolute_path_to_image, 0)
        self._h, self._w = self._grayscale_image.shape
        cv2.imshow("grayscale", self._grayscale_image)
        cv2.waitKey(0)
