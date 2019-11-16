from LoG_filter import log_filt
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d


class Pyramids(object):
    def __init__(self, grayscale_image, num_pyramids, initial_scale, scale_multiply_per_level):
        self._grayscale_image = grayscale_image
        self._h, self._w = self._grayscale_image.shape

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
        return self._pyramids, self._scales_array.copy(), [filt.shape[0] for filt in self._filters_array]

    def _create_filters(self):
        self._filters_array = []
        self._scales_array = []
        current_scale = self._initial_scale
        for i in range(self._num_pyramids):
            filter_size = self._calculate_filter_size(current_scale)
            filt = log_filt(ksize=filter_size, sig=current_scale)
            filt = self._normalize_filter(filt, sigma=current_scale)
            self._filters_array.append(filt)
            self._scales_array.append(current_scale)
            current_scale *= self._scale_multiply_per_level

        # for i, filt in enumerate(self._filters_array):
        #     # cv2.imshow(f"filt {i}", (filt - np.min(filt)) / (np.max(filt) - np.min(filt)))
        #     plt.imshow(filt, interpolation='nearest')

    @staticmethod
    def _show_matrix_as_grayscale(title, mat):
        show_this = (mat - np.min(mat) / (np.max(mat) - np.min(mat)))
        cv2.imshow(title, show_this)

    @staticmethod
    def _normalize_filter(filt, sigma):
        # Laplacian response decays as scale increases
        # We want the same response for every scale, so need to normalize up by sigma per derivative.
        # Laplacian is a second derivative, so need to normalize twice by sigma, hence sigma ** 2
        return filt * (sigma ** 2)

    @staticmethod
    def _calculate_filter_size(sigma):
        return 2 * np.ceil(3 * sigma) + 1

    def _create_pyramids(self):
        self._pyramids = np.zeros((self._h, self._w, self._num_pyramids), dtype=float)
        for i, filt in enumerate(self._filters_array):
            self._pyramids[:, :, i] = convolve2d(in1=self._grayscale_image, in2=filt, mode='same')
            # self._pyramids[:, :, i] *= self._scales_array[i] ** 2  # normalize filter result

        # for i in range(self._num_pyramids):
        #     self._show_matrix_as_grayscale(title=f"pyramid {i}", mat=self._pyramids[:, :, i])

