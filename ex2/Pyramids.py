from LoG_filter import log_filt
import os
import numpy as np


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
        self._create_pyramids()

    def _create_filters(self):
        self._filters_array = []
        current_scale = self._initial_scale
        for i in range(self._num_pyramids):
            filter_size = 2 * np.ceil(3 * current_scale) + 1
            filt = log_filt(ksize=filter_size, sig=current_scale)
            self._filters_array.append(filt)
            current_scale *= self._scale_multiply_per_level

    def _create_pyramids(self):
        self._read_image_from_disk()

    def _read_image_from_disk(self):
        pass