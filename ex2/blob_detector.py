import os
import numpy as np
from ex2.Pyramids import Pyramids
from ex2.LocalMaximaFinder import LocalMaximaFinder
import scipy.ndimage.filters as filters
import cv2


class LogBlobDetector(object):
    def __init__(self, path_to_image):
        self._absolute_path_to_image = os.path.abspath(path_to_image)
        assert os.path.isfile(self._absolute_path_to_image)
        self._read_image_from_disk()

        self._set_constants()
        self._pyramids = Pyramids(self._grayscale_image,
                                  self._num_pyramids,
                                  self._initial_scale_pixels,
                                  self._scale_multiply_per_level)


    def _read_image_from_disk(self):
        self._grayscale_image = cv2.imread(self._absolute_path_to_image, 0)
        self._h, self._w = self._grayscale_image.shape
        cv2.imshow("grayscale", self._grayscale_image)


    def _set_constants(self):
        # These should come from a config file, but I didn't know if external libraries such as pyYaml are allowed
        self._num_pyramids = 10
        self._initial_scale_pixels = 2
        self._scale_multiply_per_level = 2 ** 0.25
        self._max_min_threshold = 0

    def find_local_maxima(self):
        color_image = np.zeros((self._h, self._w, 3), dtype=self._grayscale_image.dtype)
        color_image[:, :, 0] = self._grayscale_image
        color_image[:, :, 1] = self._grayscale_image
        color_image[:, :, 2] = self._grayscale_image
        cv2.imshow("color from grayscale", color_image)

        pyramids, scales, filt_sizes = self._pyramids.get_pyramids()

        first_filter_size = filt_sizes[0]
        last_filter_size = filt_sizes[-1]
        avg_filter_size = np.median(filt_sizes)
        arbitraty_size = filt_sizes[-3]

        data_max = filters.maximum_filter(pyramids, arbitraty_size)
        maxima_mask = np.logical_and((pyramids == data_max), data_max > self._max_min_threshold)
        maxima_locations = np.where(maxima_mask)

        for maxima_locations_x, maximum_location_y, mask_ind in zip(maxima_locations[1], maxima_locations[0], maxima_locations[2]):
            cv2.circle(color_image, (maxima_locations_x, maximum_location_y), filt_sizes[mask_ind], (0, 255, 0), 1)

        cv2.imshow("local_maxima on original", color_image)

        hi=5






        hi=5




if __name__ == "__main__":
    def main():
        butterfly = r"../images/butterfly.jpg"
        einstein = r"../images/einstein.jpg"
        fishes = r"../images/fishes.jpg"
        sunflowers = r"../images/sunflowers.jpg"
        blob_detector = LogBlobDetector(butterfly)
        # blob_detector = LogBlobDetector(einstein)
        # blob_detector = LogBlobDetector(fishes)
        # blob_detector = LogBlobDetector(sunflowers)
        blob_detector.find_local_maxima()

    main()