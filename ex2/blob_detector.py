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
        cv2.waitKey(0)


    def _set_constants(self):
        # These should come from a config file, but I didn't know if external libraries such as pyYaml are allowed
        self._num_pyramids = 5
        self._initial_scale_pixels = 2
        self._scale_multiply_per_level = 2 ** 0.25

    def find_local_maxima(self):
        color_image = np.zeros((self._h, self._w, 3), dtype=self._grayscale_image.dtype)
        color_image[:, :, 0] = self._grayscale_image
        color_image[:, :, 1] = self._grayscale_image
        color_image[:, :, 2] = self._grayscale_image

        pyramids, scales, filt_sizes = self._pyramids.get_pyramids()
        for pyramid_ind, scale, filt_size in zip(range(self._num_pyramids), scales, filt_sizes):
            local_maxima_finder = LocalMaximaFinder(pyramids[:, :, pyramid_ind])
            maxima_locations = local_maxima_finder.find_local_maxima(filt_size)
            for maximum_location_X, maximum_location_y in zip(maxima_locations[1], maxima_locations[0]):
                cv2.circle(color_image, (maximum_location_X, maximum_location_y), filt_size, (0, 255, 0), -1)

        cv2.imshow("local_maxima on original", color_image)

        hi=5




if __name__ == "__main__":
    def main():
        path_to_image = r"../images/butterfly.jpg"
        blob_detector = LogBlobDetector(path_to_image)
        blob_detector.find_local_maxima()

    main()