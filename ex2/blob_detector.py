import os
import numpy as np
from ex2.Pyramids import Pyramids
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
        self._color_image = cv2.imread(self._absolute_path_to_image)
        self._h, self._w = self._grayscale_image.shape
        cv2.imshow(self._absolute_path_to_image + " grayscale", self._grayscale_image)


    def _set_constants(self):
        # These should come from a config file, but I didn't know if external libraries such as pyYaml are allowed
        self._num_pyramids = 15
        self._initial_scale_pixels = 2
        self._scale_multiply_per_level = 2 ** 0.25
        self._max_min_threshold = 15

    def _find_true_maxima(self):
        pyramids, self._scales, self._filt_sizes = self._pyramids.get_pyramids()
        suppression_diameter = np.median(self._scales) # chose something that would presumable be adaptive and won't need tuning

        data_max = filters.maximum_filter(pyramids, suppression_diameter)
        maxima_mask = np.logical_and((pyramids == data_max), data_max > self._max_min_threshold)
        data_min = filters.minimum_filter(pyramids, suppression_diameter)
        minima_mask = np.logical_and((pyramids == data_min), data_min < -self._max_min_threshold)

        self._true_max_locations = np.where(maxima_mask)
        self._true_min_locations = np.where(minima_mask)

    def detect_blobs(self):
        self._find_true_maxima()
        true_max_locations = self._true_max_locations
        true_min_locations = self._true_min_locations

        # draw blobs with scales on color image
        for maxima_locations_x, maximum_location_y, mask_ind in zip(true_max_locations[1], true_max_locations[0], true_max_locations[2]):
            cv2.circle(self._color_image, (maxima_locations_x, maximum_location_y), int(np.ceil(self._scales[mask_ind])), (0, 255, 0), 1)

        for minima_locations_x, minimum_location_y, mask_ind in zip(true_min_locations[1], true_min_locations[0], true_min_locations[2]):
            cv2.circle(self._color_image, (minima_locations_x, minimum_location_y), int(np.ceil(self._scales[mask_ind])), (0, 0, 255), 1)

        cv2.imshow(self._absolute_path_to_image + " local_maxima on original", self._color_image)


if __name__ == "__main__":
    def main():
        butterfly = r"../images/butterfly.jpg"
        einstein = r"../images/einstein.jpg"
        fishes = r"../images/fishes.jpg"
        sunflowers = r"../images/sunflowers.jpg"
        matryoshkas = r"../images/matryoshkas.jpg"
        cakes = r"../images/cakes.jpg"

        # running takes a while, please be patient.

        blob_detector = LogBlobDetector(butterfly)
        blob_detector.detect_blobs()
        blob_detector = LogBlobDetector(einstein)
        blob_detector.detect_blobs()
        blob_detector = LogBlobDetector(fishes)
        blob_detector.detect_blobs()
        blob_detector = LogBlobDetector(sunflowers)
        blob_detector.detect_blobs()
        blob_detector = LogBlobDetector(matryoshkas)
        blob_detector.detect_blobs()
        blob_detector = LogBlobDetector(cakes)
        blob_detector.detect_blobs()

    main()
    cv2.waitKey(0)