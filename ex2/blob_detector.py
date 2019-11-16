import os
import numpy as np
from ex2.Pyramids import Pyramids


class BlobDetector(object):
    def __init__(self, path_to_image):
        self._absolute_path_to_image = os.path.abspath(path_to_image)
        self._set_constants()
        self._pyramids = Pyramids(self._absolute_path_to_image,
                                  self._num_pyramids,
                                  self._initial_scale_pixels,
                                  self._scale_multiply_per_level)

    def _set_constants(self):
        # These should come from a config file, but I didn't know if external libraries are allowed
        self._num_pyramids = 10
        self._initial_scale_pixels = 2
        self._scale_multiply_per_level = 2 ** 0.25



if __name__ == "__main__":
    def main():
        path_to_image = r"../images/butterfly.jpg"
        blob_detector = BlobDetector(path_to_image)

    main()