import scipy.ndimage.filters as filters


class LocalMaximaFinder(object):
    def __init__(self, mat):
        self._mat = mat

    def find_local_maxima(self, nhood):
        data_max = filters.maximum_filter(self._mat, nhood)
        maxima = (self._mat == data_max)
        # data_min = filters.minimum_filter(self._mat, nhood)
        # diff = ((data_max - data_min) > threshold)
        # maxima[diff == 0] = 0