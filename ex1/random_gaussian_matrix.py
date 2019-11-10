import numpy as np
import cv2
import matplotlib.pyplot as plt


class Ex1(object):
    def __init__(self, mean=10, std=5, size=(100, 100)):
        self.__mean = mean
        self.__std = std
        self.__size = size
        self.__mat = None

    def generate_random_gaussian_matrix(self):  # 1a
        self.__mat = np.random.normal(loc=self.__mean, scale=self.__std, size=self.__size)
        grayscale_mat = (self.__mat - np.min(self.__mat)) / np.max(self.__mat)
        cv2.imshow(__name__, grayscale_mat)
        # cv2.waitKey(0)

    def draw_histogram(self):  # 1b
        hist, bins = np.histogram(self.__mat, bins=256)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()

    def read_my_image(self):


if __name__ == "__main__":
    ex1 = Ex1(mean=10, std=5, size=(100, 100))
    ex1.generate_random_gaussian_matrix()
    ex1.draw_histogram()

