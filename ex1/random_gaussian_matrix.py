import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


class Ex1(object):
    def __init__(self, mean=10, std=5, size=(100, 100)):
        self.__mean = mean
        self.__std = std
        self.__size = size
        self.__mat = None

        self.__abs_path_to_image = os.path.abspath(r"../images/butterfly.jpg")
        self.__color_image = None
        self.__grayscale_image = None

    def generate_random_gaussian_matrix(self):  # 1a
        self.__mat = np.random.normal(loc=self.__mean, scale=self.__std, size=self.__size)
        grayscale_mat = (self.__mat - np.min(self.__mat)) / np.max(self.__mat)
        cv2.imshow("random gaussian matrix", grayscale_mat)
        cv2.waitKey(0)

    def draw_histogram(self):  # 1b
        hist, bins = np.histogram(self.__mat, bins=256)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()

    def read_my_image(self):  # 1c
        self.__color_image = cv2.imread(self.__abs_path_to_image)
        self.__grayscale_image = cv2.imread(self.__abs_path_to_image, 0)
        cv2.imshow("color", self.__color_image)
        cv2.waitKey(0)
        cv2.imshow("grayscale", self.__grayscale_image)
        cv2.waitKey(0)

    def detect_edges(self, thres1, thres2):
        # if pixel's gradient is higher than the high threshold, it is an edge.
        # if a pixel's gradient is lower than the low threshold, it is not an edge.
        # if a pixel's gradient is between the thresholds, then it is an edge iff it is connected to an edge pixel.
        edge_image = cv2.Canny(self.__grayscale_image, thres1, thres2)
        cv2.imshow(f"edges, {thres1}, {thres2}", edge_image)
        cv2.waitKey(0)


if __name__ == "__main__":
    ex1 = Ex1(mean=10, std=5, size=(100, 100))
    # ex1.generate_random_gaussian_matrix()
    # ex1.draw_histogram()
    ex1.read_my_image()
    ex1.detect_edges(thres1=500, thres2=250)
    ex1.detect_edges(thres1=500, thres2=300)
    ex1.detect_edges(thres1=1000, thres2=250)

