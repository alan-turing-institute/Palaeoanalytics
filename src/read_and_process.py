import skimage
import skimage.feature
import skimage.viewer
from skimage.filters import threshold_minimum
from skimage import filters

import matplotlib.pyplot as plt

def read_image(filename):
    """
    Function that read an image into the skimage library

    Parameters
    ==========
    filename: str, path and file name to the directory where the image
    Returns
    =======
    an array
    """
    image = skimage.io.imread(fname=filename, as_gray=True)

    return image

def detect_lithic(image_array, config_file):
    """
    Function that given an input image array applies thresholding
    and edge detection to find the general shape of the lithic object

    Parameters
    ==========
    image_array: array, of the image read by skimage
    config_file: dict, with information of thresholding values
    Returns
    =======
    an array

    """

    thresh = threshold_minimum(image_array)
    thresh = thresh*config_file['threshold']

    binary = image_array < thresh

    binary_edge_sobel = filters.sobel(binary)

    return binary_edge_sobel




