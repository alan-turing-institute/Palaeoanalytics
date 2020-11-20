import skimage
import skimage.feature
import skimage.viewer
from skimage.filters import threshold_minimum
from skimage import filters
from skimage.measure import find_contours
import numpy as np
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
    Function that given an input image array and configuration options
    applies thresholding
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

def find_lithic_contours(image_array, config_file):
    """
    Function that given an input image array and configuration options
     finds contours on a the lithic object

    Parameters
    ==========
    image_array: array, of the image read by skimage
    config_file: dict, with information of thresholding values
    Returns
    =======
    an array

    """

    contours = find_contours(image_array, config_file['contour_parameter'], fully_connected=config_file['contour_fully_connected'])

    new_contours = []
    for cont in contours:
        if cont.shape[0] < config_file['minimum_pixels_contour']:
            continue
        else:
            new_contours.append(cont)

    new_contours = np.array(new_contours)

    return new_contours



