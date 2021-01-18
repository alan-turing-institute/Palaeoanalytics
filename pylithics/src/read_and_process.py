import skimage
import skimage.feature
import skimage.viewer
from skimage.filters import threshold_minimum, threshold_mean
from skimage import filters
from skimage.measure import find_contours
import numpy as np
from skimage.filters.rank import median
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure

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

    image_total_shape = len(image_array[0])*len(image_array[1])

    new_contours = []
    for cont in contours:

        shape = cont.shape[0]
        a = cont[0]
        b = cont[-1]
        # check minimum contour size
        if cont.shape[0]/image_total_shape*100 < config_file['minimum_pixels_contour']:
            continue
        # check that the contour is closed.
        elif any((cont[0] == cont[-1])==False):
            print ('Open contour')
            print(cont[0], cont[-1])

            continue
        else:
            new_contours.append(cont)

    new_contours = np.array(new_contours, dtype="object")

    return new_contours

def detect_scale(image_array, config_file):
    """
    Function that given an input image array and configuration options
    applies thresholding
    and edge detection to find the scale for the lithic object

    Parameters
    ==========
    image_array: array, of the image read by skimage
    config_file: dict, with information of thresholding values
    Returns
    =======
    an array

    """

    thresh = threshold_mean(image_array)
    thresh = thresh*config_file['threshold']

    binary = image_array < thresh

    binary_edge_sobel = filters.sobel_h(binary)

    return binary_edge_sobel, thresh

def find_scale_contours(image_array, config_file):
    """
    Function that given an input image array and configuration options
     finds contours on the scale object

    Parameters
    ==========
    image_array: array, of the image read by skimage
    config_file: dict, with information of thresholding values
    Returns
    =======
    an array

    """

    contours = find_contours(image_array, config_file['contour_parameter'], fully_connected=config_file['contour_fully_connected'])

    lrg_contour = sorted(contours, key = len)[-1]
    return lrg_contour


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
    a float with the threshold value

    """

    # noise removal
    img_denoise = denoise_tv_chambolle(image_array, weight=config_file['denoise_weight'], multichannel=False)

    # Contrast stretching
    p2, p98 = np.percentile(img_denoise, config_file['contrast_stretch'])
    img_rescale = exposure.rescale_intensity(img_denoise, in_range=(p2, p98))


    # thresholding
    thresh = threshold_mean(img_rescale)
    thresh = thresh+thresh*config_file['threshold']
    binary = img_rescale < thresh

    # edge detection
    binary_edge_sobel = filters.sobel(binary)

    return binary_edge_sobel, thresh

