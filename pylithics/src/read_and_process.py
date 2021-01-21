import skimage
import skimage.feature
import skimage.viewer
from skimage.filters import threshold_minimum, threshold_mean
from skimage import filters
from skimage.measure import find_contours, regionprops, label
import numpy as np
import pandas as pd
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
import cv2

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
    cont_info_list = []

    index = 0
    for cont in contours:

        # check minimum contour size
        if cont.shape[0]/image_total_shape*100 < config_file['minimum_pixels_contour']:
            continue
        # check that the contour is closed.
        elif any((cont[0] == cont[-1])==False):
            continue
        else:
            cont_info = {}
            # Expand numpy dimensions and convert it to UMat object
            area = area_contour(cont)


            cont_info['lenght'] = len(cont)
            cont_info['area'] = area
            cont_info['centroid'] = regionprops(label(cont))[0]['Centroid']
            cont_info['index'] = index

            new_contours.append(cont)
            cont_info_list.append(cont_info)

            index = index + 1

    df_cont_info = pd.DataFrame.from_dict(cont_info_list)

    indexes = contour_desambiguiation(df_cont_info)

    new_contours = [i for j, i in enumerate(new_contours) if j not in indexes]

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


def area_contour(contour):
    """
    Function that calculates the area within a contour using the open-cv library.

    Parameters
    ----------
    contour: array (array with coordinates defining the contour.)

    Returns
    -------
    A number

    """
    # Expand numpy dimensions and convert it to UMat object
    c = cv2.UMat(np.expand_dims(contour.astype(np.float32), 1))
    area = cv2.contourArea(c)

    return area

def contour_desambiguiation(df_cont_info):

    norm = max(df_cont_info['area'])
    index_to_drop = []

    for i in range(df_cont_info.shape[0]):
        area = df_cont_info['area'].iloc[i]
        percentage = area / norm * 100

        if percentage < 5:
            index_to_drop.append(i)

    unique_centroids = np.unique(df_cont_info['centroid'])

    for centroid in unique_centroids:

        cent_df = df_cont_info[df_cont_info['centroid']==centroid][['area']]
        norm = max(cent_df['area'])

        import itertools

        for i, j in itertools.combinations(cent_df.index, 2):
           d_ij = np.linalg.norm(cent_df.loc[i] - cent_df.loc[j])

           print (d_ij/norm)

           if d_ij/norm<0.15:
             if cent_df.loc[i]['area'] < cent_df.loc[j]['area']:
                 index_to_drop.append(i)
             else:
                 index_to_drop.append(j)

    return index_to_drop





