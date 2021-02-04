import skimage
import skimage.feature
import skimage.viewer
from skimage.filters import threshold_minimum, threshold_mean
import scipy.ndimage as ndi
from skimage import filters
from skimage.measure import find_contours
import numpy as np
import pandas as pd
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from pylithics.src.utils import area_contour, contour_characterisation, contour_desambiguiation, mask_image, classify_distributions
from skimage import img_as_ubyte
import cv2

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
    cont float with the threshold value

    """

    do_morfological = classify_distributions(image_array)

    # thresholding
    thresh = threshold_mean(image_array)
    thresh = thresh+thresh*config_file['threshold']
    binary = image_array < thresh

    # edge detection

    if do_morfological:
        init_ls = checkerboard_level_set(image_array.shape, 6)

        binary_image = morphological_chan_vese(binary, 35, init_level_set=init_ls, smoothing=3)
    else:
        binary_image = filters.sobel(binary)

    return binary_image, thresh


def find_lithic_contours(image_array, config_file):
    """
    Function that given an input image array and configuration options
     finds contours on cont the lithic object

    Parameters
    ==========
    image_array: array, of the image read by skimage
    config_file: dict, with information of thresholding values
    Returns
    =======
    an array

    """


    cv_image = img_as_ubyte(image_array)
    contours_cv, hierarchy = cv2.findContours(cv_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)



    #contours = find_contours(image_array, config_file['contour_parameter'],
    #                         fully_connected=config_file['contour_fully_connected'])

    image_total_shape = len(image_array[0]) * len(image_array[1])

    new_contours = []
    cont_info_list = []

    index = 0
    for cont in list(contours_cv):

        cont = np.asarray([i[0] for i in cont])


        # check minimum contour size
        if cont.shape[0] / image_total_shape * 100 < config_file['minimum_pixels_contour']:
            continue
        # check that the contour is closed.
        #elif any((cont[0] == cont[-1]) == False):
        #    continue
        else:

            # calculate characteristings of the contour.
            cont_info = contour_characterisation(cont)

            cont_info['centroid'] = ndi.center_of_mass(mask_image(image_array, cont, True))
            cont_info['index'] = index
            cont_info['hierarchy'] = list(hierarchy)[0][index]

            new_contours.append(cont)
            cont_info_list.append(cont_info)

            index = index + 1

    if len(new_contours) != 0:
        df_cont_info = pd.DataFrame.from_dict(cont_info_list)

        indexes = contour_desambiguiation(df_cont_info)

        new_contours = [i for j, i in enumerate(new_contours) if j not in indexes]

    new_contours = np.array(new_contours, dtype="object")

    return new_contours


def process_image(image_array, config_file):
    """
    Function that applies some processing to the initial image

    Parameters
    ==========
    image_array: array, of the image read by skimage
    config_file: dict, with information of processing values
    Returns
    =======
    an array

    """

    # noise removal
    img_denoise = denoise_tv_chambolle(image_array, weight=config_file['denoise_weight'], multichannel=False)

    # Contrast stretching
    p2, p98 = np.percentile(img_denoise, config_file['contrast_stretch'])
    img_rescale = exposure.rescale_intensity(img_denoise, in_range=(p2, p98))

    return img_rescale



