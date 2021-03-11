from skimage.filters import threshold_mean
from skimage import filters
import numpy as np
import pandas as pd
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from pylithics.src.utils import contour_characterisation, contour_desambiguiation, mask_image, classify_distributions, get_high_level_parent_and_hirarchy
from skimage import img_as_ubyte
import cv2
from PIL import Image

def read_image(filename):
    """
    Function that read an image into the Pillow library and returing an array and the pdi information of the image.

    Parameters
    ==========
    filename: str, path and file name to the directory where the image
    Returns
    =======
    an array
    a tuple
    """
    #image = skimage.io.imread(fname=filename, as_gray=True)
    im = Image.open(filename)
    image = np.asarray(im)
    try:
        dpi = im.info['dpi']
    except:
        dpi = 0

    return image, dpi



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

    do_morfological = False#classify_distributions(image_array)

    # thresholding
    thresh = threshold_mean(image_array)
    thresh = thresh+thresh*config_file['threshold']
    binary = image_array < thresh

    # edge detection

    if do_morfological:
        init_ls = checkerboard_level_set(image_array.shape, 6)

        binary_image = morphological_chan_vese(binary, 35, init_level_set=init_ls, smoothing=3)

        if binary_image.sum() > binary_image.shape[0]*binary_image.shape[1]*0.5:
            binary_image = (binary_image-1)*-1
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
    an dataframe with contours and its characteristics.

    """


    cv_image = img_as_ubyte(image_array)
    contours_cv, hierarchy = cv2.findContours(cv_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


    new_contours = []
    cont_info_list = []

    for index, cont in enumerate(list(contours_cv), start=0):

        cont = np.asarray([i[0] for i in cont])

            # calculate characteristings of the contour.
        cont_info = contour_characterisation(image_array, cont,config_file['conversion_px'])

        cont_info['index'] = index
        cont_info['hierarchy'] = list(hierarchy)[0][index]

        new_contours.append(cont)
        cont_info_list.append(cont_info)


    if len(new_contours) != 0:

        df_cont_info = pd.DataFrame.from_dict(cont_info_list)

        df_cont_info['parent_index'], df_cont_info['hierarchy_level'] = get_high_level_parent_and_hirarchy(df_cont_info['hierarchy'].values)

        indexes = contour_desambiguiation(df_cont_info)

        new_contours = [i for j, i in enumerate(new_contours) if j not in indexes]

        df_contours = df_cont_info.drop(index=indexes)

        df_contours['contour'] = np.array(new_contours, dtype="object")

    else:
        raise RuntimeError("No contours found in this image")

    return df_contours


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



