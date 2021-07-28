from skimage.filters import threshold_mean
from skimage import filters
import numpy as np
import pandas as pd
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from pylithics.src.utils import contour_characterisation, contour_disambiguation, classify_surfaces, \
    get_high_level_parent_and_hierarchy
from skimage import img_as_ubyte
import cv2
from PIL import Image
from pylithics.src.utils import template_matching, mask_image, subtract_masked_image, contour_selection
import os
import pylithics.src.plotting as plot


def read_image(input_dir, id, im_type='png'):
    """
    Read images from input directory.

    Parameters
    ----------
    input_dir: str
        path to input directory where images are found
    id: str
        unique image identifier code
    im_type: str
        image file extension type, default is .png

    Returns
    -------
    an array
    """

    filename = os.path.join(input_dir, id + "." + im_type)

    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    return im


def detect_lithic(image_array, config_file):
    """
    Apply thresholding on input image array based on configuration file options.
    of images that have been 'binarized' (0,1) <- do we need to mention that they are inverted?
    edge detection of 'binarized' images.

    Parameters
    ----------
    image_array: array
        of the image read by openCV
    config_file: dict
        information on thresholding values

    Returns
    -------
    An array
    """

    # thresholding
    thresh = threshold_mean(image_array)
    thresh = thresh + thresh * config_file['threshold']
    _, binary = cv2.threshold(image_array, thresh, 255, cv2.THRESH_BINARY_INV)

    # edge detection
    x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=1)
    y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=1)
    absX = cv2.convertScaleAbs(x)  # convert back to uint8
    absY = cv2.convertScaleAbs(y)
    sobelXY = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    return sobelXY, thresh


def find_lithic_contours(image_array, config_file):
    """
    Contour finding of lithic artefact images from input image array and configuration options.

    Parameters
    ----------
    image_array : array
        of the image read by openCV
    config_file : dict
        information on thresholding values

    Returns
    -------
    an array
    """

    _, contours_cv, hierarchy = cv2.findContours(image_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    new_contours = []
    cont_info_list = []

    for index, cont in enumerate(list(contours_cv), start=0):
        cont = np.asarray([i[0] for i in cont])

        # calculate character listings of the contour.
        cont_info = contour_characterisation(image_array, cont, config_file['conversion_px'])

        cont_info['index'] = index
        cont_info['hierarchy'] = list(hierarchy)[0][index]

        cont_info['contour'] = cont

        new_contours.append(cont)
        cont_info_list.append(cont_info)

    if len(new_contours) != 0:

        df_cont_info = pd.DataFrame.from_dict(cont_info_list)

        df_cont_info['parent_index'], df_cont_info['hierarchy_level'] = get_high_level_parent_and_hierarchy(
            df_cont_info['hierarchy'].values)

        indexes = contour_selection(df_cont_info)

        df_contours = df_cont_info.drop(index=indexes)

    else:
        raise RuntimeError("No contours found in this image")

    return df_contours


def process_image(image_array, config_file):
    """
    Image denoising and contrast stretching for images.

    Parameters
    ----------
    image_array : array
        of the image read by openCV
    config_file : dict
        information of processing values

    Returns
    -------
    an array
    """

    # noise removal
    img_denoise = denoise_tv_chambolle(image_array, weight=config_file['denoise_weight'], multichannel=False)

    # Contrast stretching
    p2, p98 = np.percentile(img_denoise, config_file['contrast_stretch'])
    img_rescale = exposure.rescale_intensity(img_denoise, in_range=(p2, p98))

    return img_rescale


def data_output(cont, config_file):
    """
    Create an output of a nested dictionary with data from surfaces and scars metrics
    that could be easily saved into a json file.

    Parameters
    ----------
    cont: dataframe
        dataframe with all contour information and measurements for an image
    config_file: dictionary
        configuration file with relevant information for the image

    Returns
    -------
    a dictionary
    """

    # record high level information of surfaces detected
    lithic_output = {}

    lithic_output['id'] = config_file['id']
    lithic_output['conversion_px'] = config_file['conversion_px']
    lithic_output["n_surfaces"] = cont[cont['hierarchy_level'] == 0].shape[0]

    cont.sort_values(by=["area_px"], inplace=True, ascending=False)

    # classify surfaces
    surfaces_classification = classify_surfaces(cont)

    id = 0
    outer_objects_list = []

    # loop through the contours
    for hierarchy_level, index, area_px, area_mm, width_mm, height_mm, polygon_count in cont[
        ['hierarchy_level', 'index', 'area_px',
         'area_mm', 'width_mm', 'height_mm', 'polygon_count']].itertuples(index=False):

        outer_objects = {}

        # high levels contours are surfaces
        if hierarchy_level == 0:
            outer_objects['surface_id'] = id
            outer_objects['classification'] = surfaces_classification[id] if len(
                surfaces_classification) > id else np.nan
            outer_objects['total_area_px'] = area_px
            outer_objects['total_area'] = area_mm
            outer_objects['max_breadth'] = width_mm
            outer_objects['max_length'] = height_mm
            outer_objects["polygon_count"] = polygon_count

            scars_df = cont[cont['parent_index'] == index]

            outer_objects["scar_count"] = scars_df.shape[0]
            outer_objects["percentage_detected_scars"] = round(
                scars_df['area_px'].sum() / outer_objects['total_area_px'], 2)

            # low levels contours are scars

            scars_objects_list = []
            scar_id = 0
            for index, area_px, area_mm, width_mm, height_mm, angle, polygon_count in scars_df[
                ['index', 'area_px', 'area_mm',
                 'width_mm', 'height_mm', 'angle', 'polygon_count']].itertuples(index=False):
                scars_objects = {}

                scars_objects['scar_id'] = scar_id
                scars_objects['total_area_px'] = area_px
                scars_objects['total_area'] = area_mm
                scars_objects['max_breadth'] = width_mm
                scars_objects['max_length'] = height_mm
                scars_objects['percentage_of_lithic'] = round(
                    scars_objects['total_area_px'] / outer_objects['total_area_px'], 2)
                scars_objects['scar_angle'] = angle
                scars_objects["polygon_count"] = polygon_count

                scars_objects_list.append(scars_objects)
                scar_id = scar_id + 1

            outer_objects['scar_contours'] = scars_objects_list

            id = id + 1
            outer_objects_list.append(outer_objects)

        else:
            continue

    lithic_output['lithic_contours'] = outer_objects_list

    # return nested dictionary
    return lithic_output


def associate_arrows_to_scars(image_array, cont, templates):

    """
    Use template matching to match the arrows to a given flake scar.
    contour dataframe with arrow information, i.e. arrow angles.

    Parameters
    ----------
    image_array: array
        2D array of the masked_image_array
    cont: dataframe
        dataframe with all the contour information and measurements for an masked_image_array
    templates: list of arrays
        list of arrays with arrows templates

    Returns
    -------
    A dataframe
    """

    templates_angle = []

    # iterate on each contour to select only scars
    for hierarchy_level, index, contour, area_px in cont[['hierarchy_level',
                                                          'index', 'contour', 'area_px']].itertuples(index=False):

        angle = np.nan

        # high levels contours are surfaces
        if hierarchy_level != 0:

            # TODO: Make a scar selection to not search in empty scars.

            # mask scar contour
            masked_image = mask_image(image_array, contour, False)

            # apply template matching to associate arrow to scar
            template_index = template_matching(masked_image, templates, contour)

            # if we find a matching template, get the angle.
            if template_index != -1:
                angle = templates.iloc[template_index]['angle']

        templates_angle.append(angle)

    cont['angle'] = templates_angle

    return cont


def get_scars_angles(image_array, cont, templates):
    """
    Classify contours that correspond to arrows or ripples and return the angle measurement of that scar.
    of contours and associate arrow angle information

    Parameters
    ----------
    image_array: array
        2D array of the masked_image_array
    cont: dataframe
        dataframe with all contour information and measurements for a masked_image_array
    templates: array
        list of arrays with templates

    Returns
    -------
    A dataframe
    """

    if templates.shape[0] == 0:
        cont['arrow_index'] = -1
        cont['angle'] = np.nan

        # TODO: DO SOMETHING WITH RIPPLES

    else:
        cont = associate_arrows_to_scars(image_array, cont, templates)

    return cont


def read_arrow_data(input_dir):
    """

    Parameters
    ----------
    input_dir

    Returns
    -------

    """
    id_list = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if i.endswith('.pkl')]

    df_list = []
    for i in id_list:
        df_list.append(pd.read_pickle(i))

    return pd.concat(df_list)


def find_arrows(image_array, image_processed, debug=False):
    """
    Use connected components to find arrows on an image, and return an array with the arrow templates.

    Parameters
    ----------
    image_array: array
        Original image array (0 to 255)
    image_processed: array
        Processed (binarized) image array (0 to 1)
    debug: flag to plot the outputs.

    Returns
    -------
    an array
    """

    # load the image, convert to gray, and threshold.
    thresh = cv2.threshold(image_array, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # apply connected component analysis to the threshold image
    output = cv2.connectedComponentsWithStats(
        thresh, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    templates = []
    # loop over the number of unique connected component labels
    for i in range(0, numLabels):

        # extract connected component stats and centroid
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # select arrows based on area
        if area > 1000 or area < 50:
            continue

        # extract templates from bounding box
        roi = image_processed[y:y + h, x:x + w]

        # calculate the ratio between black and while pixels
        ratio = len(roi[(roi > 0.9)]) / len(roi[(roi != 0)])

        # filter templates that are unlikely to be an arrow
        if ratio > 0.85 or ratio < 0.2:
            continue

        # plot the template in case we want to debug.
        if debug:
            plot.plot_template_arrow(image_array, roi, ratio)

        templates.append(roi)

    return templates
