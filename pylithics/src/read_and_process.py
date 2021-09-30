from skimage.filters import threshold_mean
import numpy as np
import pandas as pd
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
from pylithics.src.utils import contour_characterisation, classify_surfaces, \
    get_high_level_parent_and_hierarchy
import cv2
from pylithics.src.utils import template_matching, mask_image, contour_selection
import os
import pylithics.src.plotting as plot


def read_image(input_dir, id, im_type='png'):
    """
    Read images from input directory.

    Parameters
    ----------
    input_dir: str
        Path to input directory where images are found
    id: str
        Image identifier code
    im_type: str
        Image file extension type, default is .png

    Returns
    -------
    an array
    """

    filename = os.path.join(input_dir, id + "." + im_type)

    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    return im


def detect_lithic(image_array, config_file):
    """
    Apply binary threshold and edge detection to input image array/s based on configuration file options.
    Resulting image array has pixel values of 0,1

    Parameters
    ----------
    image_array: array
        Array of an unprocessed image (0, 255 pixels)
    config_file: dict
        Information on thresholding values and other configuration options

    Returns
    -------
    An array
    """

    # thresholding
    thresh = threshold_mean(image_array)
    thresh = thresh + thresh * config_file['threshold']
    # binary_array = 'binarized' image, 0, 255 pixels to 0, 1 pixel values
    _, binary_array = cv2.threshold(image_array, thresh, 255, cv2.THRESH_BINARY_INV)

    # edge detection using sobel filter
    x = cv2.Sobel(binary_array, cv2.CV_64F, 1, 0, ksize=1)
    y = cv2.Sobel(binary_array, cv2.CV_64F, 0, 1, ksize=1)
    absX = cv2.convertScaleAbs(x)  # convert back to uint8
    absY = cv2.convertScaleAbs(y)  # convert back to uint8
    sobelXY = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    return sobelXY, thresh


def find_lithic_contours(binary_array, config_file):
    """
    Contour finding of lithic artefact image from binary image array and configuration options.

    Parameters
    ----------
    binary_array: array
        processed image (0, 1 pixels)
    config_file : dict
        Information on conversion values and other configuration options

    Returns
    -------
    image array
    """

    # contour finding using cv2.RETR_TREE gives contour hierarchy (e.g. object 1 is nested n levels deep in object 2)
    contours_cv, hierarchy = cv2.findContours(binary_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # create empty lists to store contour information
    new_contours = []
    contour_info_list = []

    for index, cont in enumerate(list(contours_cv), start=0):
        contour_array = np.asarray([i[0] for i in cont])

        # calculate character listings of the contour.
        cont_info = contour_characterisation(binary_array, contour_array, config_file['conversion_px'])

        cont_info['index'] = index
        cont_info['hierarchy'] = list(hierarchy)[0][index]

        cont_info['contour'] = contour_array

        new_contours.append(cont)
        contour_info_list.append(cont_info)

    if len(new_contours) != 0:

        df_cont_info = pd.DataFrame.from_dict(contour_info_list)

        df_cont_info['parent_index'], df_cont_info['hierarchy_level'] = get_high_level_parent_and_hierarchy(
            df_cont_info['hierarchy'].values)

        indexes = contour_selection(df_cont_info)

        df_contours = df_cont_info.drop(index=indexes)

    else:
        raise RuntimeError("No contours found in this image")

    return df_contours


def process_image(image_array, config_file):
    """
    Applying De-noising and contrast stretching on an input image.

    Parameters
    ----------
    image_array : array
        Array of an unprocessed image (0, 255 pixels)
    config_file : dict
        Information on configuration values for denoising and contrast stretching.

    Returns
    -------
    image array
    """

    # noise removal by minimizing the variation of the image gradient
    img_denoise = denoise_tv_chambolle(image_array, weight=config_file['denoise_weight'], multichannel=False)

    # Contrast stretching to rescale all pixel intensities that fall within the 2nd and 98th percentiles
    p2, p98 = np.percentile(img_denoise, config_file['contrast_stretch'])
    img_rescale = exposure.rescale_intensity(img_denoise, in_range=(p2, p98))

    return img_rescale


def data_output(contour_df, config_file):
    """
    Create a nested dictionary with data from lithic surface and scar metrics
    that is saved into a json file.

    Parameters
    ----------
    contour_df: dataframe
        Dataframe with all contour information and measurements for a single image
    config_file: dictionary
        Configuration file with processing information for the image

    Returns
    -------
    A dictionary
    """

    # record high level information of surfaces detected
    lithic_output = {}

    lithic_output['id'] = config_file['id']
    lithic_output['conversion_px'] = config_file['conversion_px'] # convert pixels to user define metric
    lithic_output["n_surfaces"] = contour_df[contour_df['hierarchy_level'] == 0].shape[0]  # nested hierarchy of ltihic flake scars
    # based on size

    contour_df.sort_values(by=["area_px"], inplace=True, ascending=False)

    # classify surfaces
    surfaces_classification = classify_surfaces(contour_df)

    id = 0
    outer_objects_list = []

    # loop through the contours
    for hierarchy_level, index, area_px, area_mm, width_mm, height_mm, polygon_count in contour_df[
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

            scars_df = contour_df[contour_df['parent_index'] == index]

            outer_objects["scar_count"] = scars_df.shape[0]
            outer_objects["percentage_detected_scars"] = round(
                scars_df['area_px'].sum() / outer_objects['total_area_px'], 2)

            # low levels contours are scars

            scars_objects_list = []
            scar_id = 0
            for index, area_px, area_mm, width_mm, height_mm, angle, polygon_count, complexity in scars_df[
                ['index', 'area_px', 'area_mm',
                 'width_mm', 'height_mm', 'angle', 'polygon_count','complexity']].itertuples(index=False):
                scars_objects = {}

                scars_objects['scar_id'] = scar_id
                scars_objects['total_area_px'] = area_px
                scars_objects['total_area'] = area_mm
                scars_objects['max_breadth'] = width_mm
                scars_objects['max_length'] = height_mm
                scars_objects['percentage_of_surface'] = round(
                    scars_objects['total_area_px'] / outer_objects['total_area_px'], 2)
                scars_objects['scar_angle'] = angle
                scars_objects["polygon_count"] = polygon_count
                scars_objects["complexity"] = complexity

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


def associate_arrows_to_scars(image_array, contour_df, templates_df):

    """
    Use template matching to match the arrows from a template image to a given flake scar
    defined by a contour in a dataframe.

    Parameters
    ----------
    image_array: array
        2D array of the masked_image_array <- this should be the processed image - img_process
    contour_df: dataframe
        Dataframe with all the contour information and measurements for a masked_image_array
    templates_df: dataframe
        Dataframe containing arrays with arrows templates and measured angles

    Returns
    -------
    A dataframe
    """

    templates_angle = []

    # iterate on each contour to select only flake scars
    for hierarchy_level, index, contour, area_px in contour_df[['hierarchy_level',
                                                          'index', 'contour', 'area_px']].itertuples(index=False):

        angle = np.nan

        # high levels contours are surfaces
        if hierarchy_level != 0:

            # mask scar contour
            masked_image = mask_image(image_array, contour, False)

            # apply template matching to associate arrow to scar
            template_index = template_matching(masked_image, templates_df, contour)

            # if we find a matching template, get the angle.
            if template_index != -1:
                angle = templates_df.iloc[template_index]['angle']

        templates_angle.append(angle)

    contour_df['angle'] = templates_angle

    return contour_df



def get_scars_angles(image_array, contour_df, templates = pd.DataFrame()):
    """
    Classify contours that contain arrows or ripples and return the angle measurement of that scar.
    of contours and associate arrow angle information

    Parameters
    ----------
    image_array: array
        2D array of the masked_image_array
    contour_df: dataframe
        Dataframe with all contour information and measurements for a masked_image_array
    templates: dataframe
        Dataframe with template information

    Returns
    -------
    A dataframe
    """

    if templates.shape[0] == 0:
        # if there is no templates in the dataframe assing nan to angles.
        contour_df['angle'] = np.nan

        # TODO: DO SOMETHING WITH RIPPLES

    else:
        # if there is templates in the dataframe associate them to their respective scars.
        contour_df = associate_arrows_to_scars(image_array, contour_df, templates)

    return contour_df


def find_arrows(image_array, binary_array, debug=False):
    """
    Use connected components to find arrows on an image and return them as a list of templates.

    Parameters
    ----------
    image_array: array
        Array of an unprocessed image  (0:255 pixels)
    binary_array: array
        Processed (binarized) image array (0 to 1)
    debug: flag to plot the outputs.

    Returns
    -------
    A list of arrays.
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
        if area > 3000 or area < 50:
            continue

        # extract templates from bounding box
        roi = binary_array[y:y + h, x:x + w]

        # calculate the ratio between black and while pixels
        try:
            ratio = len(roi[(roi > 0.9)]) / len(roi[(roi != 0)])
        except ZeroDivisionError:
            continue

        # filter templates that are unlikely to be an arrow
        if ratio > 0.85 or ratio < 0.2:
            continue

        # plot the template in case we want to debug.
        if debug:
            plot.plot_template_arrow(image_array, roi, ratio)

        templates.append(roi)

    return templates
