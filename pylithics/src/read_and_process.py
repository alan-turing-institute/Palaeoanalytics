from skimage.filters import threshold_mean
import numpy as np
import pandas as pd
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
from pylithics.src.utils import contour_characterization, classify_surfaces, \
    get_high_level_parent_and_hierarchy
import cv2
from pylithics.src.utils import template_matching, mask_image, contour_selection
import os
import pylithics.src.plotting as plot

null = None

def read_image(input_dir: str, image_id: str, im_type: str = 'png', grayscale: bool = True) -> np.ndarray:
    """
    Read an image from the input directory and ensure it's in the correct format.

    Parameters
    ----------
    input_dir : str
        Path to the directory where images are stored.
    image_id : str
        Image identifier or name without the extension.
    im_type : str, optional
        Image file extension/type, default is 'png'.
    grayscale : bool, optional
        If True, reads the image in grayscale. If False, reads the image in color. Default is True.

    Returns
    -------
    np.ndarray
        The image as a numpy array, in 8-bit format if grayscale is True.

    Raises
    ------
    FileNotFoundError
        If the image file cannot be found or loaded.
    ValueError
        If input_dir is not a valid directory.
    """

    if not os.path.isdir(input_dir):
        raise ValueError(f"The specified directory '{input_dir}' does not exist or is not a directory.")

    # Construct the full file path
    filename = os.path.join(input_dir, f"{image_id}.{im_type}")

    # Determine the flag for reading the image (grayscale or color)
    read_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

    # Attempt to read the image
    image = cv2.imread(filename, read_flag)

    # Check if the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Image file '{filename}' not found or could not be loaded.")

    # Ensure the image is in 8-bit format if grayscale is True
    if grayscale:
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return image


def detect_lithic(image_array: np.ndarray, config: dict) -> np.ndarray:
    """
    Apply thresholding and Sobel edge detection to an input image array.
    Allows the selection of different thresholding methods via a configuration file.
    Otsu's thresholding is the default method.

    Parameters
    ----------
    image_array : np.ndarray
        Array of an unprocessed image with pixel values ranging from 0 to 255.
    config : dict
        Configuration dictionary containing options for thresholding and additional processing.

    Returns
    -------
    np.ndarray
        The binary thresholded image.
    """

    # Input validation
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if image_array.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")
    if image_array.size == 0:
        raise ValueError("Input image array is empty.")

    # Ensure image is in 8-bit format for thresholding
    if image_array.dtype != np.uint8:
        image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Get thresholding method from config or use Otsu's as default
    threshold_method = config.get("threshold_method", "otsu").lower()

    # Apply the selected thresholding method
    if threshold_method == "otsu":
        _, binary_array = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif threshold_method == "adaptive":
        binary_array = cv2.adaptiveThreshold(
            image_array,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=config.get("adaptive_block_size", 11),
            C=config.get("adaptive_C", 2)
        )
    elif threshold_method == "mean":
        mean_thresh_value = np.mean(image_array)
        _, binary_array = cv2.threshold(image_array, mean_thresh_value, 255, cv2.THRESH_BINARY_INV)
    else:
        raise KeyError(f"Invalid threshold method specified: {threshold_method}. "
                       "Choose from 'otsu', 'adaptive', or 'mean'.")

    # Optional post-processing: morphological operations
    if config.get("apply_morphology", False):
        kernel_size = config.get("morphology_kernel_size", 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        morph_operation = config.get("morph_operation", "close")
        if morph_operation == "erode":
            binary_array = cv2.erode(binary_array, kernel, iterations=1)
        elif morph_operation == "dilate":
            binary_array = cv2.dilate(binary_array, kernel, iterations=1)
        elif morph_operation == "open":
            binary_array = cv2.morphologyEx(binary_array, cv2.MORPH_OPEN, kernel)
        elif morph_operation == "close":
            binary_array = cv2.morphologyEx(binary_array, cv2.MORPH_CLOSE, kernel)

    return binary_array


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
        cont_info = contour_characterization(binary_array, contour_array, config_file['conversion_px'])

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
    lithic_output["n_surfaces"] = contour_df[contour_df['hierarchy_level'] == 0].shape[0]  # nested hierarchy of lithic flake scars
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
                surfaces_classification) > id else hierarchy_level.where(hierarchy_level.notnull(), None)
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
            for index, area_px, area_mm, width_mm, height_mm, angle, polygon_count in scars_df[
                ['index', 'area_px', 'area_mm',
                 'width_mm', 'height_mm', 'angle', 'polygon_count']].itertuples(index=False):
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

        angle = null

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
        # if there is no templates in the dataframe assign None to angles.
        contour_df['arrow_index'] = -1
        contour_df['angle'] = ''

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
