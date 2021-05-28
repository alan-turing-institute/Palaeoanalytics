from skimage.filters import threshold_mean
from skimage import filters
import numpy as np
import pandas as pd
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from pylithics.src.utils import contour_characterisation, contour_desambiguiation, classify_surfaces, get_high_level_parent_and_hirarchy
from skimage import img_as_ubyte
import cv2
from PIL import Image
from pylithics.src.utils import template_matching, mask_image, subtract_masked_image, contour_arrow_selection
import os

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
    # image = skimage.io.imread(fname=filename, as_gray=True)
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

    do_morfological = False  # classify_distributions(image_array)

    # thresholding
    thresh = threshold_mean(image_array)
    thresh = thresh + thresh * config_file['threshold']
    binary = image_array < thresh

    # edge detection

    if do_morfological:
        init_ls = checkerboard_level_set(image_array.shape, 6)

        binary_image = morphological_chan_vese(binary, 35, init_level_set=init_ls, smoothing=3)

        if binary_image.sum() > binary_image.shape[0] * binary_image.shape[1] * 0.5:
            binary_image = (binary_image - 1) * -1
    else:
        binary_image = filters.sobel(binary)

    return binary_image, thresh


def find_lithic_contours(image_array, config_file, arrows = False):
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
    _, contours_cv, hierarchy = cv2.findContours(cv_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)



    new_contours = []
    cont_info_list = []

    for index, cont in enumerate(list(contours_cv), start=0):
        cont = np.asarray([i[0] for i in cont])

        # calculate characteristings of the contour.
        cont_info = contour_characterisation(image_array, cont, config_file['conversion_px'])

        cont_info['index'] = index
        cont_info['hierarchy'] = list(hierarchy)[0][index]

        cont_info['contour'] = cont


        new_contours.append(cont)
        cont_info_list.append(cont_info)

    if len(new_contours) != 0:

        df_cont_info = pd.DataFrame.from_dict(cont_info_list)

        df_cont_info['parent_index'], df_cont_info['hierarchy_level'] = get_high_level_parent_and_hirarchy(
            df_cont_info['hierarchy'].values)


        if arrows==False:
            indexes = contour_desambiguiation(df_cont_info,image_array)
        else:
            indexes = contour_arrow_selection(df_cont_info)


        df_contours = df_cont_info.drop(index=indexes)




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


def data_output(cont, config_file):
    """

    Create a nested dictionary with output data from surfaces and scars
    that could be easily saved into a json file

    Parameters
    ----------
    cont: dataframe
        dataframe with all the contour information and measurements for an image
    config_file: dictionary
        config file with relevant information for the image

    Returns
    -------

        A dictionary
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
    for hierarchy_level, index, area_px, area_mm, width_mm, height_mm in cont[
        ['hierarchy_level', 'index', 'area_px', 'area_mm', 'width_mm', 'height_mm']].itertuples(index=False):

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

            scars_df = cont[cont['parent_index'] == index]

            outer_objects["scar_count"] = scars_df.shape[0]
            outer_objects["percentage_detected_scars"] = round(
                scars_df['area_px'].sum() / outer_objects['total_area_px'], 2)

            # low levels contours are scars

            scars_objects_list = []
            scar_id = 0
            for index, area_px, area_mm, width_mm, height_mm, arrow_angle in scars_df[
                ['index', 'area_px', 'area_mm', 'width_mm', 'height_mm','arrow_angle']].itertuples(index=False):
                scars_objects = {}

                scars_objects['scar_id'] = scar_id
                scars_objects['total_area_px'] = area_px
                scars_objects['total_area'] = area_mm
                scars_objects['max_breadth'] = width_mm
                scars_objects['max_length'] = height_mm
                scars_objects['percentage_of_lithic'] = round(
                    scars_objects['total_area_px'] / outer_objects['total_area_px'], 2)
                scars_objects['arrow_angle'] = arrow_angle

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

def get_arrows(image_array, cont, templates):
    """

    Function that classifies contours that correspond to arrows,
    turns them to templates and then uses template matching to
    match the arrows to a given scar.


    Parameters
    ----------
    image_array: array,
        2D array of the masked_image_array
    contours: dataframe
        dataframe with all the contour information and measurements for an masked_image_array

    Returns
    -------

    A contour dataframe with arrow information

    """


    if templates.shape[0]==0:
        cont['arrow_index'] = -1

    else:


        templates_id = []
        templates_angle = []

        for hierarchy_level, index, contour, area_px in cont[
            ['hierarchy_level', 'index', 'contour', 'area_px']].itertuples(index=False):

            id = np.nan
            angle = np.nan

            # high levels contours are surfaces
            if hierarchy_level != 0:

                #TODO: Make a scar selection to not search in empty scars.

                masked_image = mask_image(image_array, contour, False)



                rows, columns = subtract_masked_image(masked_image)

                new_masked_image = np.delete(masked_image, rows[:-1], 0)
                new_masked_image = np.delete(new_masked_image, columns[:-1], 1)

                template_index = template_matching(new_masked_image,templates)

                if template_index!=-1:
                    id = templates.iloc[template_index]['id']
                    angle = templates.iloc[template_index]['angle']

            templates_id.append(id)
            templates_angle.append(angle)

        cont['arrow_template_id'] = templates_id
        cont['arrow_angle'] = templates_angle

    return cont


def read_arrow_data(input_dir):

    id_list = [os.path.join(input_dir,i) for i in os.listdir(input_dir) if i.endswith('.pkl')]

    df_list = []
    for i in id_list:

        df_list.append(pd.read_pickle(i))


    return pd.concat(df_list)




