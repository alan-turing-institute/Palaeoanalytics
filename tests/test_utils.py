"""
Test utils
"""
import os

import cv2
import numpy as np
import pandas as pd
import yaml

from pylithics.src.read_and_process import read_image, detect_lithic, process_image, find_lithic_contours, find_arrows
from pylithics.src.utils import mask_image, contour_characterisation, classify_distributions, shape_detection, \
    get_high_level_parent_and_hierarchy, pixulator, classify_surfaces, subtract_masked_image, measure_vertices, \
    get_angles, \
    measure_arrow_angle, contour_selection

# Global loads for all tests
image_array = read_image(os.path.join('tests', 'test_images'), 'test')
filename_config = os.path.join('tests', 'test_config.yml')

# Read YAML file
with open(filename_config, 'r') as config_file:
    config_file = yaml.load(config_file)
config_file['conversion_px'] = 0.1  # hardcoded for now


def test_mask_image():
    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    contours_cv, hierarchy = cv2.findContours(binary_edge_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cont = np.asarray([i[0] for i in list(contours_cv)[1]])

    masked_image = mask_image(image_array, cont)

    assert masked_image.shape != (0, 0)
    assert masked_image.sum() < binary_edge_sobel.sum()


def test_contour_characterisation():
    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    contours_cv, hierarchy = cv2.findContours(binary_edge_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cont = np.asarray([i[0] for i in list(contours_cv)[2]])

    cont_info = contour_characterisation(image_array, cont, config_file['conversion_px'])

    assert cont_info['length'] == 2965
    assert cont_info['area_px'] == 490095.5
    assert cont_info['height_px'] == 1339
    assert cont_info['width_px'] == 539
    assert cont_info['centroid'] == (930.1856303869774, 1362.474110977076)
    assert cont_info['area_mm'] == 4901.0
    assert cont_info['width_mm'] == 53.9
    assert cont_info['height_mm'] == 133.9
    assert cont_info['polygon_count'] == 7


def test_classify_distributions():
    image_processed = process_image(image_array, config_file)

    is_narrow = classify_distributions(image_processed)

    assert is_narrow == True


def test_get_high_level_parent_and_hierarchy():
    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    contours_cv, hierarchy = cv2.findContours(binary_edge_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    parent_index, hierarchy_level = get_high_level_parent_and_hierarchy(list(hierarchy)[0])

    assert len(parent_index) != 0
    assert len(hierarchy_level) != 0


def test_pixulator():
    image_scale_array = read_image(os.path.join('tests', 'test_images'), 'sc_1')

    conversion = pixulator(image_scale_array, 5)

    assert conversion == 0.00423728813559322


def test_classify_surfaces():
    # initial processing of the image
    image_processed = process_image(image_array, config_file)

    # processing to detect lithic and scars
    binary_array, threshold_value = detect_lithic(image_processed, config_file)

    # find contours
    contours = find_lithic_contours(binary_array, config_file)

    surfaces_classification = classify_surfaces(contours)

    assert surfaces_classification == {0: 'Ventral', 1: 'Dorsal', 2: 'Lateral', 3: 'Platform'}


def test_subtract_masked_image():
    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    contours_cv, hierarchy = cv2.findContours(binary_edge_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cont = np.asarray([i[0] for i in list(contours_cv)[1]])

    rows, columns = subtract_masked_image(mask_image(image_array, cont))

    assert len(rows) == 1835
    assert len(columns) == 1659


def test_get_angles():
    # initial processing of the image
    image_processed = process_image(image_array, config_file)

    # get the templates for the arrows
    templates = find_arrows(image_array, image_processed)

    # measure angles for existing arrows
    arrow_df = get_angles(templates)

    assert arrow_df.shape[0] == 4
    assert arrow_df.shape[1] == 2


def test_contour_selection():
    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    contours_cv, hierarchy = cv2.findContours(image_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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

    df_cont_info = pd.DataFrame.from_dict(cont_info_list)

    df_cont_info['parent_index'], df_cont_info['hierarchy_level'] = get_high_level_parent_and_hierarchy(
        df_cont_info['hierarchy'].values)

    indexes = contour_selection(df_cont_info)

    assert len(indexes) > 1
    assert len(indexes) < 20


def test_measure_arrow_angle():
    # initial processing of the image
    image_processed = process_image(image_array, config_file)

    # get the templates for the arrows
    templates = find_arrows(image_array, image_processed)

    angle = measure_arrow_angle(templates[0])

    # TODO: Once the angle measurement is fixed we need to change this test to the actual value.
    assert angle != 0.0


def test_measure_vertices():
    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    contours_cv, hierarchy = cv2.findContours(binary_edge_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cont = np.asarray([i[0] for i in list(contours_cv)[1]])

    vertices, approx = measure_vertices(cont)

    assert vertices == 6


def test_shape_detection():
    config_file['conversion_px'] = 0.1  # hardcoded for now

    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    contours_cv, hierarchy = cv2.findContours(binary_edge_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cont = np.asarray([i[0] for i in list(contours_cv)[2]])

    shape = shape_detection(cont)

    assert shape == ('arrow', 4)
