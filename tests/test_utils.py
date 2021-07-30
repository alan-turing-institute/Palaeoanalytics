"""
Test full pipeline
"""
from pylithics.src.read_and_process import read_image, detect_lithic, process_image, find_lithic_contours
from pylithics.src.utils import mask_image, contour_characterisation, classify_distributions, shape_detection,\
    get_high_level_parent_and_hierarchy, pixulator, classify_surfaces, subtract_masked_image, measure_vertices
import os
import cv2
import numpy as np
import yaml

def test_mask_image():

    image_array = read_image(os.path.join('tests', 'test_images'), '236')


    filename_config = os.path.join('tests', 'test_config.yml')

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)
    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    _, contours_cv, hierarchy = cv2.findContours(binary_edge_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cont = np.asarray([i[0] for i in list(contours_cv)[1]])

    masked_image = mask_image(image_array, cont)

    assert masked_image.shape != (0,0)
    assert masked_image.sum()<binary_edge_sobel.sum()


def test_contour_characterisation():
    image_array = read_image(os.path.join('tests', 'test_images'), '236')

    filename_config = os.path.join('tests', 'test_config.yml')

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)
    config_file['conversion_px'] = 0.1  # hardcoded for now

    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    _, contours_cv, hierarchy = cv2.findContours(binary_edge_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cont = np.asarray([i[0] for i in list(contours_cv)[2]])

    cont_info = contour_characterisation(image_array, cont, config_file['conversion_px'])

    assert cont_info['length'] == 4
    assert cont_info['area_px'] == 2.0
    assert cont_info['height_px'] == 3
    assert cont_info['width_px'] == 3
    assert cont_info['centroid'] == (584.0, 669.0)
    assert cont_info['area_mm'] == 0.0
    assert cont_info['width_mm'] == 0.3
    assert cont_info['height_mm'] == 0.3
    assert cont_info['polygon_count'] == 4


def test_classify_distributions():

    id = '236'
    image_array = read_image(os.path.join('tests', 'test_images'), id)
    filename_config = os.path.join('tests', 'test_config.yml')

    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)
    config_file['conversion_px'] = 0.1  # hardcoded for now

    image_processed = process_image(image_array, config_file)

    is_narrow = classify_distributions(image_processed)

    assert is_narrow == True

def test_get_high_level_parent_and_hierarchy():

    image_array = read_image(os.path.join('tests', 'test_images'), '236')

    filename_config = os.path.join('tests', 'test_config.yml')

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)
    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    _, contours_cv, hierarchy = cv2.findContours(binary_edge_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    parent_index, hierarchy_level = get_high_level_parent_and_hierarchy(list(hierarchy)[0])

    assert len(parent_index) != 0
    assert len(hierarchy_level) != 0

def test_pixulator():
    image_scale_array = read_image(os.path.join('tests', 'test_images'), 'sc_1')

    conversion = pixulator(image_scale_array, 5)

    assert conversion == 0.00423728813559322

def test_classify_surfaces():


    id = '234'
    image_array = read_image(os.path.join('tests', 'test_images'),id)

    filename_config = os.path.join('tests', 'test_config.yml')

    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)
    config_file['conversion_px'] = 0.1  # hardcoded for now
    config_file['id'] = id  # hardcoded for now

    # initial processing of the image
    image_processed = process_image(image_array, config_file)

    # processing to detect lithic and scars
    binary_array, threshold_value = detect_lithic(image_processed, config_file)

    # find contours
    contours = find_lithic_contours(binary_array, config_file)

    surfaces_classification = classify_surfaces(contours)

    assert surfaces_classification == {0: 'Ventral', 1: 'Dorsal', 2: 'Lateral'}


def test_subtract_masked_image():
    image_array = read_image(os.path.join('tests', 'test_images'), '236')

    filename_config = os.path.join('tests', 'test_config.yml')

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)
    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    _, contours_cv, hierarchy = cv2.findContours(binary_edge_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cont = np.asarray([i[0] for i in list(contours_cv)[1]])

    rows, columns = subtract_masked_image(mask_image(image_array, cont))

    assert len(rows) == 688
    assert len(columns) == 1380

def test_template_matching():
    assert True

def test_get_angles():

    assert True

def test_contour_selection():

    assert True

def test_measure_arrow_angle():

    assert True

def test_measure_vertices():

    image_array = read_image(os.path.join('tests', 'test_images'), '236')

    filename_config = os.path.join('tests', 'test_config.yml')

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)
    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    _, contours_cv, hierarchy = cv2.findContours(binary_edge_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cont = np.asarray([i[0] for i in list(contours_cv)[1]])

    vertices, approx = measure_vertices(cont)

    assert vertices == 1


def test_shape_detection():
    image_array = read_image(os.path.join('tests', 'test_images'), '236')

    filename_config = os.path.join('tests', 'test_config.yml')

    # Read YAML file
    with open(filename_config, 'r') as config_file:
        config_file = yaml.load(config_file)
    config_file['conversion_px'] = 0.1  # hardcoded for now

    binary_edge_sobel, _ = detect_lithic(image_array, config_file)

    _, contours_cv, hierarchy = cv2.findContours(binary_edge_sobel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cont = np.asarray([i[0] for i in list(contours_cv)[2]])

    shape = shape_detection(cont)

    assert shape == ('square', 4)








