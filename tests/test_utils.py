"""
Test full pipeline
"""
from pylithics.src.read_and_process import read_image, detect_lithic, process_image
from pylithics.src.utils import mask_image, contour_characterisation, classify_distributions
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





