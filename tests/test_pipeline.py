"""
Test full pipeline
"""
import os
import yaml
from pylithics.src.read_and_process import read_image, detect_lithic, \
    find_lithic_contours, process_image, get_scars_angles, data_output, find_arrows
from pylithics.src.utils import get_angles, complexity_estimator


def test_pipeline():
    id = 'test'
    image_array = read_image(os.path.join('tests', 'test_images'), id)

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

    # add complexity measure
    contours = complexity_estimator(contours)

    # in case we dont have arrows
    contours = get_scars_angles(image_processed, contours)

    # save data into a .json file
    json_output = data_output(contours, config_file)

    assert len(json_output) == 4
    assert contours.shape == (11, 17)
    assert binary_array.shape == (1841, 1665)
    assert len(json_output['lithic_contours']) == 4


def test_arrow_pipeline():
    id = 'test'
    image_array = read_image(os.path.join('tests', 'test_images'), id)

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

    # add complexity measure
    contours = complexity_estimator(contours)

    # get the templates for the arrows
    templates = find_arrows(image_array, image_processed, False)

    # measure angles for existing arrows
    arrow_df = get_angles(templates)

    # associate arrows to scars, add that info into the contour
    contours_final = get_scars_angles(image_processed, contours, arrow_df)

    assert len(templates) == 4
    assert contours_final.shape == (11, 15)
    assert arrow_df.shape == (4, 2)
